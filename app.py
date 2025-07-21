from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

app = Flask(__name__)

# Load dataset
#this is the new line
try:
    df = pd.read_csv('dataset.csv')
    if not {'text', 'label'}.issubset(df.columns):
        raise ValueError("dataset.csv must have 'text' and 'label' columns")
    if not all(df['label'].isin(['CG', 'OR'])):
        raise ValueError("Labels must be 'CG' or 'OR'")
except FileNotFoundError:
    print("Error: dataset.csv not found.")
    exit(1)

# Train model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('clf', LogisticRegression())
])
pipeline.fit(df['text'], df['label'])

# Initialize Selenium driver
def init_driver():
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run without browser UI
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    return driver

# Fetch reviews with Selenium
def fetch_reviews(product_link):
    driver = init_driver()
    reviews = []
    try:
        driver.get(product_link)
        # Wait for page to load
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        time.sleep(2)  # Extra wait for dynamic content

        # Platform-specific logic (Amazon or Flipkart)
        if 'amazon' in product_link.lower():
            # Amazon: Navigate to reviews section
            try:
                see_all_reviews = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.ID, 'reviews-medley-footer'))
                )
                see_all_reviews.find_element(By.TAG_NAME, 'a').click()
                time.sleep(2)
                # Extract reviews
                review_elements = driver.find_elements(By.CSS_SELECTOR, 'span[data-hook="review-body"] span')
                reviews = [elem.text.strip() for elem in review_elements if elem.text.strip()]
            except Exception as e:
                print(f"Amazon error: {e}")
                reviews = ['Could not fetch Amazon reviews.']

        elif 'flipkart' in product_link.lower():
            # Flipkart: Navigate to reviews
            try:
                # Scroll to load reviews
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                # Click "All reviews" if available
                try:
                    all_reviews = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.CLASS_NAME, '_27M2h_'))
                    )
                    all_reviews.click()
                    time.sleep(2)
                except:
                    pass
                # Extract reviews
                review_elements = driver.find_elements(By.CSS_SELECTOR, 'div.t-ZTKy div div')
                reviews = [elem.text.strip() for elem in review_elements if elem.text.strip()]
            except Exception as e:
                print(f"Flipkart error: {e}")
                reviews = ['Could not fetch Flipkart reviews.']

        else:
            reviews = ['Unsupported website. Try Amazon or Flipkart.']

        # Limit to 10 reviews to avoid overload
        reviews = reviews[:10] if reviews else ['No reviews found.']

    except Exception as e:
        print(f"Error fetching reviews: {e}")
        reviews = ['Error fetching reviews.']

    finally:
        driver.quit()

    return reviews

# Clean review text
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower().strip()
    return text

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Check reviews route
@app.route('/check', methods=['POST'])
def check_reviews():
    product_link = request.form['product_link']
    
    # Fetch reviews
    reviews = fetch_reviews(product_link)
    
    # Clean and classify
    cleaned_reviews = [clean_text(review) for review in reviews]
    predictions = pipeline.predict(cleaned_reviews)
    
    # Prepare results
    results = []
    for review, pred in zip(reviews, predictions):
        results.append({
            'text': review,
            'label': pred,
            'color': 'red' if pred == 'CG' else 'green'
        })
    
    return render_template('index.html', results=results, product_link=product_link)

if __name__ == '__main__':
    app.run(debug=True)