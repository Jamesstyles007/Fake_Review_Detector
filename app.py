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


try:
    df = pd.read_csv('dataset.csv')
    if not {'text', 'label'}.issubset(df.columns):
        raise ValueError("dataset.csv must have 'text' and 'label' columns")
    if not all(df['label'].isin(['CG', 'OR'])):
        raise ValueError("Labels must be 'CG' or 'OR'")
except FileNotFoundError:
    print("Error: dataset.csv not found.")
    exit(1)


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('clf', LogisticRegression())
])
pipeline.fit(df['text'], df['label'])


def init_driver():
    chrome_options = Options()
    chrome_options.add_argument('--headless')  
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    return driver


def fetch_reviews(product_link):
    driver = init_driver()
    reviews = []
    try:
        driver.get(product_link)
       
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        time.sleep(2) 

        
        if 'amazon' in product_link.lower():
           
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
           
            try:
                
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                
                try:
                    all_reviews = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.CLASS_NAME, '_27M2h_'))
                    )
                    all_reviews.click()
                    time.sleep(2)
                except:
                    pass
            
                review_elements = driver.find_elements(By.CSS_SELECTOR, 'div.t-ZTKy div div')
                reviews = [elem.text.strip() for elem in review_elements if elem.text.strip()]
            except Exception as e:
                print(f"Flipkart error: {e}")
                reviews = ['Could not fetch Flipkart reviews.']

        else:
            reviews = ['Unsupported website. Try Amazon or Flipkart.']

        
        reviews = reviews[:10] if reviews else ['No reviews found.']

    except Exception as e:
        print(f"Error fetching reviews: {e}")
        reviews = ['Error fetching reviews.']

    finally:
        driver.quit()

    return reviews


def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower().strip()
    return text


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/check', methods=['POST'])
def check_reviews():
    product_link = request.form['product_link']
    
 
    reviews = fetch_reviews(product_link)
    

    cleaned_reviews = [clean_text(review) for review in reviews]
    predictions = pipeline.predict(cleaned_reviews)
    

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