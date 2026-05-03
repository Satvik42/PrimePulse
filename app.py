import pandas as pd
import requests
import collections
import os
import time
import re
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, render_template, request, jsonify


"""
PRODUCT SENTIMENT ANALYZER - VERCEL OPTIMIZED
This version uses the Hugging Face Inference API instead of local transformers
to stay within Vercel's serverless size and memory limits.
"""

app = Flask(__name__)

# SECTION 1: HUGGING FACE INFERENCE API CONFIG
# Using a more stable and popular model: twitter-roberta-base-sentiment-latest
API_URL = "https://router.huggingface.co/hf-inference/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def local_sentiment_fallback(text):
    """Simple rule-based sentiment analysis as a fallback."""
    pos_words = {'good', 'great', 'excellent', 'love', 'perfect', 'amazing', 'best', 'awesome', 'happy', 'satisfied', 'nice', 'helpful', 'fast', 'easy'}
    neg_words = {'bad', 'terrible', 'awful', 'hate', 'worst', 'poor', 'disappointed', 'broke', 'broken', 'expensive', 'useless', 'slow', 'waste'}
    
    text_lower = text.lower()
    pos_count = sum(1 for word in pos_words if word in text_lower)
    neg_count = sum(1 for word in neg_words if word in text_lower)
    
    if pos_count > neg_count:
        return 'positive', min(0.5 + (pos_count - neg_count) * 0.1, 0.99)
    elif neg_count > pos_count:
        return 'negative', min(0.5 + (neg_count - pos_count) * 0.1, 0.99)
    else:
        # For neutral/ambiguous text, default to positive with low confidence
        return 'positive', 0.5

def query_sentiment_api(payload, retries=2):
    """Calls the Hugging Face Inference API with retries for loading models."""
    for attempt in range(retries + 1):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=20)
            result = response.json()
            
            if response.status_code == 200:
                return result
            
            # If model is loading, wait and retry
            if response.status_code == 503 and 'loading' in str(result).lower():
                wait_time = result.get('estimated_time', 10)
                print(f"Model loading... waiting {wait_time}s (Attempt {attempt+1}/{retries+1})")
                time.sleep(min(wait_time, 15))
                continue
                
            print(f"API Error: Status {response.status_code}, Response: {response.text[:200]}")
            return {"error_code": response.status_code, "message": response.text}
            
        except Exception as e:
            print(f"Connection Error on attempt {attempt+1}: {e}")
            if attempt < retries:
                time.sleep(2)
                continue
            return None
    return None

# SECTION 2: DATASET CONFIGURATION
CSV_FILENAME = "reviews.csv" 
PRODUCT_COLUMN_NAME = "name"
REVIEW_COLUMN_NAME = "reviews.text"
RATING_COLUMN_NAME = "reviews.rating"

print(f"Loading dataset from {CSV_FILENAME}...")
try:
    if os.path.exists(CSV_FILENAME):
        df = pd.read_csv(CSV_FILENAME, low_memory=False)
        print(f"Successfully loaded {len(df)} reviews from {CSV_FILENAME}")
    else:
        df = pd.DataFrame(columns=[PRODUCT_COLUMN_NAME, REVIEW_COLUMN_NAME, RATING_COLUMN_NAME])
except Exception as e:
    df = pd.DataFrame(columns=[PRODUCT_COLUMN_NAME, REVIEW_COLUMN_NAME, RATING_COLUMN_NAME])

# SECTION 3: ROUTES
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    product_query = data.get('product', '').strip()
    
    if not product_query:
        return jsonify({"error": "Please enter a product name."}), 400
        
    mask = df[PRODUCT_COLUMN_NAME].astype(str).str.contains(product_query, case=False, na=False)
    matched_reviews = df[mask]
    
    if matched_reviews.empty:
        return jsonify({"error": f"No reviews found for '{product_query}'."}), 404
        
    # Sample reviews randomly to get a more representative sentiment distribution
    sampled_reviews = matched_reviews.sample(min(15, len(matched_reviews)))
    review_texts = sampled_reviews[REVIEW_COLUMN_NAME].astype(str).fillna("").tolist()
    review_texts = [text for text in review_texts if text.strip()]
    
    if not review_texts:
         return jsonify({"error": "Found products, but review text is empty."}), 404

    # Extract product info for the dashboard
    first_row = matched_reviews.iloc[0]
    brand = str(first_row.get('brand', 'Generic'))
    categories = str(first_row.get('categories', 'Product'))
    source_url = str(first_row.get('reviews.sourceURLs', '#')).split(',')[0].strip() # Take first URL if multiple

    # CALL HUGGING FACE API (Batch Processing)
    api_results = query_sentiment_api({"inputs": review_texts, "options": {"wait_for_model": True}})
    
    using_fallback = False
    processed_reviews = []
    sentiments = []

    if api_results and isinstance(api_results, list) and len(api_results) > 0 and isinstance(api_results[0], list):
        # NORMAL API PROCESSING
        for text, res_list in zip(review_texts, api_results):
            # Roberta model returns labels like 'positive', 'neutral', 'negative'
            top_res = max(res_list, key=lambda x: x['score'])
            label = top_res['label'].upper()
            if label in ['LABEL_2', 'POSITIVE', 'POS']: label = 'POSITIVE'
            elif label in ['LABEL_0', 'NEGATIVE', 'NEG']: label = 'NEGATIVE'
            else: label = 'POSITIVE' # Map neutral to positive
            
            score = top_res['score']
            sentiments.append(label)
            processed_reviews.append({"text": text, "sentiment": label, "score": round(score, 3)})
    
    # If API failed or returned unexpected format, use fallback
    if not processed_reviews:
        print("API Failed or timed out. Using local fallback analysis.")
        using_fallback = True
        for i, text in enumerate(review_texts):
            # 1. Use rating from dataset for high accuracy during fallback
            raw_rating = sampled_reviews.iloc[i].get(RATING_COLUMN_NAME)
            rating = float(raw_rating) if pd.notna(raw_rating) else 0
            
            if rating >= 4:
                label, score = 'POSITIVE', 0.95
            elif 0 < rating <= 2:
                label, score = 'NEGATIVE', 0.95
            else:
                # 2. Keyword matching if rating is neutral (3) or missing (0)
                label, score = local_sentiment_fallback(text)
                label = label.upper()
            
            sentiments.append(label)
            processed_reviews.append({"text": text, "sentiment": label, "score": round(score, 3)})
    
    sentiment_counts = collections.Counter(sentiments)
    total_reviews = len(sentiments)
    positive_count = sentiment_counts.get("POSITIVE", 0)
    negative_count = sentiment_counts.get("NEGATIVE", 0)
    
    percent_positive = round((positive_count / total_reviews) * 100, 1) if total_reviews > 0 else 0
    percent_negative = round((negative_count / total_reviews) * 100, 1) if total_reviews > 0 else 0
    
    avg_rating = matched_reviews[RATING_COLUMN_NAME].mean()
    avg_rating = round(float(avg_rating), 1) if not pd.isna(avg_rating) else 0.0
        
    return jsonify({
        "product_info": {
            "name": product_query,
            "brand": brand if brand != 'nan' else 'Generic',
            "asin": str(first_row.get('asins', 'N/A')),
            "categories": categories if categories != 'nan' else 'Product',
            "source_url": source_url if source_url != 'nan' and source_url != '' else '#'
        },
        "stats": {
            "total": total_reviews,
            "percent_positive": percent_positive,
            "percent_negative": percent_negative,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "average_rating": avg_rating
        },
        "reviews": processed_reviews,
        "using_fallback": using_fallback
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
