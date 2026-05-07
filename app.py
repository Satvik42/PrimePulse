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
        
    TOTAL_SAMPLE_SIZE = 150
    API_BATCH_SIZE = 15

    # Sample reviews randomly to get a more representative sentiment distribution
    sampled_reviews = matched_reviews.sample(min(TOTAL_SAMPLE_SIZE, len(matched_reviews)))
    
    # Filter out empty reviews while preserving index alignment
    valid_reviews = sampled_reviews[sampled_reviews[REVIEW_COLUMN_NAME].astype(str).str.strip() != ""].copy()
    
    if valid_reviews.empty:
         return jsonify({"error": "Found products, but review text is empty."}), 404

    # Extract product info for the dashboard
    first_row = matched_reviews.iloc[0]
    brand = str(first_row.get('brand', 'Generic'))
    categories = str(first_row.get('categories', 'Product'))
    source_url = str(first_row.get('reviews.sourceURLs', '#')).split(',')[0].strip() # Take first URL if multiple

    # Create numeric rating column for filtering
    valid_reviews['numeric_rating'] = pd.to_numeric(valid_reviews[RATING_COLUMN_NAME], errors='coerce').fillna(5)
    
    # Attempt to get a balanced mix of negative/positive reviews for the API (UI Highlights)
    neg_reviews = valid_reviews[valid_reviews['numeric_rating'] <= 3]
    pos_reviews = valid_reviews[valid_reviews['numeric_rating'] >= 4]

    neg_sample = neg_reviews.head(7)
    pos_sample = pos_reviews.head(API_BATCH_SIZE - len(neg_sample))
    
    # If we don't have enough positive, fill with negative and vice versa just in case
    if len(pos_sample) + len(neg_sample) < API_BATCH_SIZE and len(valid_reviews) >= API_BATCH_SIZE:
        api_batch_df = valid_reviews.head(API_BATCH_SIZE)
    else:
        api_batch_df = pd.concat([neg_sample, pos_sample]).sample(frac=1) # Shuffle
    
    api_indices = api_batch_df.index
    local_batch_df = valid_reviews.drop(api_indices)

    api_batch_texts = api_batch_df[REVIEW_COLUMN_NAME].astype(str).tolist()
    api_batch_ratings = api_batch_df['numeric_rating'].tolist()
    
    local_batch_texts = local_batch_df[REVIEW_COLUMN_NAME].astype(str).tolist()
    local_batch_ratings = local_batch_df['numeric_rating'].tolist()
    
    # CALL HUGGING FACE API (Batch Processing) on smaller subset
    api_results = query_sentiment_api({"inputs": api_batch_texts, "options": {"wait_for_model": True}})
    
    using_fallback = False
    processed_reviews = []
    sentiments = []

    # Process API Results for the first batch
    if api_results and isinstance(api_results, list) and len(api_results) > 0 and isinstance(api_results[0], list):
        # NORMAL API PROCESSING
        for text, res_list in zip(api_batch_texts, api_results):
            top_res = max(res_list, key=lambda x: x['score'])
            label = top_res['label'].upper()
            if label in ['LABEL_2', 'POSITIVE', 'POS']: label = 'POSITIVE'
            elif label in ['LABEL_0', 'NEGATIVE', 'NEG']: label = 'NEGATIVE'
            else: label = 'POSITIVE'
            
            score = top_res['score']
            sentiments.append(label)
            processed_reviews.append({"text": text, "sentiment": label, "score": round(score, 3)})
    else:
        print("API Failed or timed out. Using local fallback analysis for all reviews.")
        using_fallback = True
        for i, text in enumerate(api_batch_texts):
            rating = api_batch_ratings[i]
            
            if rating >= 4:
                label, score = 'POSITIVE', 0.95
            elif 0 < rating <= 3:
                label, score = 'NEGATIVE', 0.95
            else:
                label, score = local_sentiment_fallback(text)
                label = label.upper()
            
            sentiments.append(label)
            processed_reviews.append({"text": text, "sentiment": label, "score": round(score, 3)})

    # Process remaining reviews locally (hybrid approach)
    for i, text in enumerate(local_batch_texts):
        rating = local_batch_ratings[i]
        
        if rating >= 4:
            label = 'POSITIVE'
        elif 0 < rating <= 3:
            label = 'NEGATIVE'
        else:
            label, _ = local_sentiment_fallback(text)
            label = label.upper()
        
        sentiments.append(label)
    
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
