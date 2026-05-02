from flask import Flask, request, jsonify, render_template
import pandas as pd
import requests
import collections
import os

"""
PRODUCT SENTIMENT ANALYZER - VERCEL OPTIMIZED
This version uses the Hugging Face Inference API instead of local transformers
to stay within Vercel's serverless size and memory limits.
"""

app = Flask(__name__)

# SECTION 1: HUGGING FACE INFERENCE API CONFIG
# We offload the AI processing to Hugging Face to keep the deployment tiny (< 100MB).
# Using the modern router endpoint for better reliability.
API_URL = "https://router.huggingface.co/hf-inference/models/distilbert/distilbert-base-uncased-finetuned-sst-2-english"
# If you have a token, add it here in your Vercel Environment Variables as HF_TOKEN
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def query_sentiment_api(payload):
    """Calls the Hugging Face Inference API for sentiment classification."""
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=15)
        # Log error if status is not 200
        if response.status_code != 200:
            print(f"API Error: Status {response.status_code}, Response: {response.text[:200]}")
            return {"error_code": response.status_code, "message": response.text}
        return response.json()
    except Exception as e:
        print(f"Connection Error: {e}")
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
        
    # Reduce sample size to avoid Hugging Face Inference API payload limits/timeouts
    sampled_reviews = matched_reviews.head(15)
    review_texts = sampled_reviews[REVIEW_COLUMN_NAME].astype(str).fillna("").tolist()
    review_texts = [text for text in review_texts if text.strip()]
    
    if not review_texts:
         return jsonify({"error": "Found products, but review text is empty."}), 404

    # CALL HUGGING FACE API (Batch Processing)
    api_results = query_sentiment_api({"inputs": review_texts, "options": {"wait_for_model": True}})
    
    if not api_results or not isinstance(api_results, list):
        # Handle specific error cases
        if isinstance(api_results, dict) and api_results.get("error_code") == 401:
            return jsonify({"error": "Authentication failed. Please set a valid HF_TOKEN in your environment."}), 401
        
        # Fallback if API is slow, errors out, or rate limited
        return jsonify({
            "error": "The sentiment analysis service is currently unavailable or busy.",
            "details": "This can happen if the model is loading or rate limits are reached. Please try again in a few seconds."
        }), 503

    # Extract Product Metadata
    first_row = matched_reviews.iloc[0]
    brand = str(first_row.get('brand', 'Unknown'))
    categories = str(first_row.get('categories', 'N/A'))
    source_url = str(first_row.get('reviews.sourceURLs', ''))
    if source_url and ',' in source_url:
        source_url = source_url.split(',')[0].strip().replace('"', '').replace('[', '').replace(']', '')
    
    # AGGREGATE RESULTS
    # The API returns labels like 'POSITIVE' or 'NEGATIVE' in a nested list/dict
    sentiments = []
    processed_reviews = []
    
    for text, res_list in zip(review_texts, api_results):
        # API usually returns a list of results for each input, pick the highest score
        top_res = max(res_list, key=lambda x: x['score'])
        label = top_res['label']
        score = top_res['score']
        
        sentiments.append(label)
        processed_reviews.append({
            "text": text,
            "sentiment": label,
            "score": round(score, 3)
        })
    
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
        "reviews": processed_reviews
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
