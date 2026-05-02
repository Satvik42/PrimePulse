from flask import Flask, request, jsonify, render_template
import pandas as pd
from transformers import pipeline
import collections
import os

"""
PRODUCT SENTIMENT ANALYZER - MAIN APPLICATION
This script initializes the Flask server, loads the NLP model, 
and provides endpoints for product sentiment analysis.
"""

app = Flask(__name__)

# SECTION 1: GLOBAL MODEL LOADING
# We load the model once at startup to avoid overhead on every request.
# Model: distilbert-base-uncased-finetuned-sst-2-english
# Purpose: High-speed, high-accuracy binary sentiment classification.
print("Loading NLP Model...")
sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
print("Model loaded successfully!")

# SECTION 2: DATASET CONFIGURATION
# Load the Amazon reviews dataset using Pandas for efficient searching.
CSV_FILENAME = "reviews.csv" 
PRODUCT_COLUMN_NAME = "name"          # Column for product search
REVIEW_COLUMN_NAME = "reviews.text"   # Column containing review body
RATING_COLUMN_NAME = "reviews.rating" # Column containing numeric rating

print(f"Loading dataset from {CSV_FILENAME}...")
try:
    if os.path.exists(CSV_FILENAME):
        df = pd.read_csv(CSV_FILENAME, low_memory=False)
        print(f"Successfully loaded {len(df)} reviews from {CSV_FILENAME}")
    else:
        print(f"WARNING: {CSV_FILENAME} not found. Creating a blank dataframe.")
        df = pd.DataFrame(columns=[PRODUCT_COLUMN_NAME, REVIEW_COLUMN_NAME, RATING_COLUMN_NAME])
except Exception as e:
    print(f"Error loading CSV: {e}")
    df = pd.DataFrame(columns=[PRODUCT_COLUMN_NAME, REVIEW_COLUMN_NAME, RATING_COLUMN_NAME])

# SECTION 3: ROUTES
@app.route('/')
def home():
    """
    Renders the main dashboard landing page.
    """
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Inference endpoint:
    1. Receives product name from the frontend.
    2. Filters the dataset for matching reviews.
    3. Runs batch inference using the Transformers pipeline.
    4. Aggregates results (stats, product info, sample reviews).
    5. Returns JSON response.
    """
    data = request.get_json()
    product_query = data.get('product', '').strip()
    
    if not product_query:
        return jsonify({"error": "Please enter a product name."}), 400
        
    # DATA SEARCH: Case-insensitive partial string matching
    mask = df[PRODUCT_COLUMN_NAME].astype(str).str.contains(product_query, case=False, na=False)
    matched_reviews = df[mask]
    
    if matched_reviews.empty:
        return jsonify({"error": f"No reviews found for '{product_query}'."}), 404
        
    # PERFORMANCE OPTIMIZATION: Limit to 50 reviews for real-time response
    sampled_reviews = matched_reviews.head(50)
    review_texts = sampled_reviews[REVIEW_COLUMN_NAME].astype(str).fillna("").tolist()
    
    # Filter empty strings
    review_texts = [text for text in review_texts if text.strip()]
    if not review_texts:
         return jsonify({"error": "Found products, but review text is empty."}), 404

    # BATCH INFERENCE: Process reviews in chunks of 16 for speed
    try:
        results = sentiment_pipeline(review_texts, batch_size=16, truncation=True, max_length=512)
    except Exception as e:
        return jsonify({"error": f"Error during model inference: {str(e)}"}), 500
        
    # PRODUCT METADATA EXTRACTION (Glimpse)
    first_row = matched_reviews.iloc[0]
    brand = str(first_row.get('brand', 'Unknown'))
    categories = str(first_row.get('categories', 'N/A'))
    source_url = str(first_row.get('reviews.sourceURLs', ''))
    if source_url and ',' in source_url:
        source_url = source_url.split(',')[0].strip().replace('"', '').replace('[', '').replace(']', '')
    
    # DATA AGGREGATION & STATS
    sentiments = [res['label'] for res in results]
    sentiment_counts = collections.Counter(sentiments)
    total_reviews = len(sentiments)
    positive_count = sentiment_counts.get("POSITIVE", 0)
    negative_count = sentiment_counts.get("NEGATIVE", 0)
    
    percent_positive = round((positive_count / total_reviews) * 100, 1) if total_reviews > 0 else 0
    percent_negative = round((negative_count / total_reviews) * 100, 1) if total_reviews > 0 else 0
    
    # Calculate Average Rating from dataset
    avg_rating = matched_reviews[RATING_COLUMN_NAME].mean()
    avg_rating = round(float(avg_rating), 1) if not pd.isna(avg_rating) else 0.0
    
    # PAYLOAD PREPARATION
    reviews_data = []
    for text, res in zip(review_texts, results):
        reviews_data.append({
            "text": text,
            "sentiment": res['label'],
            "score": round(res['score'], 3)
        })
        
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
        "reviews": reviews_data
    })

if __name__ == '__main__':
    # Launch Flask development server
    app.run(debug=True, port=5000)
