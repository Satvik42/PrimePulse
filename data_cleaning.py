import pandas as pd
import os

"""
DATA CLEANING & PREPROCESSING MODULE
This script handles the preparation of the Amazon Product Review dataset.
"""

def preprocess_data(input_file, output_file):
    print(f"--- Starting Data Cleaning for {input_file} ---")
    
    # 1. Load Data
    # We use low_memory=False to handle mixed types in large datasets
    df = pd.read_csv(input_file, low_memory=False)
    
    # 2. Missing Value Treatment
    # Reviews without text are useless for sentiment analysis, so we drop them
    initial_count = len(df)
    df = df.dropna(subset=['reviews.text', 'name'])
    print(f"Dropped {initial_count - len(df)} rows with missing text or names.")
    
    # 3. Data Normalization
    # Strip whitespace and normalize case for product names to improve search matching
    df['name'] = df['name'].str.strip()
    
    # 4. Feature Engineering
    # Ensure ratings are numeric for statistical calculations
    df['reviews.rating'] = pd.to_numeric(df['reviews.rating'], errors='coerce')
    df = df.dropna(subset=['reviews.rating'])
    
    # 5. Sampling for Performance & Size Limits
    # To keep the GitHub repo under 10MB, we take a representative sample
    # This also ensures the DistilBERT model runs efficiently on standard hardware
    df_sample = df.head(5000) 
    
    # 6. Save Processed Data
    df_sample.to_csv(output_file, index=False)
    print(f"--- Preprocessing Complete. Saved {len(df_sample)} rows to {output_file} ---")

if __name__ == "__main__":
    preprocess_data('reviews.csv', 'reviews_processed.csv')
