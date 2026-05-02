import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np

"""
MODEL EVALUATION MODULE
Evaluates the pre-trained DistilBERT model on the product review dataset.
"""

def evaluate():
    print("--- Loading Model for Evaluation ---")
    # Load the DistilBERT sentiment pipeline
    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    
    # Load data
    df = pd.read_csv('reviews.csv').head(100) # Small sample for quick evaluation
    
    # Prepare true labels
    # Dataset ratings: 4-5 -> POSITIVE, 1-2 -> NEGATIVE, 3 -> IGNORED for binary eval
    df = df[df['reviews.rating'] != 3]
    y_true = df['reviews.rating'].apply(lambda x: 'POSITIVE' if x > 3 else 'NEGATIVE').tolist()
    texts = df['reviews.text'].tolist()
    
    print(f"--- Running Inference on {len(texts)} samples ---")
    results = sentiment_pipeline(texts, truncation=True)
    y_pred = [res['label'] for res in results]
    
    # Calculate Metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label='POSITIVE')
    cm = confusion_matrix(y_true, y_pred, labels=['POSITIVE', 'NEGATIVE'])
    
    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Positive): {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    evaluate()
