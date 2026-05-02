# Product Sentiment AI 

A professional, real-time product review sentiment analyzer built with Python, Flask, and Deep Learning (Transformers). This tool allows users to search for products, visualize customer sentiment, and get a direct glimpse of product details.

##  Features
- **Real-time AI Analysis**: Uses `DistilBERT` (state-of-the-art NLP) for high-accuracy sentiment classification.
- **Interactive Dashboard**: Visualizes data using `Chart.js` with sleek doughnut charts.
- **Product Glimpse**: Extracts brand, category, and Product ID (ASIN) from the dataset.
- **Direct Product View**: Link directly to the original product page.
- **Premium UI/UX**: Modern dark-mode interface with smooth animations and intuitive layout.

##  Technology Stack
- **Backend**: Flask (Python)
- **NLP**: Hugging Face Transformers (`distilbert-base-uncased-finetuned-sst-2-english`)
- **Data**: Pandas
- **Frontend**: Vanilla HTML5, CSS3, JavaScript
- **Visualization**: Chart.js

##  Installation & Setup

### Prerequisites
- Python 3.8+
- pip

### Cloning the Repository
```bash
git clone https://github.com/Satvik42/Product-review-sentiment-analyser.git
cd Product-review-sentiment-analyser
```

### Installation
1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App
```bash
python app.py
```
Visit `http://localhost:5000` in your browser.

##  Model Evaluation
The project includes an `evaluate_model.py` script to verify the performance of the DistilBERT model on the product review dataset.
- **Accuracy**: ~92% (on SST-2 benchmark)
- **Metrics**: Precision, Recall, and F1-score are calculated and printed during evaluation.

##  Data Preprocessing
The `data_cleaning.py` script documents the cleaning steps:
- Handling missing review text.
- Normalizing product names.
- Extracting metadata (Brand, Categories).
- Sampling for performance optimization.

##  Contributors
The following members have contributed to the core features, data preprocessing, and UI/UX of this project:
- **Satvik** (@Satvik42) - Lead Developer
- **Tanush** (@JustforCode28) - Data Preprocessing & Documentation
- **Varshini** (@varshininisharohith-ui) - Model Evaluation & Metrics
- **Tharun Gowda** (@Tharungowda23) - Installation Guide & Setup

##  License
This project is licensed under the **MIT License**. The license covers all contributions and commits made by the team members listed above.
