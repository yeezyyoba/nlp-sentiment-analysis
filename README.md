# NLP Sentiment Analysis — Financial News Headlines

Classifying financial news sentences as **positive**, **negative**, or **neutral** using classical NLP techniques and machine learning baselines.

---

## Overview

This project applies NLP to the domain of financial text. Given a sentence like *"The company reported record profits this quarter"*, the goal is to predict market sentiment — a core problem in algorithmic trading, risk assessment, and fintech analytics.

**Dataset:** [Financial PhraseBank](https://huggingface.co/datasets/financial_phrasebank) — ~4,800 English sentences sourced from financial news, annotated by domain experts.

---

## Project Structure

```
nlp-sentiment-analysis/
├── data/                          # Raw data loaded via code (not committed)
├── notebooks/
│   └── sentiment_analysis.ipynb  # Main analysis notebook
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Methodology

### 1. Data Loading & Exploration
- Load dataset via Hugging Face `datasets` library
- Class distribution analysis
- Sample sentence inspection

### 2. Text Preprocessing
- Lowercasing
- Punctuation and special character removal
- Stopword removal (NLTK)
- Lemmatization (WordNetLemmatizer)

### 3. Feature Extraction
- **TF-IDF Vectorizer** (unigrams + bigrams, top 5,000 features)

### 4. Model Training
| Model | Notes |
|---|---|
| Logistic Regression | Strong baseline for text classification |
| Multinomial Naive Bayes | Classic NLP model |

### 5. Evaluation
- Classification report (precision, recall, F1 per class)
- Confusion matrix heatmap
- Error analysis — inspecting misclassified examples

---

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | TBD |
| Multinomial Naive Bayes | TBD |

*Results will be updated after training.*

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/nlp-sentiment-analysis.git
cd nlp-sentiment-analysis

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Download NLTK data (run once)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

jupyter notebook notebooks/sentiment_analysis.ipynb
```

---

## Key Libraries

- `datasets` — Hugging Face dataset loader
- `scikit-learn` — TF-IDF, Logistic Regression, Naive Bayes, evaluation metrics
- `nltk` — Stopwords, lemmatization
- `matplotlib` / `seaborn` — Visualization
- `wordcloud` — Word frequency visualization

---

## Context

This project is part of a broader data science portfolio focused on **fintech analytics for emerging markets**. It complements the [Ethiopian Credit Risk & Fraud Analytics Platform](https://github.com/YOUR_USERNAME/ethiopian-fintech-analytics) by adding NLP capabilities — sentiment in financial text is a key signal in credit risk and fraud detection pipelines.

---

## Author

**Eyob** — Computer Science student, Addis Ababa University  
Software Engineering Intern, IE Networks PLC
