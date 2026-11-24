"""
================================================================================
Email Spam Detection - Model Training Script
================================================================================
This script:
1. Loads the email/SMS dataset (spam.csv)
2. Cleans and prepares the data
3. Converts text into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency)
4. Trains a Multinomial Naive Bayes classifier (and compares with Logistic Regression)
5. Evaluates model performance (Accuracy, Precision, Recall, F1 Score)
6. Saves the trained model pipeline for instant predictions
================================================================================
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def load_data(filepath='spam.csv'):
    """Loads and cleans the spam dataset."""
    print(f"📂 Loading dataset from '{filepath}'...")
    
    # The dataset uses latin-1 encoding and has extra unnamed columns
    df = pd.read_csv(filepath, encoding='latin-1')
    
    # Keep only the first two columns (v1 = label, v2 = message text)
    df = df.iloc[:, [0, 1]]
    df.columns = ['label', 'text']
    
    # Remove any missing values
    df = df.dropna()
    
    # Clean whitespace
    df['text'] = df['text'].str.strip()
    
    # Map labels: ham -> 0 (Legitimate), spam -> 1 (Spam)
    df['target'] = df['label'].map({'ham': 0, 'spam': 1})
    
    total = len(df)
    spam_count = (df['target'] == 1).sum()
    ham_count = (df['target'] == 0).sum()
    
    print(f"✅ Loaded {total:,} messages successfully!")
    print(f"   • Ham (Legitimate): {ham_count:,} ({ham_count / total * 100:.1f}%)")
    print(f"   • Spam:             {spam_count:,} ({spam_count / total * 100:.1f}%)")
    
    return df

def train_and_evaluate(df):
    """Splits data, trains models, evaluates metrics, and returns the best model."""
    print("\n✂️  Splitting data into 80% Training and 20% Testing sets...")
    X = df['text']
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"   • Training samples: {len(X_train):,}")
    print(f"   • Testing samples:  {len(X_test):,}")

    # Build Pipeline: TF-IDF Vectorizer + Multinomial Naive Bayes
    # TF-IDF converts text into numerical weights:
    # - Words frequent in one email but rare across the dataset get higher weight.
    # - Common English stopwords ('the', 'is', 'at') are removed.
    print("\n🧠 Training Multinomial Naive Bayes Classifier...")
    pipeline_nb = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),  # Consider single words and 2-word phrases
            max_features=5000
        )),
        ('classifier', MultinomialNB(alpha=0.2))  # Additive smoothing
    ])
    
    pipeline_nb.fit(X_train, y_train)
    y_pred_nb = pipeline_nb.predict(X_test)
    
    # Also test Logistic Regression for comparison
    print("🧠 Training Logistic Regression for comparison...")
    pipeline_lr = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),
            max_features=5000
        )),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipeline_lr.fit(X_train, y_train)
    y_pred_lr = pipeline_lr.predict(X_test)

    # Calculate metrics
    def get_metrics(y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'confusion': confusion_matrix(y_true, y_pred)
        }

    metrics_nb = get_metrics(y_test, y_pred_nb)
    metrics_lr = get_metrics(y_test, y_pred_lr)

    # Print Comparison Table
    print("\n" + "=" * 65)
    print("             📊 MODEL EVALUATION & COMPARISON")
    print("=" * 65)
    print(f"{'Metric':<18} | {'Naive Bayes':<18} | {'Logistic Regression':<18}")
    print("-" * 65)
    print(f"{'Accuracy':<18} | {metrics_nb['accuracy'] * 100:>16.2f}% | {metrics_lr['accuracy'] * 100:>16.2f}%")
    print(f"{'Precision (Spam)':<18} | {metrics_nb['precision'] * 100:>16.2f}% | {metrics_lr['precision'] * 100:>16.2f}%")
    print(f"{'Recall (Spam)':<18} | {metrics_nb['recall'] * 100:>16.2f}% | {metrics_lr['recall'] * 100:>16.2f}%")
    print(f"{'F1 Score':<18} | {metrics_nb['f1'] * 100:>16.2f}% | {metrics_lr['f1'] * 100:>16.2f}%")
    print("=" * 65)

    tn, fp, fn, tp = metrics_nb['confusion'].ravel()
    print("\n📌 Naive Bayes Confusion Matrix Details:")
    print(f"   • True Negatives (Ham correctly identified):  {tn}")
    print(f"   • False Positives (Ham mistaken for Spam):   {fp}  <-- Lower is critical!")
    print(f"   • False Negatives (Spam missed):             {fn}")
    print(f"   • True Positives (Spam caught):              {tp}")

    print("\n📋 Detailed Classification Report (Naive Bayes):")
    print(classification_report(y_test, y_pred_nb, target_names=['Ham (0)', 'Spam (1)']))

    # We choose Naive Bayes as our primary spam detection model
    return pipeline_nb, metrics_nb

def save_model(pipeline, filepath='model.pkl'):
    """Saves the trained pipeline to disk."""
    joblib.dump(pipeline, filepath)
    print(f"💾 Trained model saved successfully to '{filepath}'!")

def run_sample_tests(pipeline):
    """Runs a quick sanity check on a few realistic test emails."""
    print("\n🧪 Running quick sanity tests on sample inputs:")
    test_cases = [
        ("Congratulations! You won a $1,000 Walmart Gift Card! Click here to claim your prize now.", "Spam"),
        ("Hey, are we still meeting for lunch at 12:30 tomorrow?", "Ham"),
        ("URGENT: Your bank account has been suspended. Verify your credentials immediately.", "Spam"),
        ("Can you please send me the updated project report when you get a chance?", "Ham")
    ]
    
    for text, expected in test_cases:
        prediction = pipeline.predict([text])[0]
        proba = pipeline.predict_proba([text])[0]
        label = "🚨 SPAM" if prediction == 1 else "✅ HAM (NOT SPAM)"
        confidence = proba[prediction] * 100
        print(f"\n   Input: '{text}'")
        print(f"   Result: {label} (Confidence: {confidence:.1f}%) [Expected: {expected}]")

if __name__ == '__main__':
    print("=" * 65)
    print("       🚀 EMAIL SPAM DETECTION - MODEL TRAINING 🚀")
    print("=" * 65)
    
    df = load_data('spam.csv')
    best_model, metrics = train_and_evaluate(df)
    save_model(best_model, 'model.pkl')
    run_sample_tests(best_model)
    
    print("\n🎉 Model training and export complete! Run 'python predict.py' or 'streamlit run app.py' to use it.")
