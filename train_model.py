"""
Phishing Detection Model Training Script
This script trains a Logistic Regression model to detect phishing URLs
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

def main():
    print("=" * 60)
    print("AI-BASED PHISHING DETECTION - MODEL TRAINING")
    print("=" * 60)
    
    # Step 1: Load Dataset
    print("\n[1/6] Loading dataset...")
    try:
        df = pd.read_csv('phishing_dataset.csv')
        print(f"✓ Dataset loaded successfully!")
        print(f"  - Total URLs: {len(df)}")
        print(f"  - Phishing URLs: {len(df[df['label'] == 1])}")
        print(f"  - Safe URLs: {len(df[df['label'] == 0])}")
    except FileNotFoundError:
        print("✗ Error: phishing_dataset.csv not found!")
        print("  Make sure the CSV file is in the same folder as this script.")
        return
    
    # Step 2: Data Preprocessing
    print("\n[2/6] Preprocessing data...")
    # Remove duplicates
    df = df.drop_duplicates()
    # Remove missing values
    df = df.dropna()
    print(f"✓ Data cleaned!")
    print(f"  - Remaining URLs after cleaning: {len(df)}")
    
    # Check if we have enough data
    if len(df) < 20:
        print("✗ Error: Not enough data! Need at least 20 URLs.")
        return
    
    # Step 3: Split Data
    print("\n[3/6] Splitting data into training and testing sets...")
    X = df['url']  # URLs (features)
    y = df['label']  # Labels (0=safe, 1=phishing)
    
    # 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Data split complete!")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Testing samples: {len(X_test)}")
    
    # Step 4: Feature Extraction (TF-IDF)
    print("\n[4/6] Extracting features using TF-IDF...")
    print("  (Converting URLs into numbers the AI can understand...)")
    
    # TF-IDF Vectorizer
    # analyzer='char' means it looks at character patterns
    # ngram_range=(2,5) means it looks at 2-5 character combinations
    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 5),
        max_features=3000
    )
    
    # Transform training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    # Transform testing data (using same vectorizer)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"✓ Features extracted!")
    print(f"  - Feature dimensions: {X_train_tfidf.shape[1]}")
    
    # Step 5: Train Model
    print("\n[5/6] Training Logistic Regression model...")
    print("  (The AI is learning patterns from phishing URLs...)")
    
    model = LogisticRegression(
        max_iter=1000,  # Number of iterations
        random_state=42,
        solver='lbfgs'
    )
    
    model.fit(X_train_tfidf, y_train)
    print("✓ Model trained successfully!")
    
    # Step 6: Evaluate Model
    print("\n[6/6] Evaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE RESULTS")
    print("=" * 60)
    print(f"\n✓ Accuracy: {accuracy * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Safe', 'Phishing'],
                                digits=3))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"  True Safe:     {cm[0][0]} (correctly identified as safe)")
    print(f"  False Phishing: {cm[0][1]} (safe marked as phishing)")
    print(f"  False Safe:    {cm[1][0]} (phishing marked as safe) ⚠️")
    print(f"  True Phishing:  {cm[1][1]} (correctly identified as phishing)")
    
    # Save Model and Vectorizer
    print("\n" + "=" * 60)
    print("SAVING MODEL AND VECTORIZER")
    print("=" * 60)
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✓ Model saved as 'model.pkl'")
    
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("✓ Vectorizer saved as 'vectorizer.pkl'")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE! 🎉")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run: python app.py")
    print("2. Open browser: http://127.0.0.1:5000")
    print("3. Test with phishing and safe URLs!")
    print("\n")

if __name__ == "__main__":
    main()
