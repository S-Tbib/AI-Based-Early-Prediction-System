"""
Model training module for diabetes prediction
Tests multiple algorithms and saves the best model
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score, classification_report,
    precision_score, recall_score, f1_score,
    confusion_matrix
)

from sklearn.model_selection import cross_val_score

import os
import sys
import joblib
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PROCESSED = BASE_DIR / "data" / "processed" 
MODEL_DIR = BASE_DIR / "model"

# Add root path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def load_processed_data(filepath=DATA_PROCESSED):
    """Load preprocessed data"""
    train_df = pd.read_csv(filepath / "train.csv")
    test_df = pd.read_csv(filepath / "test.csv")
    
    X_train = train_df.drop('Outcome', axis=1)
    y_train = train_df['Outcome']
    X_test = test_df.drop('Outcome', axis=1)
    y_test = test_df['Outcome']
    
    print("Data loaded:")
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def get_models():
    """
    Define models to test
    """
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        ),
        'SVM': SVC(
            kernel='rbf',
            probability=True,
            random_state=42,
            class_weight='balanced'
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB()
    }
    
    return models


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate a model using key metrics (NO DATA LEAKAGE)
    """
    from sklearn.base import clone

    # Cross-validation FIRST (clean evaluation)
    cv_model = clone(model)
    cv_score = cross_val_score(cv_model, X_train, y_train, cv=5).mean()

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model.__class__.__name__}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save figure
    os.makedirs('../reports/figures', exist_ok=True)
    plt.savefig(f'../reports/figures/confusion_matrix_{model.__class__.__name__}.png')
    plt.close()

    # Metrics
    metrics = {
        'Train Accuracy': accuracy_score(y_train, y_train_pred),
        'Test Accuracy': accuracy_score(y_test, y_test_pred),
        'Test Precision': precision_score(y_test, y_test_pred),
        'Test Recall': recall_score(y_test, y_test_pred),
        'Test F1-Score': f1_score(y_test, y_test_pred),
        'Cross-Val Score': cv_score
    }

    return metrics, model


def train_all_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate all models
    """
    models = get_models()
    results = {}
    trained_models = {}
    
    print("=" * 70)
    print(" MODEL TRAINING")
    print("=" * 70)
    
    for name, model in models.items():
        print(f"\n{name}...")
        try:
            metrics, trained_model = evaluate_model(
                model, X_train, y_train, X_test, y_test
            )
            results[name] = metrics
            trained_models[name] = trained_model
            
            print(f"   Train Accuracy: {metrics['Train Accuracy']:.4f}")
            print(f"   Test Accuracy:  {metrics['Test Accuracy']:.4f}")
            print(f"   Test F1-Score:  {metrics['Test F1-Score']:.4f}")
            print(f"   Cross-Val:      {metrics['Cross-Val Score']:.4f}")
            
        except Exception as e:
            print(f"   Error: {e}")
    
    return results, trained_models


def display_comparison(results):
    """
    Display model comparison table
    """
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    df_results = pd.DataFrame(results).T
    df_results = df_results.sort_values('Test Accuracy', ascending=False)
    
    print(df_results.to_string(float_format=lambda x: f'{x:.4f}'))
    
    return df_results


def save_best_model(trained_models, results, X_test, y_test):
    """
    Save the best model based on F1-score
    """
    best_model_name = max(results, key=lambda x: results[x]['Test F1-Score'])
    best_model = trained_models[best_model_name]

    print("\nBest Model Detailed Report:")
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save model
    model_path = MODEL_DIR / "model.pkl"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    
    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best_model_name}")
    print("=" * 70)
    print(f"   Test Accuracy: {results[best_model_name]['Test Accuracy']:.4f}")
    print(f"   Test F1-Score: {results[best_model_name]['Test F1-Score']:.4f}")
    print(f"   Model saved at: {model_path}")
    
    return best_model_name, best_model

def save_model_info(results, best_model_name):
    """
    Save model performance metrics
    """
    df_results = pd.DataFrame(results).T

    # Ensure directory exists
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save file
    output_path = MODEL_DIR / "model_performance.csv"
    df_results.to_csv(output_path)

    print(f"Performance saved: {output_path}")

def train_pipeline():
    """
    Complete training pipeline
    """
    print("=" * 70)
    print(" STARTING TRAINING PIPELINE")
    print("=" * 70)
    
    # 1. Load data
    print("\n[1/4] Loading data...")
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # 2. Train models
    print("\n[2/4] Training models...")
    results, trained_models = train_all_models(X_train, X_test, y_train, y_test)
    
    # 3. Compare models
    print("\n[3/4] Comparing models...")
    df_results = display_comparison(results)
    
    # 4. Save best model
    print("\n[4/4] Saving best model...")
    best_model_name, best_model = save_best_model(
        trained_models, results, X_test, y_test
    )
    
    save_model_info(results, best_model_name)
    
    print("\n" + "=" * 70)
    print(" TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    
    return best_model, results


# Run pipeline
if __name__ == "__main__":
    best_model, results = train_pipeline()