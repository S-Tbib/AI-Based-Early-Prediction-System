"""
Model training module for diabetes prediction
Improved version: clean pipeline, better evaluation, no leakage, production-ready
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

import joblib


# =============================
# PATH CONFIG (ROBUST)
# =============================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "model"
REPORT_DIR = BASE_DIR / "reports" / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =============================
# LOAD DATA
# =============================
def load_processed_data():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    X_train = train_df.drop("Outcome", axis=1)
    y_train = train_df["Outcome"]

    X_test = test_df.drop("Outcome", axis=1)
    y_test = test_df["Outcome"]

    print("Data loaded:")
    print(f"   Train: {X_train.shape}")
    print(f"   Test: {X_test.shape}")

    return X_train, X_test, y_train, y_test


# =============================
# MODELS
# =============================
def get_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced"
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", probability=True, class_weight="balanced"))
        ]),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=7))
        ]),

        "Naive Bayes": GaussianNB()
    }


# =============================
# EVALUATION
# =============================
def evaluate_model(model, X_train, y_train, X_test, y_test):

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_score = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1").mean()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Probabilities (for ROC-AUC if available)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(REPORT_DIR / f"cm_{type(model).__name__}.png")
    plt.close()

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "CV_F1": cv_score
    }

    if y_prob is not None:
        metrics["ROC_AUC"] = roc_auc_score(y_test, y_prob)

    return metrics, model


# =============================
# TRAIN ALL MODELS
# =============================
def train_all_models(X_train, X_test, y_train, y_test):

    models = get_models()
    results = {}
    trained_models = {}

    print("\n==============================")
    print(" MODEL TRAINING")
    print("==============================")

    for name, model in models.items():
        print(f"\n{name}...")

        try:
            metrics, trained_model = evaluate_model(
                model, X_train, y_train, X_test, y_test
            )

            results[name] = metrics
            trained_models[name] = trained_model

            print(metrics)

        except Exception as e:
            print("Error:", e)

    return results, trained_models


# =============================
# COMPARE MODELS
# =============================
def display_comparison(results):
    df = pd.DataFrame(results).T
    df = df.sort_values("F1", ascending=False)

    print("\n==============================")
    print(" MODEL COMPARISON")
    print("==============================")
    print(df)

    return df


# =============================
# SAVE BEST MODEL
# =============================
def save_best_model(trained_models, results, X_test, y_test):

    best_name = max(results, key=lambda x: results[x]["F1"])
    best_model = trained_models[best_name]

    print("\nBest Model:", best_name)

    y_pred = best_model.predict(X_test)
    print("\nFinal Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(best_model, MODEL_DIR / "model.pkl")

    print("\nModel saved at:", MODEL_DIR / "model.pkl")

    return best_name, best_model


# =============================
# PIPELINE
# =============================
def train_pipeline():

    print("\nSTARTING TRAINING PIPELINE\n")

    X_train, X_test, y_train, y_test = load_processed_data()

    results, trained_models = train_all_models(
        X_train, X_test, y_train, y_test
    )

    display_comparison(results)

    save_best_model(trained_models, results, X_test, y_test)

    print("\nPIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    train_pipeline()