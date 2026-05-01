"""
Advanced Model Evaluation Module
ROC Curve + PR Curve + Confusion Matrix + Metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import sys

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =========================
# PATH CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data" / "processed"
REPORT_DIR = MODEL_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# LOAD MODEL + DATA
# =========================
def load_model_and_data():
    model = joblib.load(MODEL_DIR / "model.pkl")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    X_test = test_df.drop("Outcome", axis=1)
    y_test = test_df["Outcome"]

    print(f"Model loaded ")
    print(f"Test samples: {X_test.shape[0]}")

    return model, X_test, y_test


# =========================
# CONFUSION MATRIX
# =========================
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    path = REPORT_DIR / "confusion_matrix.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")


# =========================
# ROC CURVE
# =========================
def plot_roc(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")

    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()

    path = REPORT_DIR / "roc_curve.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")

    return roc_auc


# =========================
# PRECISION RECALL
# =========================
def plot_pr(y_true, y_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AUC = {pr_auc:.3f}")

    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()

    path = REPORT_DIR / "pr_curve.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")

    return pr_auc


# =========================
# MAIN EVALUATION
# =========================
def evaluate():
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    model, X_test, y_test = load_model_and_data()

    y_pred = model.predict(X_test)

    # probabilités (safe)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = None
    threshold = 0.5  # important pour diabète (plus sensible)
    y_pred = (y_proba >= threshold).astype(int)
    # ================= metrics =================
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nMetrics Summary:")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1-score :", f1_score(y_test, y_pred))

    # ================= plots =================
    plot_confusion_matrix(y_test, y_pred)

    if y_proba is not None:
        roc_auc = plot_roc(y_test, y_proba)
        pr_auc = plot_pr(y_test, y_proba)

        print("\nROC AUC:", roc_auc)
        print("PR AUC :", pr_auc)

    print("\nEvaluation completed")


if __name__ == "__main__":
    evaluate()