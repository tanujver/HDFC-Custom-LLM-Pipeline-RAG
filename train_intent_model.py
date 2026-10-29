from pathlib import Path
import json
import logging

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "cleaned"
OUTPUT_DIR = PROJECT_DIR / "artifacts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "banking77_train_cleaned.csv"
TEST_FILE = DATA_DIR / "banking77_test_cleaned.csv"

MODEL_FILE = OUTPUT_DIR / "intent_classifier.joblib"
REPORT_FILE = OUTPUT_DIR / "intent_classification_report.csv"
PREDICTIONS_FILE = OUTPUT_DIR / "test_predictions.csv"
CONFUSION_CSV = OUTPUT_DIR / "confusion_matrix.csv"
CONFUSION_PNG = OUTPUT_DIR / "confusion_matrix.png"
METRICS_FILE = OUTPUT_DIR / "intent_metrics.json"
CLASSES_FILE = OUTPUT_DIR / "intent_classes.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# =====================================================
# 2. LOAD AND VALIDATE DATA
# =====================================================

def load_data(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    required_columns = {"text", "category"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{file_path.name} must contain "
            "'text' and 'category' columns."
        )

    df = df[["text", "category"]].copy()

    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["category"] = (
        df["category"].fillna("").astype(str).str.strip()
    )

    # Remove empty questions and missing/empty labels
    df = df[
        (df["text"] != "") &
        (df["category"] != "")
    ].reset_index(drop=True)

    return df


logger.info("Loading Banking77 datasets...")

train_df = load_data(TRAIN_FILE)
test_df = load_data(TEST_FILE)

X_train = train_df["text"]
y_train = train_df["category"]

X_test = test_df["text"]
y_test = test_df["category"]

train_classes = set(y_train.unique())
test_classes = set(y_test.unique())

if train_classes != test_classes:
    missing_in_train = sorted(test_classes - train_classes)
    missing_in_test = sorted(train_classes - test_classes)

    raise ValueError(
        "Train and test class labels do not match.\n"
        f"Classes only in test: {missing_in_train}\n"
        f"Classes only in train: {missing_in_test}"
    )

logger.info("Training rows: %s", len(train_df))
logger.info("Testing rows: %s", len(test_df))
logger.info("Intent classes: %s", len(train_classes))


# =====================================================
# 3. BUILD INTENT CLASSIFICATION PIPELINE
# =====================================================

# TF-IDF learns vocabulary/IDF from training data only.
# Logistic Regression predicts one Banking77 intent.

intent_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_features=100000,
            sublinear_tf=True,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            C=4.0,
            random_state=42,
        ),
    ),
])


# =====================================================
# 4. TRAIN MODEL
# =====================================================

logger.info("Training TF-IDF + Logistic Regression...")

intent_model.fit(X_train, y_train)

logger.info("Training completed.")


# =====================================================
# 5. EVALUATE ON TEST DATA
# =====================================================

logger.info("Predicting test intents...")

y_pred = intent_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0,
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0,
)

logger.info("Test Accuracy: %.4f", accuracy)
logger.info("Macro F1: %.4f", macro_f1)
logger.info("Weighted F1: %.4f", weighted_f1)


# =====================================================
# 6. SAVE CLASSIFICATION REPORT
# =====================================================

report_dict = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(REPORT_FILE, index=True)

print("\n===== CLASSIFICATION REPORT =====")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )
)


# =====================================================
# 7. SAVE TEST PREDICTIONS
# =====================================================

predictions_df = test_df.copy()

predictions_df["predicted_intent"] = y_pred
predictions_df["correct_prediction"] = (
    predictions_df["category"] ==
    predictions_df["predicted_intent"]
)

# Probability is model confidence, not guaranteed correctness.
probabilities = intent_model.predict_proba(X_test)

predictions_df["predicted_intent_confidence"] = (
    probabilities.max(axis=1)
)

predictions_df.to_csv(
    PREDICTIONS_FILE,
    index=False,
)

logger.info("Saved test predictions: %s", PREDICTIONS_FILE)


# =====================================================
# 8. SAVE CONFUSION MATRIX
# =====================================================

logger.info("Generating confusion matrix...")

labels = sorted(train_classes)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels,
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels,
)

cm_df.index.name = "actual_intent"
cm_df.columns.name = "predicted_intent"

cm_df.to_csv(CONFUSION_CSV)

fig, ax = plt.subplots(figsize=(22, 22))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels,
)

display.plot(
    ax=ax,
    xticks_rotation=90,
    colorbar=False,
    values_format="",
)

ax.set_title("Banking77 Intent Classification")
ax.set_xlabel("Predicted Intent")
ax.set_ylabel("Actual Intent")

plt.tight_layout()
plt.savefig(
    CONFUSION_PNG,
    dpi=150,
    bbox_inches="tight",
)

plt.close(fig)

logger.info("Confusion matrix saved.")


# =====================================================
# 9. SAVE MODEL AND CLASS LABELS
# =====================================================

joblib.dump(intent_model, MODEL_FILE)

with open(CLASSES_FILE, "w", encoding="utf-8") as f:
    json.dump(
        labels,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =====================================================
# 10. SAVE SUMMARY METRICS
# =====================================================

metrics = {
    "model": "TF-IDF + Logistic Regression",
    "train_rows": int(len(train_df)),
    "test_rows": int(len(test_df)),
    "number_of_classes": int(len(labels)),
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
}

with open(METRICS_FILE, "w", encoding="utf-8") as f:
    json.dump(
        metrics,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =====================================================
# 11. TEST WITH CUSTOM QUERIES
# =====================================================

def predict_intent(text):
    predicted = intent_model.predict([text])[0]

    probability = intent_model.predict_proba([text])[0]
    classes = intent_model.named_steps[
        "classifier"
    ].classes_

    top_indices = np.argsort(probability)[::-1][:3]

    top_predictions = [
        {
            "intent": str(classes[i]),
            "confidence": round(float(probability[i]), 4),
        }
        for i in top_indices
    ]

    return {
        "query": text,
        "predicted_intent": str(predicted),
        "top_3_predictions": top_predictions,
    }


# =====================================================
# 12. FINAL SUMMARY
# =====================================================

print("\n===== FINAL MODEL METRICS =====")
print(json.dumps(metrics, indent=2))

print("\n===== SAVED FILES =====")
for path in [
    MODEL_FILE,
    REPORT_FILE,
    PREDICTIONS_FILE,
    CONFUSION_CSV,
    CONFUSION_PNG,
    METRICS_FILE,
    CLASSES_FILE,
]:
    print(path)

print("\n===== CUSTOM QUERY TEST =====")

test_queries = [
    "My bank transfer is still pending",
    "I forgot my app passcode",
    "My card payment was declined",
]

for query in test_queries:
    print(json.dumps(
        predict_intent(query),
        indent=2,
        ensure_ascii=False,
    ))

print("\nIntent classifier training and evaluation completed!")