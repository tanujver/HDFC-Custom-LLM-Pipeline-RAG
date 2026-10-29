import json
from pathlib import Path

import joblib
import pytest


# Project root and artifact paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "intent_classifier.joblib"
CLASSES_PATH = ARTIFACTS_DIR / "intent_classes.json"
METRICS_PATH = ARTIFACTS_DIR / "intent_metrics.json"


@pytest.fixture(scope="module")
def classifier():
    assert MODEL_PATH.exists(), f"Model file missing: {MODEL_PATH}"
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def intent_classes():
    assert CLASSES_PATH.exists(), f"Classes file missing: {CLASSES_PATH}"

    with open(CLASSES_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Support either a JSON list or a dictionary containing class names
    if isinstance(data, list):
        classes = data
    elif isinstance(data, dict):
        classes = next(
            (
                data[key]
                for key in ("classes", "intent_classes", "labels")
                if key in data
            ),
            None,
        )
        if classes is None:
            classes = list(data.keys())
    else:
        pytest.fail("Unexpected format in intent_classes.json")

    assert isinstance(classes, list)
    return set(classes)


def test_model_file_exists():
    assert MODEL_PATH.exists()


def test_classes_file_exists():
    assert CLASSES_PATH.exists()


def test_model_loads_successfully(classifier):
    assert classifier is not None


def test_model_has_predict_method(classifier):
    assert callable(getattr(classifier, "predict", None))


def test_model_predicts_single_query(classifier):
    predictions = classifier.predict([
        "I want to check my bank account balance"
    ])

    assert len(predictions) == 1
    assert isinstance(predictions[0], str)


def test_model_predicts_multiple_queries(classifier):
    queries = [
        "I want to check my bank account balance",
        "How do I transfer money to another account?",
        "I lost my debit card",
    ]

    predictions = classifier.predict(queries)

    assert len(predictions) == len(queries)


def test_predictions_belong_to_known_classes(classifier, intent_classes):
    queries = [
        "I want to check my bank account balance",
        "How do I transfer money to another account?",
        "I lost my debit card",
    ]

    predictions = classifier.predict(queries)

    assert set(predictions).issubset(intent_classes), (
        f"Unknown predicted labels: {set(predictions) - intent_classes}"
    )


def test_expected_number_of_classes(intent_classes):
    assert len(intent_classes) == 77, (
        f"Expected 77 classes, found {len(intent_classes)}"
    )


def test_metrics_file_exists():
    assert METRICS_PATH.exists()


def test_metrics_contain_expected_scores():
    assert METRICS_PATH.exists(), "Metrics file is missing"

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    assert isinstance(metrics, dict)
    assert len(metrics) > 0

    # Check accuracy if the file uses this common key
    if "accuracy" in metrics:
        assert 0 <= float(metrics["accuracy"]) <= 1