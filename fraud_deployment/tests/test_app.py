from pathlib import Path
import sys

import joblib
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import (
    EXPECTED_V_FEATURES,
    MODEL_PATH,
    SCALER_PATH,
    METADATA_PATH,
    prepare_features,
)


def test_required_artifacts_exist():
    assert MODEL_PATH.exists(), f"Missing {MODEL_PATH}"
    assert SCALER_PATH.exists(), f"Missing {SCALER_PATH}"
    assert METADATA_PATH.exists(), f"Missing {METADATA_PATH}"


def test_model_has_predict_proba():
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict_proba")


def test_metadata_has_expected_fields():
    metadata = joblib.load(METADATA_PATH)
    for key in [
        "feature_cols",
        "threshold",
        "model_name",
        "model_version",
        "cost_false_negative",
        "cost_false_positive",
    ]:
        assert key in metadata


def test_feature_preparation_matches_saved_order():
    scaler = joblib.load(SCALER_PATH)
    metadata = joblib.load(METADATA_PATH)

    raw = {feature: 0.0 for feature in EXPECTED_V_FEATURES}
    raw["Amount"] = 22.0
    raw["Time"] = 0.0

    raw_df = pd.DataFrame([raw])
    X = prepare_features(raw_df, scaler, metadata["feature_cols"])

    assert X.columns.tolist() == metadata["feature_cols"]
    assert X.shape[0] == 1
    assert np.isfinite(X.to_numpy()).all()


def test_model_prediction_is_valid_probability():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    metadata = joblib.load(METADATA_PATH)

    raw = {feature: 0.0 for feature in EXPECTED_V_FEATURES}
    raw["Amount"] = 22.0
    raw["Time"] = 0.0

    X = prepare_features(
        pd.DataFrame([raw]),
        scaler,
        metadata["feature_cols"],
    )
    probability = float(model.predict_proba(X)[0, 1])

    assert 0.0 <= probability <= 1.0
