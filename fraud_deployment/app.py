from pathlib import Path
import datetime as dt
import hashlib
import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Credit Card Fraud Detection — Streamlit Deployment
# ============================================================
# This app is designed to work with the artifacts produced by
# the ML notebook:
#   models/best_ml_model_latest.joblib
#   models/amount_scaler.joblib
#   models/model_metadata.joblib
#
# The trained model expects:
#   V1 ... V28 + Log_Amount
#
# Time and Amount are accepted as user-facing transaction
# fields, but Time is NOT a model feature. Log_Amount is
# calculated from Amount and transformed with the saved scaler.
# ============================================================

APP_VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
LOG_PATH = REPORTS_DIR / "prediction_log.csv"

MODEL_PATH = MODEL_DIR / "best_ml_model_latest.joblib"
SCALER_PATH = MODEL_DIR / "amount_scaler.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.joblib"

EXPECTED_V_FEATURES = [f"V{i}" for i in range(1, 29)]


def check_artifacts():
    missing = [
        str(p.relative_to(ROOT))
        for p in [MODEL_PATH, SCALER_PATH, METADATA_PATH]
        if not p.exists()
    ]
    return missing


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, scaler, metadata


def prepare_features(df: pd.DataFrame, scaler, feature_cols):
    """Convert raw transaction columns into the exact model input."""
    work = df.copy()

    if "Amount" not in work.columns:
        raise ValueError("The input must contain an 'Amount' column.")

    missing_v = [c for c in EXPECTED_V_FEATURES if c not in work.columns]
    if missing_v:
        raise ValueError(
            "Missing required model features: " + ", ".join(missing_v)
        )

    for col in EXPECTED_V_FEATURES + ["Amount"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    if work[EXPECTED_V_FEATURES + ["Amount"]].isna().any().any():
        bad = work[EXPECTED_V_FEATURES + ["Amount"]].isna().sum()
        bad = bad[bad > 0]
        raise ValueError(
            "Some required numeric fields contain invalid/missing values: "
            + ", ".join(f"{k} ({v})" for k, v in bad.items())
        )

    # Same feature engineering as the ML notebook.
    work["Log_Amount"] = np.log1p(work["Amount"])

    # Same preprocessing as training: only Log_Amount is robust-scaled.
    work[["Log_Amount"]] = scaler.transform(work[["Log_Amount"]])

    # Force exact training order from metadata.
    missing_model_cols = [c for c in feature_cols if c not in work.columns]
    if missing_model_cols:
        raise ValueError(
            "The saved model expects columns that could not be created: "
            + ", ".join(missing_model_cols)
        )

    return work[feature_cols].copy()


def make_input_hash(row: dict) -> str:
    payload = json.dumps(row, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def log_prediction(
    raw_features: dict,
    probability: float,
    predicted_class: int,
    model_version: str,
):
    """Log prediction metadata without storing raw transaction values."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_version": model_version,
        "predicted_probability": round(float(probability), 6),
        "predicted_class": int(predicted_class),
        "input_hash": make_input_hash(raw_features),
    }

    log_df = pd.DataFrame([record])
    header_needed = not LOG_PATH.exists()
    log_df.to_csv(
        LOG_PATH,
        mode="a",
        header=header_needed,
        index=False,
    )
    return record


def predict_dataframe(raw_df: pd.DataFrame, model, scaler, metadata):
    feature_cols = metadata["feature_cols"]
    threshold = float(metadata["threshold"])
    model_version = str(metadata["model_version"])

    model_input = prepare_features(raw_df, scaler, feature_cols)
    probabilities = model.predict_proba(model_input)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    result = raw_df.copy()
    result["fraud_probability"] = probabilities.round(6)
    result["predicted_class"] = predictions
    result["prediction"] = np.where(
        predictions == 1, "Fraud Alert", "Normal"
    )

    for position, (_, row) in enumerate(raw_df.iterrows()):
        log_prediction(
            row.to_dict(),
            float(probabilities[position]),
            int(predictions[position]),
            model_version,
        )

    return result, probabilities, predictions

def main():
    # ------------------------- Page setup -------------------------
    st.set_page_config(
        page_title="Credit Card Fraud Detection",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("🛡️ Credit Card Fraud Detection")
    st.caption(
        "Leakage-safe ML deployment • Tuned Random Forest • "
        "Validation-selected decision threshold"
    )

    missing_artifacts = check_artifacts()
    if missing_artifacts:
        st.error("Model artifacts are missing.")
        st.markdown(
            "Run the ML notebook first so it creates the following files:"
        )
        for item in missing_artifacts:
            st.code(item)
        st.info(
            "Do not upload the raw Kaggle dataset to GitHub just to make the app run. "
            "The deployment needs the trained model artifacts, not the full dataset."
        )
        st.stop()

    try:
        model, scaler, metadata = load_artifacts()
    except Exception as exc:
        st.error("The saved model artifacts could not be loaded.")
        st.exception(exc)
        st.stop()

    feature_cols = metadata["feature_cols"]
    threshold = float(metadata["threshold"])
    model_name = str(metadata["model_name"])
    model_version = str(metadata["model_version"])

    # --------------------------- Sidebar -------------------------
    with st.sidebar:
        st.header("Model Information")
        st.write(f"**Model:** {model_name}")
        st.write(f"**Version:** {model_version}")
        st.write(f"**Threshold:** {threshold:.2f}")
        st.write(f"**App version:** {APP_VERSION}")
        st.divider()
        st.write("Expected model features:")
        st.code("V1 ... V28 + Log_Amount")

    # ---------------------------- Tabs ----------------------------
    tab1, tab2, tab3 = st.tabs(
        ["🔎 Single Transaction", "📁 Batch CSV", "📊 Monitoring"]
    )

    with tab1:
        st.subheader("Check one transaction")
        st.write(
            "Enter the 28 anonymized PCA features and the transaction amount. "
            "The app calculates Log_Amount and applies the saved preprocessing."
        )

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=22.0,
            step=1.0,
            format="%.2f",
            help="Raw transaction amount used to calculate Log_Amount.",
        )

        time_value = st.number_input(
            "Time (optional display field)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            help="Kept for transaction context. Time is not passed to the trained model.",
        )

        st.markdown("### PCA features")
        v_values = {}

        cols = st.columns(4)
        for i, feature in enumerate(EXPECTED_V_FEATURES):
            with cols[i % 4]:
                v_values[feature] = st.number_input(
                    feature,
                    value=0.0,
                    format="%.6f",
                    key=f"single_{feature}",
                )

        if st.button("Run Fraud Prediction", type="primary", use_container_width=True):
            raw = {"Time": time_value, "Amount": amount, **v_values}
            raw_df = pd.DataFrame([raw])

            try:
                model_input = prepare_features(raw_df, scaler, feature_cols)
                probability = float(model.predict_proba(model_input)[0, 1])
                predicted_class = int(probability >= threshold)

                log_prediction(
                    raw,
                    probability,
                    predicted_class,
                    model_version,
                )

                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                c1.metric("Fraud Probability", f"{probability:.2%}")
                c2.metric("Decision Threshold", f"{threshold:.2f}")
                c3.metric(
                    "Result",
                    "FRAUD ALERT" if predicted_class else "NORMAL",
                )

                if predicted_class:
                    st.error(
                        "⚠️ Fraud Alert — the predicted probability is above "
                        "the validation-selected threshold."
                    )
                else:
                    st.success(
                        "✅ Normal — the predicted probability is below "
                        "the validation-selected threshold."
                    )

                st.caption(
                    "This is a fraud-screening prototype and not a replacement "
                    "for human review or a bank authorization system."
                )

            except Exception as exc:
                st.error("Prediction failed.")
                st.exception(exc)

    with tab2:
        st.subheader("Batch scoring from CSV")
        st.write(
            "Upload a CSV containing `V1` through `V28` and `Amount`. "
            "`Time` may also be included for context and is preserved in the output."
        )

        template = pd.DataFrame(
            [{**{f"V{i}": 0.0 for i in range(1, 29)}, "Amount": 22.0, "Time": 0.0}]
        )
        st.download_button(
            "Download CSV Template",
            data=template.to_csv(index=False).encode("utf-8"),
            file_name="fraud_prediction_template.csv",
            mime="text/csv",
        )

        uploaded = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Required columns: V1...V28 and Amount.",
        )

        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded)
                st.write(f"Rows uploaded: **{len(batch_df):,}**")
                st.dataframe(batch_df.head(10), use_container_width=True)

                if st.button(
                    "Score CSV",
                    type="primary",
                    use_container_width=True,
                ):
                    result, probabilities, predictions = predict_dataframe(
                        batch_df, model, scaler, metadata
                    )

                    fraud_count = int(predictions.sum())
                    normal_count = int(len(predictions) - fraud_count)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Transactions", f"{len(result):,}")
                    c2.metric("Fraud Alerts", f"{fraud_count:,}")
                    c3.metric("Normal", f"{normal_count:,}")

                    st.dataframe(result.head(100), use_container_width=True)

                    st.download_button(
                        "Download Predictions CSV",
                        data=result.to_csv(index=False).encode("utf-8"),
                        file_name="fraud_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

            except Exception as exc:
                st.error("Batch scoring failed.")
                st.exception(exc)

    with tab3:
        st.subheader("Basic Prediction Monitoring")

        if LOG_PATH.exists():
            try:
                logs = pd.read_csv(LOG_PATH)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Logged Predictions", f"{len(logs):,}")
                c2.metric(
                    "Fraud Alerts",
                    f"{int(logs['predicted_class'].sum()):,}",
                )
                c3.metric(
                    "Alert Rate",
                    f"{logs['predicted_class'].mean():.2%}",
                )
                c4.metric(
                    "Avg. Probability",
                    f"{logs['predicted_probability'].mean():.2%}",
                )

                st.markdown("### Recent prediction log")
                st.dataframe(logs.tail(100).iloc[::-1], use_container_width=True)

                st.caption(
                    "The log stores timestamp, model version, prediction, "
                    "probability, and a short input hash — not the raw transaction values."
                )
            except Exception as exc:
                st.warning(f"Could not read the prediction log: {exc}")
        else:
            st.info("No predictions have been logged yet.")

    st.markdown("---")
    st.caption(
        "Credit Card Fraud Detection project • Deployment layer • "
        "Model artifacts must be generated by the ML notebook before running the app."
    )


if __name__ == "__main__":
    main()
