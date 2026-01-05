import os
import io
import numpy as np
import pandas as pd
import streamlit as st

# TensorFlow import can be slow; keep it after Streamlit init
import tensorflow as tf


# ----------------------------
# Config (paths)
# ----------------------------
MODEL_FILENAME = "20251203_111401_clf.keras"
DEFAULT_THRESHOLD = 0.55

# Anchor paths to this file's directory so Streamlit can be launched from anywhere.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Prefer `models/` if it exists, otherwise fall back to `model/`.
MODELS_DIR_ABS = os.path.join(BASE_DIR, "models")
MODEL_DIR_ABS = os.path.join(BASE_DIR, "model")

CHOSEN_DIR = MODELS_DIR_ABS if os.path.isdir(MODELS_DIR_ABS) else MODEL_DIR_ABS
MODEL_PATH = os.path.join(CHOSEN_DIR, MODEL_FILENAME)

# Typical wine features (UCI red wine dataset)
FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


# ----------------------------
# Helpers
# ----------------------------
@st.cache_resource
def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    return tf.keras.models.load_model(model_path, compile=False)


def ensure_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            "CSV is missing required columns:\n"
            + "\n".join(missing)
            + "\n\nExpected columns:\n"
            + ", ".join(FEATURES)
        )
    return df[FEATURES].copy()


def preprocess(X_df: pd.DataFrame):
    return X_df.values.astype(np.float32)


def predict_proba(model, X: np.ndarray):
    """Fast inference using TensorFlow."""
    X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
    y = model(X_tf, training=False)
    y = y.numpy() if hasattr(y, "numpy") else np.asarray(y)

    if y.ndim == 2 and y.shape[1] == 1:
        return y[:, 0]
    if y.ndim == 1:
        return y
    if y.ndim == 2 and y.shape[1] == 2:
        return y[:, 1]

    raise ValueError(f"Unexpected model output shape: {y.shape}")


def proba_to_label(p: np.ndarray, threshold: float):
    return np.where(p >= threshold, "Good", "Bad")


# ----------------------------
# UI (minimal, intuitive)
# ----------------------------
st.set_page_config(page_title="Wine Quality Classifier", layout="wide")
st.title("Wine Quality Classifier")

# Load artifacts
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Warm-up
try:
    _ = model(tf.zeros((1, len(FEATURES)), dtype=tf.float32), training=False)
except Exception:
    pass

manual_tab, csv_tab = st.tabs(["Manual", "CSV"])

# ---- Manual input ----
with manual_tab:
    st.subheader("Enter wine properties")

    defaults = {
        "fixed acidity": 8.3,
        "volatile acidity": 0.53,
        "citric acid": 0.27,
        "residual sugar": 2.4,
        "chlorides": 0.08,
        "free sulfur dioxide": 15.0,
        "total sulfur dioxide": 46.0,
        "density": 0.997,
        "pH": 3.31,
        "sulphates": 0.66,
        "alcohol": 10.4,
    }

    cols = st.columns(2)
    values = {}
    for i, feat in enumerate(FEATURES):
        col = cols[i % 2]
        values[feat] = col.number_input(
            feat,
            value=float(defaults.get(feat, 0.0)),
            format="%.6f" if feat in ["density", "pH"] else "%.4f",
        )

    if st.button("🔮 Predict", use_container_width=True):
        X_df = pd.DataFrame([values], columns=FEATURES)
        X = preprocess(X_df)
        p = float(predict_proba(model, X)[0])
        label = "✅ Good" if p >= DEFAULT_THRESHOLD else "❌ Bad"

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Result", label)
        with col2:
            st.metric("Confidence", f"{p:.1%}")

# ---- CSV upload ----
with csv_tab:
    st.subheader("Batch predictions")

    uploaded = st.file_uploader("📊 Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            X_df = ensure_feature_frame(df)
            X = preprocess(X_df)
            p = predict_proba(model, X)
            labels = proba_to_label(p, DEFAULT_THRESHOLD)

            out = df.copy()
            out["p_good"] = p.astype(float)
            out["prediction"] = labels

            st.success(f"✅ Predicted {len(out)} rows")
            st.dataframe(out, use_container_width=True, hide_index=True)

            buf = io.StringIO()
            out.to_csv(buf, index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=buf.getvalue(),
                file_name="wine_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as e:
            st.error(str(e))