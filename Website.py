import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# --- PAGE CONFIG ---
st.set_page_config(page_title="Cyber Attack Detector", layout="wide")

# --- LOAD MODEL ---
current_dir = Path(__file__).parent
model_path = current_dir / "model.pkl"

@st.cache_resource
def load_model(path):
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model(model_path)

# --- UI HEADER ---
st.title("🛡️ Cyber Security Attack Type Detection")
st.markdown("Upload a CSV file containing network traffic data to detect cyber attacks.")

# --- MODEL CHECK ---
if model is None:
    st.error("❌ model.pkl not found or failed to load.")
    st.info(f"Ensure model.pkl is in this folder:\n{model_path}")
    st.stop()
else:
    st.success("✅ Model loaded successfully!")

# =========================================================
# 📁 CSV UPLOAD SECTION (ONLY INPUT METHOD)
# =========================================================

st.divider()
st.subheader("📁 Upload Network Traffic CSV File")

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        # Read CSV
        data = pd.read_csv(uploaded_file)

        st.write("### 📊 Uploaded Data Preview")
        st.dataframe(data.head())

        # --- FEATURE ALIGNMENT ---
        for col in model.feature_names_in_:
            if col not in data.columns:
                data[col] = 0

        # Reorder columns to match training
        data = data[model.feature_names_in_]

        # --- PREDICTION ---
        predictions = model.predict(data)

        # Add predictions column
        data["Prediction"] = predictions

        st.success("✅ Prediction Completed Successfully!")

        st.write("### 🧾 Prediction Results")
        st.dataframe(data.head())

        # --- DOWNLOAD BUTTON ---
        csv_output = data.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Full Prediction Results",
            data=csv_output,
            file_name="cyber_attack_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
