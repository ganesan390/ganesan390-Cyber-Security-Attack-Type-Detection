import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load trained model & preprocessors
# -------------------------------
model = joblib.load("model.pkl")

# Optional (only if used during training)
# encoder = joblib.load("encoder.pkl")
# scaler = joblib.load("scaler.pkl")

# -------------------------------
# App UI
# -------------------------------
st.set_page_config(page_title="Cyber Attack Detection", layout="wide")

st.title("🔐 Cyber Security Attack Type Detection")
st.write("Upload a CSV file to predict cyber attack types")

# -------------------------------
# CSV Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

# -------------------------------
# Preprocessing function
# -------------------------------
def preprocess_data(df):
    df = df.copy()

    # Example preprocessing (adjust to your project)
    # Drop target column if present
    if "Attack Type" in df.columns:
        df.drop(columns=["Attack Type"], inplace=True)

    # Handle missing values
    df.fillna(0, inplace=True)

    # Example: Encode categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col] = df[col].astype("category").cat.codes

    # Example: Scaling (if used)
    # df[df.columns] = scaler.transform(df)

    return df

# -------------------------------
# Prediction logic
# -------------------------------
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(data.head())

        if st.button("🚀 Predict Attack Type"):
            processed_data = preprocess_data(data)

            predictions = model.predict(processed_data)

            data["Predicted Attack Type"] = predictions

            st.subheader("✅ Prediction Results")
            st.dataframe(data)

            # Download result
            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Predictions as CSV",
                data=csv,
                file_name="attack_predictions.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")
