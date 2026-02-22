import streamlit as st
import pandas as pd
import joblib

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="Cyber Attack Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Cyber Attack Detection System")
st.markdown("Upload network traffic data and detect attack type instantly.")

# ======================================
# LOAD SAVED MODEL
# ======================================
model = joblib.load("attack_model.pkl")
model_columns = joblib.load("model_columns.pkl")
encoder = joblib.load("label_encoder.pkl")

# ======================================
# FILE UPLOAD
# ======================================
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(data.head())

    if st.button("🚀 Predict Attack Type"):

        # Remove target column if exists
        if "Attack Type" in data.columns:
            X = data.drop(columns=["Attack Type"])
        else:
            X = data.copy()

        # One-hot encode
        X = pd.get_dummies(X)

        # Align with training columns
        X = X.reindex(columns=model_columns, fill_value=0)

        # Predict
        predictions = model.predict(X)

        # Decode labels if encoder exists
        if encoder:
            predictions = encoder.inverse_transform(predictions)

        # Add prediction column
        data["Predicted_Attack_Type"] = predictions

        st.success("✅ Prediction Completed!")

        st.subheader("🛡️ Prediction Results")
        st.dataframe(data.head())

        # Show first prediction clearly
        st.markdown("---")
        st.markdown(
            f"## 🚨 Predicted Attack Type: **{predictions[0]}**"
        )

        # Download button
        csv_output = data.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Prediction Results",
            csv_output,
            "attack_predictions.csv",
            "text/csv"
        )
