import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ======================================
# PAGE CONFIG (DASHBOARD STYLE)
# ======================================
st.set_page_config(
    page_title="Cyber Attack Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# ===== SIDEBAR =====
with st.sidebar:
    st.title("🛡️ Cyber Dashboard")
    st.markdown("""
    - Upload dataset
    - View analytics
    - Predict attacks
    """)
    st.info("AI-based Cyber Attack Detection")

# ===== HEADER =====
st.title("Cyber Attack Detection Dashboard")
st.markdown("Analyze network data and detect security threats.")
st.write("---")

# ======================================
# LOAD MODEL
# ======================================
model = joblib.load("attack_model.pkl")
model_columns = joblib.load("model_columns.pkl")
encoder = joblib.load("label_encoder.pkl")

# ======================================
# FILE UPLOAD
# ======================================
uploaded_file = st.file_uploader(
    "📂 Upload CSV File",
    type=["csv"],
    help="Upload network traffic dataset"
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # ===== DASHBOARD STATS =====
    st.subheader("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(data))
    col2.metric("Columns", len(data.columns))
    col3.metric("Missing Values", data.isnull().sum().sum())

    st.write("---")

    # ===== DATA PREVIEW =====
    st.subheader("Data Preview")
    st.dataframe(data.head(), use_container_width=True)

    # ===== ATTACK TYPE DISTRIBUTION =====
    if "Attack Type" in data.columns:
        st.subheader("📈 Attack Type Distribution")

        fig = px.pie(
            data,
            names="Attack Type",
            title="Attack Type Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.write("---")

    # ===== PREDICTION BUTTON =====
    if st.button("🚀 Predict Attack Type", use_container_width=True):

        with st.spinner("Analyzing data..."):

            # Prepare data
            X = data.drop(columns=["Attack Type"]) if "Attack Type" in data.columns else data.copy()
            X = pd.get_dummies(X)
            X = X.reindex(columns=model_columns, fill_value=0)

            # Predict
            predictions = model.predict(X)

            if encoder:
                predictions = encoder.inverse_transform(predictions)
            else:
                label_map = {0: "Malware", 1: "DDoS", 2: "Intrusion"}
                predictions = [label_map.get(int(p), str(p)) for p in predictions]

            data["Predicted_Attack_Type"] = predictions

        st.success("✅ Prediction Completed!")

        # ===== DASHBOARD RESULTS =====
        st.subheader("🛡️ Prediction Dashboard")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Prediction Results")
            st.dataframe(data.head(), use_container_width=True)

        with col2:
            st.markdown("### Attack Distribution (Predicted)")

            fig2 = px.pie(
                data,
                names="Predicted_Attack_Type",
                title="Predicted Attack Distribution"
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.write("---")

        # ===== DOWNLOAD =====
        st.download_button(
            label="📥 Download Results",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name="attack_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )
