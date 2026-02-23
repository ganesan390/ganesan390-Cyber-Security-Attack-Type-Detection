import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from collections import Counter

# ======================================
# PAGE CONFIG
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

    # CSV Upload in Sidebar
    uploaded_file = st.file_uploader(
        "📂 Upload CSV File",
        type=["csv"],
        help="Upload network traffic dataset"
    )

# ===== HEADER =====
st.title("Cyber Attack Detection Dashboard")
st.markdown("Analyze network data and detect security threats.")
st.write("---")

# ======================================
# LOAD MODEL & UTILS
# ======================================
# Note: Ensure these files exist in your working directory
try:
    model = joblib.load("attack_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    encoder = joblib.load("label_encoder.pkl")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# ======================================
# PROCESS FILE ONLY IF UPLOADED
# ======================================
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

    # ===== ATTACK TYPE DISTRIBUTION (IF IN DATASET) =====
    if "Attack Type" in data.columns:
        st.subheader("📈 Original Attack Distribution")
        fig = px.pie(
            data,
            names="Attack Type",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig, use_container_width=True)

    st.write("---")

    # ===== PREDICTION BUTTON =====
    if st.button("🚀 Run AI Threat Analysis", use_container_width=True):

        with st.spinner("Analyzing network packets..."):
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

        st.success("✅ Analysis Completed!")

        # ===== MODIFIED FINAL DETECTION RESULT =====
        # This section calculates the majority and displays it prominently
        attack_counts = Counter(predictions)
        final_attack_name = attack_counts.most_common(1)[0][0]

        st.markdown("---")
        
        # Centered Alert Box with Large Majority Name
        st.markdown(
            f"""
            <div style="
                background-color: #fff5f5;
                padding: 40px;
                border-radius: 20px;
                border: 4px solid #ff4b4b;
                text-align: center;
                box-shadow: 0px 4px 15px rgba(255, 75, 75, 0.2);
            ">
                <p style="color: #ff4b4b; font-size: 20px; font-weight: bold; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 2px;">
                    🚨 Final Detection Result
                </p>
                <h1 style="color: #1E1E1E; font-size: 72px; margin: 0; padding: 10px 0;">
                    {final_attack_name}
                </h1>
                <p style="color: #555; font-size: 16px; margin-top: 10px;">
                    Identified as the primary threat based on <b>{attack_counts[final_attack_name]}</b> matching patterns.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # ===== PREDICTION DETAILS =====
        st.subheader("📊 Prediction Analytics")
        col_left, col_right = st.columns(2)

        with col_left:
            st.write("**Recent Predictions Table**")
            st.dataframe(data.head(10), use_container_width=True)

        with col_right:
            st.write("**Predicted Threat Distribution**")
            fig2 = px.pie(
                data,
                names="Predicted_Attack_Type",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.write("---")

        # ===== DOWNLOAD =====
        st.download_button(
            label="📥 Download Detailed Threat Report (CSV)",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name="threat_analysis_results.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    st.info("📂 Please upload a network traffic CSV file from the sidebar to begin analysis.")
