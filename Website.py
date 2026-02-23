import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="Cyber Attack Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# ======================================
# SIDEBAR
# ======================================
with st.sidebar:
    st.title("🛡️ Cyber Dashboard")
    st.markdown("""
    - Upload dataset
    - View analytics
    - Predict attacks
    - View model evaluation
    """)
    st.info("AI-based Cyber Attack Detection")

    uploaded_file = st.file_uploader(
        "📂 Upload CSV File",
        type=["csv"]
    )

# ======================================
# HEADER
# ======================================
st.markdown(
    """
    <div style="text-align:center;">
        <h1>Cyber Attack Detection Dashboard</h1>
        <p>Analyze network data and detect security threats</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("---")

# ======================================
# LOAD MODEL + METRICS
# ======================================
try:
    model = joblib.load("attack_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    encoder = joblib.load("label_encoder.pkl")
    model_metrics = joblib.load("model_metrics.pkl")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# ======================================
# SHOW OFFICIAL TRAINING PERFORMANCE
# ======================================
st.subheader("📊 Official Model Test Performance (From Training Phase)")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", model_metrics["accuracy"])
col2.metric("F1 (Macro)", model_metrics["f1_macro"])
col3.metric("Precision (Macro)", model_metrics["precision_macro"])
col4.metric("Recall (Macro)", model_metrics["recall_macro"])

st.write("---")

# ======================================
# PROCESS UPLOADED FILE
# ======================================
if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Dataset Overview")
    colA, colB, colC = st.columns(3)
    colA.metric("Total Rows", len(data))
    colB.metric("Columns", len(data.columns))
    colC.metric("Missing Values", data.isnull().sum().sum())

    st.write("---")

    st.subheader("Preview of Uploaded Data")
    st.dataframe(data.head(10), use_container_width=True)

    st.write("---")

    if st.button("🚀 Run Threat Analysis", use_container_width=True):

        with st.spinner("Running predictions..."):

            X = data.drop(columns=["Attack Type"]) if "Attack Type" in data.columns else data.copy()
            X = pd.get_dummies(X)
            X = X.reindex(columns=model_columns, fill_value=0)

            predictions = model.predict(X)

            # Convert predictions to readable labels
            try:
                predictions = encoder.inverse_transform(predictions)
            except:
                predictions = predictions.astype(str)

            data["Predicted_Attack_Type"] = predictions

        st.success("✅ Prediction Completed")

        # ======================================
        # ROW-WISE RESULTS (TEAM REQUIREMENT)
        # ======================================
        st.subheader("🔍 Row-wise Prediction Results")

        if "Attack Type" in data.columns:
            comparison_df = data[["Attack Type", "Predicted_Attack_Type"]]
            st.dataframe(comparison_df, use_container_width=True)
        else:
            st.dataframe(data[["Predicted_Attack_Type"]], use_container_width=True)

        st.write("---")

        # ======================================
        # PREDICTED DISTRIBUTION
        # ======================================
        st.subheader("📈 Predicted Attack Distribution")

        distribution = Counter(predictions)
        dist_df = pd.DataFrame(distribution.items(), columns=["Attack Type", "Count"])
        dist_df["Percentage (%)"] = (dist_df["Count"] / len(data) * 100).round(2)

        st.dataframe(dist_df, use_container_width=True)

        fig = px.pie(dist_df, names="Attack Type", values="Count")
        st.plotly_chart(fig, use_container_width=True)

        st.write("---")

                
               # ======================================
        # SAFE EVALUATION BLOCK
        # ======================================
        if "Attack Type" in data.columns:
        
            st.subheader("📊 Evaluation on Uploaded Dataset")
        
            y_true = data["Attack Type"]
            y_pred = data["Predicted_Attack_Type"]
        
            # 🔥 FORCE BOTH TO STRING (Fixes mixed label error)
            y_true = y_true.astype(str)
            y_pred = y_pred.astype(str)
        
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
        
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
            recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy:.4f}")
            col2.metric("Precision (Macro)", f"{precision:.4f}")
            col3.metric("Recall (Macro)", f"{recall:.4f}")
            col4.metric("F1 (Macro)", f"{f1:.4f}")
        
            st.subheader("Detailed Classification Report")
            report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(3), use_container_width=True)
        # ======================================
        # DOWNLOAD RESULTS
        # ======================================
        st.download_button(
            "📥 Download Full Results",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name="threat_analysis_results.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    st.info("📂 Upload a CSV file from the sidebar to begin analysis.")
