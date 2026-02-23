import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="Cyber Attack Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# ======================================
# LABEL MAPPING (Aligned with Image)
# ======================================
LABEL_MAP = {
    0: "Malware",
    1: "DDoS",
    2: "Intrusion"
}

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
# LOAD MODEL ARTIFACTS
# ======================================
@st.cache_resource
def load_model_files():
    try:
        model = joblib.load("attack_model.pkl")
        model_columns = joblib.load("model_columns.pkl")
        encoder = joblib.load("label_encoder.pkl")
        return model, model_columns, encoder
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

model, model_columns, encoder = load_model_files()

if model is None:
    st.stop()

# ======================================
# HELPER FUNCTION FOR READABLE LABELS
# ======================================
def make_readable(val):
    """Converts numeric predictions/labels to text using strict mapping."""
    try:
        val_int = int(float(val))
        if val_int in LABEL_MAP:
            return LABEL_MAP[val_int]
        # Fallback to encoder or return original if mapping fails
        return encoder.inverse_transform([val_int])[0]
    except:
        # Special case for "Normal Connection" if it's in your dataset
        if str(val).lower() == 'normal' or val == 3: # Assuming 3 or 'normal' is baseline
            return "Normal Connection"
        return str(val)

# ======================================
# PROCESS UPLOADED FILE
# ======================================
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Store labels for evaluation, then drop from visual preview
    actual_labels = None
    if "Attack Type" in data.columns:
        actual_labels = data["Attack Type"].copy()

    st.subheader("📊 Uploaded Dataset Overview")
    colA, colB, colC = st.columns(3)
    colA.metric("Total Rows", len(data))
    
    # PREVIEW LOGIC: Remove 'Attack Type' from display
    preview_df = data.drop(columns=["Attack Type"]) if "Attack Type" in data.columns else data.copy()
    
    colB.metric("Feature Columns", len(preview_df.columns))
    colC.metric("Missing Values", data.isnull().sum().sum())

    st.write("---")
    st.subheader("Preview of Uploaded Data (Features Only)")
    st.dataframe(preview_df.head(10), use_container_width=True)
    st.write("---")

    if st.button("🚀 Run Threat Analysis", use_container_width=True):
        with st.spinner("Analyzing network patterns..."):
            # Prepare Features
            X = preview_df.copy()
            X = pd.get_dummies(X)
            X = X.reindex(columns=model_columns, fill_value=0)

            # Predict
            raw_predictions = model.predict(X)
            
            # Map predictions to names
            data["Predicted_Attack_Type"] = [make_readable(p) for p in raw_predictions]

        st.success("✅ Analysis Completed")

        # ======================================
        # VISUALIZATION
        # ======================================
        st.subheader("📈 Predicted Threat Distribution")
        counts = data["Predicted_Attack_Type"].value_counts().reset_index()
        counts.columns = ["Attack Type", "Count"]
        
        fig = px.pie(
            counts, 
            names="Attack Type", 
            values="Count",
            hole=0.4,
            color="Attack Type",
            color_discrete_map={
                "SQL Injection / Malware": "#EF553B", 
                "DDoS Attack": "#636EFA", 
                "Intrusion": "#00CC96",
                "Normal Connection": "#AB63FA"
            },
            title="Distribution of Detected Threats"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ======================================
        # EVALUATION (Comparing Ground Truth if exists)
        # ======================================
        if actual_labels is not None:
            st.write("---")
            st.subheader("📊 Model Performance Evaluation")
            
            y_true = [make_readable(x) for x in actual_labels]
            y_pred = data["Predicted_Attack_Type"].tolist()

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

            m1, m2 = st.columns(2)
            m1.metric("Detection Accuracy", f"{acc:.2%}")
            m2.metric("F1-Score", f"{f1:.4f}")

        # ======================================
        # RESULTS TABLE
        # ======================================
        st.write("---")
        st.subheader("🔍 Detailed Results (Predictions)")
        
        # Displaying predictions in the exact style of your image
        st.dataframe(data[["Predicted_Attack_Type"]].head(20), use_container_width=True)

        st.download_button(
            "📥 Download Threat Analysis Results",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name="threat_analysis_report.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    st.info("📂 Please upload a CSV file from the sidebar to begin threat detection.")
