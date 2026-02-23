import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, classification_report

# ======================================
# 1. PAGE CONFIG & STYLING
# ======================================
st.set_page_config(
    page_title="Cyber Threat Intelligence Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS for better organization
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# ======================================
# 2. FEEDBACK-DRIVEN MAPPING
# ======================================
# Mapping provided: 0: Malware, 1: DDoS, 2: Intrusion
LABEL_MAP = {
    0: "Malware",
    1: "DDoS",
    2: "Intrusion"
}

# ======================================
# 3. ASSET LOADING
# ======================================
@st.cache_resource
def load_model_assets():
    try:
        model = joblib.load("attack_model.pkl")
        cols = joblib.load("model_columns.pkl")
        metrics = joblib.load("model_metrics.pkl")
        return model, cols, metrics
    except Exception as e:
        return None, None, None

model, model_columns, model_metrics = load_model_assets()

# ======================================
# 4. SIDEBAR - FILE UPLOAD
# ======================================
with st.sidebar:
    st.title("🛡️ Control Panel")
    st.write("Upload your network logs to identify threats.")
    uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])
    
    if model_metrics:
        st.divider()
        st.subheader("Model Training Stats")
        st.write(f"**Global Accuracy:** {model_metrics.get('accuracy', 'N/A')}")
        st.write(f"**Macro F1:** {model_metrics.get('f1_macro', 'N/A')}")

# ======================================
# 5. MAIN DASHBOARD LOGIC
# ======================================
st.title("Cyber Attack Detection Dashboard")
st.markdown("---")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # ADDRESSING FEEDBACK: "The attack type doesn't use the good columns"
    # We let the user select which column is the Actual Label for Precision calculation
    st.subheader("⚙️ Analysis Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        all_cols = data.columns.tolist()
        # Try to auto-detect the ground truth column if it exists
        default_idx = all_cols.index("Attack Type") if "Attack Type" in all_cols else 0
        actual_label_col = st.selectbox(
            "Which column contains the ACTUAL labels? (Ground Truth)",
            options=["None - Prediction Only"] + all_cols,
            index=default_idx + 1 if "Attack Type" in all_cols else 0
        )
    
    with col2:
        st.info("The model will use the remaining columns as features for prediction.")

    if st.button("🚀 Run Comprehensive Threat Analysis", use_container_width=True):
        with st.spinner("Analyzing network patterns..."):
            
            # Prepare Features
            # We drop the 'actual label' column from X so the model doesn't "cheat"
            drop_cols = [actual_label_col] if actual_label_col != "None - Prediction Only" else []
            X = data.drop(columns=drop_cols)
            X = pd.get_dummies(X)
            X = X.reindex(columns=model_columns, fill_value=0)

            # Perform Prediction
            raw_predictions = model.predict(X)
            data["Predicted_Attack_Type"] = [LABEL_MAP.get(p, str(p)) for p in raw_predictions]

            # Convert Ground Truth to readable labels for comparison
            if actual_label_col != "None - Prediction Only":
                # Ensure ground truth is mapped to the same names (Malware, DDoS, etc.)
                data["Mapped_Actual"] = data[actual_label_col].map(LABEL_MAP).fillna("Unknown")

        # --- SECTION: LOGICAL ORGANIZATION ---
        
        # A. HIGH-LEVEL METRICS
        st.subheader("📊 Execution Summary")
        m1, m2, m3, m4 = st.columns(4)
        
        m1.metric("Total Traffic", len(data))
        
        threats = data[data["Predicted_Attack_Type"] != "Normal"].shape[0] # Adjust if 'Normal' is a label
        m2.metric("Threats Flagged", threats)

        if actual_label_col != "None - Prediction Only":
            y_true = data["Mapped_Actual"].astype(str)
            y_pred = data["Predicted_Attack_Type"].astype(str)
            
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
            
            m3.metric("Dataset Accuracy", f"{acc:.2%}")
            m4.metric("Dataset Precision", f"{prec:.2%}")

        st.divider()

        # B. VISUAL DISTRIBUTION
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.subheader("📈 Predicted Distribution")
            dist_df = data["Predicted_Attack_Type"].value_counts().reset_index()
            dist_df.columns = ["Attack Type", "Count"]
            fig = px.pie(dist_df, names="Attack Type", values="Count", hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Safe)
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            if actual_label_col != "None - Prediction Only":
                st.subheader("🎯 Precision/Recall Report")
                report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose().iloc[:-3, :3] # Only show main classes
                st.dataframe(report_df.style.background_gradient(cmap='Blues'), use_container_width=True)
            else:
                st.warning("Upload ground truth labels to see Precision and Recall analysis.")

        # C. DATA PREVIEW & DOWNLOAD
        st.subheader("🔍 Detailed Prediction Logs")
        display_cols = ["Predicted_Attack_Type"]
        if actual_label_col != "None - Prediction Only":
            display_cols = ["Mapped_Actual"] + display_cols
        
        # Show the first few rows of the result
        st.dataframe(data[display_cols + [c for c in data.columns if c not in display_cols]].head(100), use_container_width=True)

        # Download Result
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Analysis Results (CSV)", data=csv, file_name="threat_report.csv", mime="text/csv")

else:
    # Initial landing state
    st.info("Welcome! Please upload a network log CSV file in the sidebar to begin.")
    st.image("https://img.icons8.com/clouds/200/shield.png") # Just a placeholder icon
