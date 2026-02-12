import streamlit as st
import pandas as pd
import seaborn as sns  # optional but useful
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Cyber Attack Detection",
    page_icon="🛡️",
    layout="wide"
)

# =====================================================
# CUSTOM HEADER
# =====================================================
st.markdown("""
    <h1 style='text-align: center; color: #1f77b4;'>
    🛡️ Cyber Attack Detection Dashboard
    </h1>
    <p style='text-align: center; font-size:18px;'>
    Machine Learning based Attack Type Classification
    </p>
    <hr>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV Dataset", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.info("Developed using Random Forest Classifier")

# =====================================================
# MAIN CONTENT
# =====================================================
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Dataset Preview")
        st.dataframe(data.head())

    with col2:
        st.subheader("📌 Dataset Info")
        st.write("Rows:", data.shape[0])
        st.write("Columns:", data.shape[1])

    st.markdown("---")

    # =====================================================
    # TARGET SELECTION
    # =====================================================
    st.subheader("🎯 Model Configuration")
    possible_targets = [col for col in data.columns if "label" in col.lower() or "attack" in col.lower() or "class" in col.lower()]
    target_column = st.selectbox("Select Target Column", possible_targets if possible_targets else data.columns)

    if st.button("🚀 Train Model"):
        with st.spinner("Training model... Please wait ⏳"):
            try:
                # ---------------------------
                # DATA PREP
                # ---------------------------
                X = data.drop(columns=[target_column])
                y = data[target_column]

                X = pd.get_dummies(X)
                if y.dtype == "object":
                    encoder = LabelEncoder()
                    y = encoder.fit_transform(y)

                # ---------------------------
                # SPLIT DATA
                # ---------------------------
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                # ---------------------------
                # TRAIN MODEL
                # ---------------------------
                model = RandomForestClassifier(n_estimators=200, random_state=42)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.success(f"✅ Model Training Completed! Accuracy: {round(accuracy*100,2)}%")

                # ---------------------------
                # METRICS DISPLAY
                # ---------------------------
                col1, col2, col3 = st.columns(3)
                col1.metric("Model", "Random Forest")
                col2.metric("Test Size", "20%")
                col3.metric("Accuracy", f"{round(accuracy*100,2)}%")

                st.markdown("---")

                # ---------------------------
                # CLASSIFICATION REPORT + CONFUSION MATRIX
                # ---------------------------
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📄 Classification Report")
                    report_dict = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report_dict).transpose().round(3)
                    st.dataframe(
                        report_df.style
                        .background_gradient(cmap="Blues")
                        .set_properties(**{"text-align": "center", "font-size": "14px"}),
                        use_container_width=True
                    )
                    st.markdown("### 📊 Overall Metrics")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Precision (Avg)", round(report_df.loc["weighted avg", "precision"], 3))
                    m2.metric("Recall (Avg)", round(report_df.loc["weighted avg", "recall"], 3))
                    m3.metric("F1-Score (Avg)", round(report_df.loc["weighted avg", "f1-score"], 3))

                with col2:
                    st.subheader("📊 Confusion Matrix")
                    fig, ax = plt.subplots()
                    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
                    st.pyplot(fig)

                st.markdown("---")

                # ---------------------------
                # FULL DATASET PREDICTION
                # ---------------------------
                full_predictions = model.predict(X)
                result_df = data.copy()
                result_df["Predicted_Attack_Type"] = full_predictions

                st.subheader("🧾 Sample Prediction Results")
                st.dataframe(result_df.head())

                csv_output = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Full Prediction Results",
                    csv_output,
                    "attack_predictions.csv",
                    "text/csv"
                )

            except Exception as e:
                st.error(f"Error: {e}")

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<hr>
<p style='text-align: center; font-size:14px;'>
Cyber Security Attack Detection System
</p>
""", unsafe_allow_html=True)
