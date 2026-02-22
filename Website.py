import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

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
    Machine Learning based Attack Type Classification (Logistic Regression)
    </p>
    <hr>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV Dataset", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.info("Developed using Logistic Regression + GridSearchCV")

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
    # ==========================
    # LOAD DATA
    # ==========================
    data = pd.read_csv(uploaded_file)
    
    st.subheader("📊 Uploaded Dataset Preview")
    st.dataframe(data.head())
    
    # ==========================
    # PREPARE FEATURES
    # ==========================
    
    # Remove target column if it exists
    if "Attack Type" in data.columns:
        X = data.drop(columns=["Attack Type"])
    else:
        X = data.copy()
    
    X = pd.get_dummies(X)
    
    # Align columns (important if model trained previously)
    X = X.reindex(columns=X_train.columns, fill_value=0)

    # ==========================
    # PREDICT BUTTON
    # ==========================
    if st.button("🚀 Predict Attack Type"):
    
        predictions = best_model.predict(X)
    
        result_df = data.copy()
        result_df["Predicted_Attack_Type"] = predictions
    
        st.success("✅ Prediction Completed!")
    
        st.subheader("🛡️ Prediction Results")
        st.dataframe(result_df.head())
    
        # Show first prediction clearly
        first_prediction = predictions[0]

    if label_map:
        first_prediction = label_map[first_prediction]

    st.markdown("---")
    st.markdown(f"## 🚨 Predicted Attack Type: **{first_prediction}**")
                # ---------------------------
                # DATA PREPARATION
                # ---------------------------
                X = data.drop(columns=[target_column])
                y = data[target_column]

                # One-hot encode categorical features
                X = pd.get_dummies(X)

                # Encode target if categorical
                if y.dtype == "object":
                    encoder = LabelEncoder()
                    y = encoder.fit_transform(y)

                # ---------------------------
                # TRAIN TEST SPLIT (Stratified)
                # ---------------------------
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=0.2,
                    random_state=42,
                    stratify=y
                )

                # ---------------------------
                # PIPELINE + GRID SEARCH
                # ---------------------------
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("logreg", LogisticRegression(max_iter=2000))
                ])

                param_grid = {
                    "logreg__C": [0.01, 0.1, 1, 10],
                    "logreg__penalty": ["l2"],
                    "logreg__solver": ["lbfgs"]
                }

                grid = GridSearchCV(
                    pipeline,
                    param_grid,
                    cv=5,
                    scoring="f1_macro",
                    n_jobs=-1
                )

                grid.fit(X_train, y_train)

                best_model = grid.best_estimator_

                # ---------------------------
                # EVALUATION
                # ---------------------------
                y_pred = best_model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="macro")
                precision = precision_score(y_test, y_pred, average="macro")
                recall = recall_score(y_test, y_pred, average="macro")

                st.success(f"✅ Model Training Completed! F1-Macro: {round(f1,4)}")

                # ---------------------------
                # METRICS DISPLAY
                # ---------------------------
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Model", "Logistic Regression")
                col2.metric("Accuracy", f"{round(accuracy*100,2)}%")
                col3.metric("F1-Macro", round(f1,4))
                col4.metric("Precision (Macro)", round(precision,4))

                st.markdown("---")

                # ---------------------------
                # CLASSIFICATION REPORT
                # ---------------------------
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📄 Classification Report")
                    report_dict = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report_dict).transpose().round(3)
                    st.dataframe(report_df, use_container_width=True)

                with col2:
                    st.subheader("📊 Confusion Matrix")
                    fig, ax = plt.subplots()
                    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
                    st.pyplot(fig)

                st.markdown("---")

                # ---------------------------
                # FULL DATASET PREDICTION
                # ---------------------------
                full_predictions = best_model.predict(X)
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
Cyber Security Attack Detection System - Logistic Regression Model
</p>
""", unsafe_allow_html=True)
