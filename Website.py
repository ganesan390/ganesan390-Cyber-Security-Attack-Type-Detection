import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Cyber Attack Detection", layout="wide")

st.title("🛡️ Cyber Attack Detection Using Machine Learning")
st.write("Upload a dataset to train a model and predict cyber attack types.")

# =====================================================
# FILE UPLOAD
# =====================================================
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head())

    st.write("Dataset Shape:", data.shape)

    # =====================================================
    # TARGET SELECTION
    # =====================================================
    target_column = st.selectbox("Select Target Column (Attack Type)", data.columns)

    if st.button("🚀 Train Model"):
        try:
            X = data.drop(columns=[target_column])
            y = data[target_column]

            # -------------------------------
            # Handle categorical features
            # -------------------------------
            X = pd.get_dummies(X)  # Convert strings to numeric

            # Encode target if categorical
            if y.dtype == "object":
                encoder = LabelEncoder()
                y = encoder.fit_transform(y)

            # -------------------------------
            # Train-test split
            # -------------------------------
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42
            )

            # -------------------------------
            # Train model
            # -------------------------------
            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model.fit(X_train, y_train)

            # -------------------------------
            # Predictions on test set
            # -------------------------------
            y_pred = model.predict(X_test)

            # -------------------------------
            # Evaluation
            # -------------------------------
            accuracy = accuracy_score(y_test, y_pred)
            st.success("✅ Model Training Completed")
            st.write(f"### 🎯 Accuracy: {round(accuracy * 100, 2)}%")

            st.subheader("📄 Classification Report")
            st.text(classification_report(y_test, y_pred))

            # -------------------------------
            # Confusion Matrix
            # -------------------------------
            st.subheader("📊 Confusion Matrix")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
            st.pyplot(fig)

            # -------------------------------
            # Full dataset prediction
            # -------------------------------
            full_predictions = model.predict(X)
            result_df = data.copy()
            result_df["Predicted_Attack_Type"] = full_predictions

            st.subheader("🧾 Prediction Results")
            st.dataframe(result_df.head())

            # -------------------------------
            # Download results
            # -------------------------------
            csv_output = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Prediction Results",
                csv_output,
                "attack_predictions.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"Error: {e}")
