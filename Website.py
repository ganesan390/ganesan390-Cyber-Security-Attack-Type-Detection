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
    target_column = st.selectbox("Select_
