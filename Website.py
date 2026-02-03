import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# --- PAGE CONFIG ---
st.set_page_config(page_title="Cyber Attack Detector", layout="wide")

st.title("🛡️ Cyber Security Attack Type Detection")
st.markdown("Enter the network traffic metrics below to analyze the attack type.")

# --- LOAD MODEL ---
# Using Path ensures it finds the file in your repo root
model_path = Path(__file__).parent / "model.pkl"

@st.cache_resource
def load_my_model():
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error: Could not load model.pkl. Specific error: {e}")
        return None

model = load_my_model()

# --- INPUT UI ---
if model:
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Network Info")
            src_ip = st.text_input("Source IP Address", "192.168.1.1")
            dest_ip = st.text_input("Destination IP Address", "10.0.0.1")
            src_port = st.number_input("Source Port", 0, 65535, 80)
            dest_port = st.number_input("Destination Port", 0, 65535, 443)
            protocol = st.selectbox("Protocol", ["TCP", "UDP", "ICMP", "HTTP"])

        with col2:
            st.subheader("Traffic Metrics")
            pkt_len = st.number_input("Packet Length", 0, 65535, 512)
            pkt_type = st.selectbox("Packet Type", ["Data", "Control", "Management"])
            traffic_type = st.selectbox("Traffic Type", ["Inbound", "Outbound"])
            anomaly_score = st.slider("Anomaly Score", 0.0, 100.0, 10.0)
            severity = st.selectbox("Severity Level", ["Low", "Medium", "High", "Critical"])

        with col3:
            st.subheader("Security Indicators")
            malware_ind = st.selectbox("Malware Indicators", ["None", "Detected"])
            alerts = st.selectbox("Alerts/Warnings", ["None", "Low Alert", "High Alert"])
            proxy_info = st.text_input("Proxy Information", "None")
            user_info = st.text_input("User Information", "Standard User")
            
        submit = st.form_submit_button("Analyze Traffic")

    # --- PREDICTION LOGIC ---
    if submit:
        # Create a dataframe with the 25 metrics 
        # (Ensure names match the features used during model training)
        input_data = pd.DataFrame({
            "Source Port": [src_port],
            "Destination Port": [dest_port],
            "Packet Length": [pkt_len],
            "Anomaly Scores": [anomaly_score],
            # Add all other features required by your specific model here...
        })

        try:
            prediction = model.predict(input_data)
            
            st.success(f"### Prediction Result: {prediction[0]}")
            
            # Visual feedback based on result
            if prediction[0].lower() != "normal":
                st.warning("⚠️ High Risk Detected!")
            else:
                st.balloons()
                st.info("✅ Traffic appears Normal.")
                
        except Exception as e:
            st.error(f"Prediction failed. Ensure the input data matches the model's training features. Error: {e}")

else:
    st.warning("⚠️ Please upload 'model.pkl' to your GitHub repository to enable predictions.")
