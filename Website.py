import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# 1. INITIALIZE THE VARIABLE FIRST
model = None 

# 2. DEFINE PATHS
current_dir = Path(__file__).parent
model_path = current_dir / "model.pkl"

# 3. DEFINE THE LOADING FUNCTION
@st.cache_resource
def load_my_model():
    if not model_path.exists():
        return None
    try:
        # Use the path variable, not the string 'model.pkl'
        return joblib.load(model_path)
    except Exception as e:
        # This will catch and display errors without crashing the whole app
        st.error(f"Technical error during loading: {e}")
        return None

# 4. ASSIGN THE MODEL
model = load_my_model()

# 5. NOW THE CHECK ON LINE 29 WILL WORK
if model is None:
    st.error("❌ **model.pkl** not found or could not be loaded.")
    st.stop() # Prevents the rest of the app from running

# --- PAGE CONFIG ---
st.set_page_config(page_title="Cyber Attack Detector", layout="wide")

st.title("🛡️ Cyber Security Attack Type Detection")
st.markdown("Enter network traffic metrics below to predict potential threats.")

# --- INITIALIZE & LOAD MODEL ---
# --- PATH SETUP ---
current_dir = Path(__file__).parent
model_path = current_dir / "model.pkl"

@st.cache_resource
def load_my_model():
    if not model_path.exists():
        return None
    try:
        # FIX: Change 'model.pkl' to 'model_path'
        return joblib.load(model.pkl) 
    except Exception as e:
        st.error(f"Technical error during loading: {e}")
        return None

# --- VALIDATION CHECK ---
if model is None:
    st.error("❌ **model.pkl** not found or could not be loaded.")
    st.info(f"Make sure **model.pkl** is uploaded to your GitHub in the same folder as this script. Expected location: `{model_path}`")
    st.stop() 
else:
    st.success("✅ Model loaded successfully!")

# --- INPUT UI ---
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Network Info")
        src_port = st.number_input("Source Port", 0, 65535, 80)
        dest_port = st.number_input("Destination Port", 0, 65535, 443)
        protocol = st.selectbox("Protocol", [0, 1, 2, 3], format_func=lambda x: ["TCP", "UDP", "ICMP", "HTTP"][x])

    with col2:
        st.subheader("Traffic Metrics")
        pkt_len = st.number_input("Packet Length", 0, 65535, 512)
        traffic_type = st.selectbox("Traffic Type", [0, 1], format_func=lambda x: ["Inbound", "Outbound"][x])
        anomaly_score = st.slider("Anomaly Score", 0.0, 100.0, 10.0)

    with col3:
        st.subheader("Security Indicators")
        severity = st.selectbox("Severity Level", [0, 1, 2, 3], format_func=lambda x: ["Low", "Medium", "High", "Critical"][x])
        malware_ind = st.selectbox("Malware Indicators", [0, 1], format_func=lambda x: ["None", "Detected"][x])
        
    submit = st.form_submit_button("Analyze Traffic")

# --- PREDICTION LOGIC ---
if submit:
    # 1. Map UI inputs to a dictionary
    ui_data = {
        "Source Port": [src_port],
        "Destination Port": [dest_port],
        "Protocol": [protocol],
        "Packet Length": [pkt_len],
        "Traffic Type": [traffic_type],
        "Anomaly Scores": [anomaly_score],
        "Severity Level": [severity],
        "Malware Indicators": [malware_ind]
    }
    
    input_df = pd.DataFrame(ui_data)

    # 2. FEATURE ALIGNMENT (The Fix for the 25 metrics)
    # This automatically adds the missing 17 columns as 0s so the model doesn't crash
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0 

    # 3. Ensure columns are in the EXACT order the model learned
    input_df = input_df[model.feature_names_in_]

    try:
        # 4. Predict
        prediction = model.predict(input_df)
        
        st.divider()
        st.subheader("Analysis Result:")
        
        if str(prediction[0]).lower() != "normal":
            st.error(f"⚠️ **Threat Detected: {prediction[0]}**")
        else:
            st.balloons()
            st.success(f"✅ **Traffic is Normal**")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
