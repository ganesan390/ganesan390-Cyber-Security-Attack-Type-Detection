import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# 1. SETUP PATHS
# Path(__file__).parent finds the folder where Website.py lives
current_dir = Path(__file__).parent
model_path = current_dir / "model.pkl"

# 2. LOAD MODEL FUNCTION
@st.cache_resource
def load_my_model():
    # Check if file exists first to avoid the 'Unbound' error
    if not model_path.exists():
        return None  # Return None so we can handle it gracefully below
    
    try:
        loaded_model = joblib.load(model_path)
        return loaded_model
    except Exception as e:
        st.error(f"⚠️ Technical error during loading: {e}")
        return None

# 3. CALL THE FUNCTION
model = load_my_model()

# 4. CONDITIONAL UI
if model is None:
    st.error("❌ **model.pkl** not found or could not be loaded.")
    st.info(f"Make sure **model.pkl** is uploaded to your GitHub in the same folder as this script. Expected location: `{model_path}`")
    st.stop() # Stops the app here so it doesn't try to use 'model' later
else:
    st.success("✅ Model loaded successfully!")
    # ... rest of your code (the form, prediction logic, etc.)

# --- INPUT UI ---
if model is not None:
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Network Info")
            src_port = st.number_input("Source Port", 0, 65535, 80)
            dest_port = st.number_input("Destination Port", 0, 65535, 443)
            # Match the categories exactly to your dataset
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
        # 1. MUST: Match the EXACT column names used in Colab Step 2/3
        input_dict = {
            "Source Port": [src_port],
            "Destination Port": [dest_port],
            "Protocol": [protocol],
            "Packet Length": [pkt_len],
            "Traffic Type": [traffic_type],
            "Anomaly Scores": [anomaly_score],
            "Severity Level": [severity],
            "Malware Indicators": [malware_ind]
        }
        
        input_df = pd.DataFrame(input_dict)

        try:
            # 2. Predict using the loaded model
            prediction = model.predict(input_df)
            
            st.success(f"### Prediction Result: {prediction[0]}")
            
            if str(prediction[0]).lower() != "normal":
                st.warning("⚠️ Threat Detected!")
            else:
                st.balloons()
                st.info("✅ Traffic is Normal.")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.info("Tip: Ensure the number of columns in 'input_dict' matches what the model was trained on.")

else:
    st.warning("⚠️ model.pkl not found in repository.")
