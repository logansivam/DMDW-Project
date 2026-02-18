import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(page_title="Driver Efficiency Predictor", layout="wide")

# 2. Load the pre-trained model and encoder
@st.cache_resource
def load_assets():
    model = joblib.load('delivery_model.pkl')
    encoder = joblib.load('label_encoder.pkl')
    return model, encoder

try:
    model, le = load_assets()
except:
    st.error("Model files not found. Please upload 'delivery_model.pkl' and 'label_encoder.pkl'.")
    st.stop()

# 3. Sidebar Interface for User Inputs
st.sidebar.header("Rider & Trip Details")

def get_user_inputs():
    age = st.sidebar.slider("Rider Age", 18, 60, 30)
    rating = st.sidebar.slider("Rider Rating", 1.0, 5.0, 4.5)
    distance = st.sidebar.number_input("Distance (km)", min_value=1.0, max_value=100.0, value=5.0)
    
    # Selection boxes based on your dataset categories
    weather = st.sidebar.selectbox("Weather Condition", 
                                   ["conditions Sunny", "conditions Stormy", "conditions Sandstorms", 
                                    "conditions Windy", "conditions Cloudy", "conditions Fog"])
    
    traffic = st.sidebar.selectbox("Traffic Density", ["Low ", "Medium ", "High ", "Jam "])
    
    vehicle_cond = st.sidebar.selectbox("Vehicle Condition (0-2)", [0, 1, 2])
    order_hour = st.sidebar.slider("Hour of Day (24h)", 0, 23, 18)
    
    # Derived feature: Peak Hour (Matching your notebook logic)
    peak_hour = 1 if (11 <= order_hour <= 14 or 18 <= order_hour <= 21) else 0
    
    return pd.DataFrame({
        'Delivery_person_Age': [age],
        'Delivery_person_Ratings': [rating],
        'distance_km': [distance],
        'Weather_Code': [le.transform([weather])[0]], # Encode weather
        'Traffic_Code': [0], # Note: You may need a separate encoder for traffic if used in training
        'Vehicle_condition': [vehicle_cond],
        'Order_Hour': [order_hour],
        'Peak_Hour': [peak_hour]
    }), weather, traffic

input_df, weather_label, traffic_label = get_user_inputs()

# 4. Main Panel Interface
st.title("🚚 Food Delivery Efficiency Predictor")
st.markdown("This dashboard estimates how many orders a driver can complete based on current conditions.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Selected Parameters")
    st.write(f"**Weather:** {weather_label}")
    st.write(f"**Traffic:** {traffic_label}")
    st.write(f"**Time:** {input_df['Order_Hour'][0]}:00")

with col2:
    # 5. Perform Prediction
    prediction_time = model.predict(input_df)[0]
    
    # Calculate Estimated Orders per 8-hour shift (Matching your notebook logic)
    estimated_orders = (8 * 60) / prediction_time
    
    st.subheader("Prediction Result")
    st.metric(label="Est. Delivery Time", value=f"{round(prediction_time, 1)} min")
    st.metric(label="Total Orders / Day (8h)", value=f"{int(estimated_orders)} Orders")

st.divider()
st.info("Performance Insight: Higher ratings and clearer weather significantly boost delivery volume.")