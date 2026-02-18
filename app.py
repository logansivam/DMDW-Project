import streamlit as st
import pandas as pd
import joblib

# Set Page Title
st.set_page_config(page_title="Delivery Order Predictor", layout="wide")

# --- Load Model and Encoder ---
@st.cache_resource
def load_data():
    model = joblib.load('delivery_model.pkl')
    le = joblib.load('label_encoder.pkl')
    return model, le

try:
    model, le = load_data()
except:
    st.error("Error: 'delivery_model.pkl' or 'label_encoder.pkl' not found. Please run your training script first.")
    st.stop()

# --- Sidebar Interface ---
st.sidebar.header("Shift & Rider Details")

def user_input_features():
    age = st.sidebar.slider("Rider Age", 18, 60, 30)
    rating = st.sidebar.slider("Rider Rating", 1.0, 5.0, 4.5)
    distance = st.sidebar.number_input("Average Trip Distance (km)", 1.0, 50.0, 5.0)
    
    # Selection boxes based on your notebook labels
    weather = st.sidebar.selectbox("Current Weather", 
        ["conditions Sunny", "conditions Stormy", "conditions Sandstorms", "conditions Windy", "conditions Cloudy", "conditions Fog"])
    
    traffic = st.sidebar.selectbox("Traffic Density", ["Low ", "Medium ", "High ", "Jam "])
    
    vehicle = st.sidebar.selectbox("Vehicle Condition (0=Poor, 2=Excellent)", [0, 1, 2])
    hour = st.sidebar.slider("Start Hour (24h format)", 0, 23, 12)
    
    # Calculate Peak Hour (Logic from your notebook Cell 152)
    peak_hour = 1 if (11 <= hour <= 14 or 18 <= hour <= 21) else 0
    
    # Map weather to code using your encoder
    weather_code = le.transform([weather])[0]
    
    # Create DataFrame for prediction
    data = {
        'Delivery_person_Age': [age],
        'Delivery_person_Ratings': [rating],
        'distance_km': [distance],
        'Weather_Code': [weather_code],
        'Traffic_Code': [1], # Mapping logic based on your notebook fit_transform
        'Vehicle_condition': [vehicle],
        'Order_Hour': [hour],
        'Peak_Hour': [peak_hour]
    }
    return pd.DataFrame(data), weather, traffic

input_df, weather_lbl, traffic_lbl = user_input_features()

# --- Main Interface ---
st.title("🚚 Delivery Driver Prediction System")
st.write("Determine how many orders you can realistically complete before your shift starts.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Shift Parameters")
    st.write(f"**Weather:** {weather_lbl}")
    st.write(f"**Traffic:** {traffic_lbl}")
    st.write(f"**Peak Hour:** {'Yes' if input_df['Peak_Hour'][0] == 1 else 'No'}")

with col2:
    # Perform Prediction
    predicted_time = model.predict(input_df)[0]
    
    # Calculate Estimated Orders (Logic from your notebook Cell 158: 8-hour shift)
    working_minutes = 8 * 60
    estimated_orders = working_minutes / predicted_time
    
    st.subheader("Your Prediction Results")
    st.metric(label="Est. Time per Delivery", value=f"{round(predicted_time, 1)} min")
    st.success(f"### Predicted Orders for an 8hr Shift: {int(estimated_orders)}")

st.divider()
st.info("💡 Note: This prediction assumes a standard 8-hour shift based on your performance data.")