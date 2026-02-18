import streamlit as st
import pandas as pd
import joblib
import random

# --- CONFIGURATION ---
FIXED_AGE = 26
FIXED_RATING = 4.7
BASE_PAY = 5.0  
PEAK_BONUS = 2.0 

# Load Assets
@st.cache_resource
def load_assets():
    model = joblib.load('delivery_model.pkl')
    le = joblib.load('label_encoder.pkl')
    weather_options = joblib.load('weather_classes.pkl')
    return model, le, weather_options

try:
    model, le, weather_options = load_assets()
except:
    st.error("Assets missing. Please run train.py first.")
    st.stop()

# --- INITIALIZE SESSION STATE ---
if 'submitted' not in st.session_state:
    st.session_state.submitted = False
if 'today_weather' not in st.session_state:
    st.session_state.today_weather = random.choice(weather_options)

# --- PAGE SETUP ---
st.set_page_config(page_title="Rider Intelligence", layout="centered")
st.title("MyRider : Shift & Earnings Planner")

weather_today = st.session_state.today_weather
st.info(f"🌦️ **Detected Weather:** {weather_today}")

# --- INPUT SECTION ---
st.divider()
col1, col2 = st.columns(2)

with col1:
    work_hours = st.slider("Planned Shift Duration (Hours)", 0, 12, 0)
    avg_dist = st.slider("Target Delivery Radius (km)", 0.0, 10.0, 0.0)

with col2:
    start_time = st.slider("Shift Start Time (24h)", 0, 23, 0)
    
st.write("") 

# --- SUBMIT LOGIC WITH VALIDATION ---
if st.button("Generate My Schedule"):
    if work_hours > 0 and avg_dist > 0:
        st.session_state.submitted = True
        st.session_state.error = False
    else:
        st.session_state.submitted = False
        st.session_state.error = True

if 'error' in st.session_state and st.session_state.error:
    st.error("⚠️ Please set both **Shift Duration** and **Delivery Radius** above 0 to generate your plan.")

# --- HIDDEN RESULTS SECTION ---
if st.session_state.submitted:
    st.divider()
    
    # Logic
    is_peak = (11 <= start_time <= 14) or (18 <= start_time <= 21)
    peak_val = 1 if is_peak else 0
    weather_encoded = le.transform([weather_today])[0]
    
    input_data = pd.DataFrame([[FIXED_AGE, FIXED_RATING, avg_dist, weather_encoded, 1, start_time, peak_val]], 
                              columns=['Delivery_person_Age', 'Delivery_person_Ratings', 'distance_km', 
                                       'Weather_Code', 'Vehicle_condition', 'Order_Hour', 'Peak_Hour'])

    time_pred = model.predict(input_data)[0]

    # Efficiency Logic
    efficient_time = time_pred * 0.6 if avg_dist <= 3.0 else time_pred * 0.85
    total_orders = int((work_hours * 60) / (efficient_time + 3))
    
    pay_rate = BASE_PAY + (PEAK_BONUS if is_peak else 0)
    total_earnings = total_orders * pay_rate

    # Display
    if total_earnings >= 150:
        st.balloons()
        st.success(f"🔥 **High Productivity Goal:** Complete {total_orders} orders to earn **RM {round(total_earnings, 2)}**!")
    else:
        st.info(f"✅ **Daily Goal:** Complete {total_orders} orders to earn **RM {round(total_earnings, 2)}**.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Peak Hour Bonus", "Yes ✅" if is_peak else "No ❌")
    m2.metric("Target Orders", f"{total_orders}")

    m3.metric("Est. Total", f"RM {round(total_earnings, 2)}")
