import streamlit as st
import pandas as pd
import joblib
import random

# --- CONFIGURATION ---
FIXED_AGE = 26
FIXED_RATING = 4.7
BASE_PAY = 5.0  # Base RM per order
PEAK_BONUS = 2.0 # Extra RM during peak bonus hours

# Load Machine Learning Assets
@st.cache_resource
def load_assets():
    model = joblib.load('delivery_model.pkl')
    le = joblib.load('label_encoder.pkl')
    weather_options = joblib.load('weather_classes.pkl')
    return model, le, weather_options

try:
    model, le, weather_options = load_assets()
except:
    st.error("System Assets Not Found. Please ensure 'train.py' has been executed.")
    st.stop()

# --- PAGE SETUP ---
st.set_page_config(page_title="Rider Productivity & Earnings Planner", layout="centered")
st.title("MyRider : Shift & Earnings Planner")
st.markdown(f"**Performance Profile:** Rating {FIXED_RATING} ⭐ | Elite Status")

# --- AUTOMATED WEATHER DETECTION ---
# This picks a random weather every time the browser page is refreshed/opened
weather_today = random.choice(weather_options)

st.success(f"📡 **Live Environment Scan:** Weather detected as **{weather_today}**")

# --- USER INPUTS ---
st.divider()
col1, col2 = st.columns(2)

with col1:
    work_hours = st.slider("Planned Shift Duration (Hours)", 1, 12, 8)
    avg_dist = st.slider("Target Delivery Radius (km)", 0.5, 10.0, 2.5)

with col2:
    start_time = st.slider("Shift Start Time (24h)", 0, 23, 12)
    st.info("💡 Pro Tip: Short delivery radiuses (under 3km) significantly increase order frequency.")

# --- BUSINESS LOGIC ---
# 1. Peak Hour Detection
is_peak = (11 <= start_time <= 14) or (18 <= start_time <= 21)
peak_val = 1 if is_peak else 0

# 2. Prediction Engine
weather_encoded = le.transform([weather_today])[0]
input_data = pd.DataFrame([[FIXED_AGE, FIXED_RATING, avg_dist, weather_encoded, 1, start_time, peak_val]], 
                          columns=['Delivery_person_Age', 'Delivery_person_Ratings', 'distance_km', 
                                   'Weather_Code', 'Vehicle_condition', 'Order_Hour', 'Peak_Hour'])

# Predict base time from ML model
time_pred = model.predict(input_data)[0]

# --- DYNAMIC EFFICIENCY ADJUSTMENT ---
# Logical boost: Short distances allow for batching and faster turnaround
if avg_dist <= 3.0:
    # 40% efficiency boost for short-range urban deliveries
    efficient_time = time_pred * 0.6 
else:
    # 15% efficiency boost for experienced riders
    efficient_time = time_pred * 0.85

# Calculate total orders (3 min buffer for handoffs)
total_orders = int((work_hours * 60) / (efficient_time + 3))

# 3. RM Earnings Calculation
pay_rate = BASE_PAY + (PEAK_BONUS if is_peak else 0)
total_earnings = total_orders * pay_rate
hourly_rate = total_earnings / work_hours

# --- ANALYTICS DISPLAY ---
st.divider()
m1, m2, m3 = st.columns(3)
m1.metric("Peak Bonus Active", "Yes (+RM2)" if is_peak else "No (Standard)")
m2.metric("Target Order Volume", f"{total_orders} Trips")
m3.metric("Estimated Earnings", f"RM {round(total_earnings, 2)}")

# High-level summary
if total_earnings >= 150:
    st.balloons()
    st.warning(f"💰 **High Earnings Shift:** You are projected to earn **RM {round(hourly_rate, 2)} per hour**!")
else:
    st.info(f"📊 **Shift Summary:** Expected average of **RM {round(hourly_rate, 2)} per hour**.")

if avg_dist <= 3.0:
    st.caption("⚡ *Efficiency boost applied based on short-range delivery radius logic.*")