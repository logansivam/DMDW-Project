import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Setup the Look
st.set_page_config(page_title="Driver Efficiency Dashboard", layout="wide")

# 2. Load your saved model
@st.cache_resource
def load_model():
    return joblib.load('delivery_model.pkl')

model = load_model()

# 3. Sidebar (The Interface part)
st.sidebar.header("Input Parameters")

def get_input():
    age = st.sidebar.slider("Driver Age", 18, 60, 25)
    ratings = st.sidebar.slider("Driver Rating", 1.0, 5.0, 4.5)
    weather = st.sidebar.selectbox("Weather", ["Sunny", "Stormy", "Sandstorms", "Windy", "Cloudy", "Fog"])
    traffic = st.sidebar.selectbox("Traffic Density", ["Low", "Medium", "High", "Jam"])
    
    # Create a dataframe to match your model's expected input
    data = {'Delivery_person_Age': age, 'Delivery_person_Ratings': ratings}
    # Note: You'll need to map Weather/Traffic to numbers if your model uses them
    return pd.DataFrame(data, index=[0])

user_data = get_input()

# 4. Main Panel
st.title("🚚 Delivery Driver Efficiency Project")
st.write("Predicting how many orders can be completed based on your DMDW project logic.")

if st.button("Predict Efficiency"):
    prediction = model.predict(user_data)
    st.success(f"Estimated Result: {prediction[0]:.2f}")
    
    # You can also show the plots you made in your DMDW_Project.py here
    st.image("rating_by_age.png") # If your script saves the plot as an image