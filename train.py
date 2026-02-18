import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression # Changed this
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# 1. Load and Clean (Same as before)
df = pd.read_csv('train.csv')
df['Time_taken(min)'] = pd.to_numeric(df['Time_taken(min)'].astype(str).str.extract(r'(\d+)', expand=False), errors='coerce')

# 2. Feature Engineering (Same as before)
def haversine(lat1, lon1, lat2, lon2):
    r = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlambda = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * r * np.arcsin(np.sqrt(a))

df['distance_km'] = haversine(df['Restaurant_latitude'], df['Restaurant_longitude'], 
                              df['Delivery_location_latitude'], df['Delivery_location_longitude'])

df['Order_Hour'] = pd.to_numeric(df['Time_Orderd'].astype(str).str.split(':').str[0], errors='coerce')
df['Peak_Hour'] = df['Order_Hour'].apply(lambda x: 1 if (11 <= x <= 14 or 18 <= x <= 21) else 0)

# 3. Encoding & Imputation
le = LabelEncoder()
df['Weather_Code'] = le.fit_transform(df['Weatherconditions'].astype(str).str.strip())

features = ['Delivery_person_Age', 'Delivery_person_Ratings', 'distance_km', 
            'Weather_Code', 'Vehicle_condition', 'Order_Hour', 'Peak_Hour']

imputer = SimpleImputer(strategy='median')
df[features] = imputer.fit_transform(df[features])
df = df.dropna(subset=['Time_taken(min)'])

# --- 4. NEW LINEAR REGRESSION MODEL ---
# No arguments needed here!
model = LinearRegression() 
model.fit(df[features], df['Time_taken(min)'])

# 5. Save Assets
joblib.dump(model, 'delivery_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(le.classes_, 'weather_classes.pkl')

import os
file_size = os.path.getsize('delivery_model.pkl') / 1024
print(f"--- SUCCESS! ---")
print(f"Model Size: {round(file_size, 2)} KB") # It will be very tiny now!