import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# 1. Load Data
df = pd.read_csv('train.csv')

# 2. Extract Numbers from Target (Fixes string errors)
df['Time_taken(min)'] = pd.to_numeric(df['Time_taken(min)'].astype(str).str.extract(r'(\d+)', expand=False), errors='coerce')

# 3. Feature Engineering
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

# 4. Encoding
le = LabelEncoder()
df['Weather_Code'] = le.fit_transform(df['Weatherconditions'].astype(str).str.strip())

# 5. Imputation (Fill missing values)
features = ['Delivery_person_Age', 'Delivery_person_Ratings', 'distance_km', 
            'Weather_Code', 'Vehicle_condition', 'Order_Hour', 'Peak_Hour']
imputer = SimpleImputer(strategy='median')
df[features] = imputer.fit_transform(df[features])
df = df.dropna(subset=['Time_taken(min)'])

# 6. Train and Save
model = LinearRegression(
    n_estimators=50, 
    max_depth=10, 
    min_samples_leaf=5, 
    random_state=42
)
model.fit(df[features], df['Time_taken(min)'])

joblib.dump(model, 'delivery_model.pkl', compress=3)
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(le.classes_, 'weather_classes.pkl')

print("--- Success: Driver Model Ready! ---")
