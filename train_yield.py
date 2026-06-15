import pandas as pd
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset
df = pd.read_csv("data/crop_yield.csv")

# Clean column names
df.columns = df.columns.str.strip()

print("Columns found:")
print(df.columns.tolist())

# Create encoders
region_encoder = LabelEncoder()
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
weather_encoder = LabelEncoder()

# Encode categorical columns
df["Region_encoded"] = region_encoder.fit_transform(df["Region"])
df["Soil_encoded"] = soil_encoder.fit_transform(df["Soil_Type"])
df["Crop_encoded"] = crop_encoder.fit_transform(df["Crop"])
df["Weather_encoded"] = weather_encoder.fit_transform(df["Weather_Condition"])

# Convert booleans to integers
df["Fertilizer_Used"] = df["Fertilizer_Used"].astype(int)
df["Irrigation_Used"] = df["Irrigation_Used"].astype(int)

# Features and target
FEATURES = [
    "Region_encoded",
    "Soil_encoded",
    "Crop_encoded",
    "Rainfall_mm",
    "Temperature_Celsius",
    "Fertilizer_Used",
    "Irrigation_Used",
    "Weather_encoded",
    "Days_to_Harvest"
]

TARGET = "Yield_tons_per_hectare"

X = df[FEATURES]
y = df[TARGET]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

print("Training yield prediction model...")
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

print(f"R² Score : {r2_score(y_test, y_pred):.4f}")
print(f"MAE      : {mean_absolute_error(y_test, y_pred):.4f}")

# Save model
pickle.dump(model, open("model/yield_model_small.pkl", "wb"))

# Save encoders
pickle.dump(region_encoder, open("model/region_encoder.pkl", "wb"))
pickle.dump(soil_encoder, open("model/soil_encoder.pkl", "wb"))
pickle.dump(crop_encoder, open("model/crop_encoder.pkl", "wb"))
pickle.dump(weather_encoder, open("model/weather_encoder.pkl", "wb"))

print("\nFiles saved:")
print("✓ model/yield_model.pkl")
print("✓ model/region_encoder.pkl")
print("✓ model/soil_encoder.pkl")
print("✓ model/crop_encoder.pkl")
print("✓ model/weather_encoder.pkl")