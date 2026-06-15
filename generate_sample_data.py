import numpy as np
import pandas as pd

np.random.seed(42)
n = 60

# Simulate a grid of GPS points around Bengaluru farmland
lats = np.random.uniform(13.020, 13.035, n)
lons = np.random.uniform(77.560, 77.575, n)

# Create realistic NDVI clusters: some stressed, some healthy
ndvi_base = np.random.choice([0.2, 0.5, 0.75], n, p=[0.25, 0.40, 0.35])
ndvi = np.clip(ndvi_base + np.random.normal(0, 0.08, n), 0.05, 0.95)

df = pd.DataFrame({
    'lat': lats,
    'lon': lons,
    'ndvi': np.round(ndvi, 3),
    'zone': ['North' if lat > 13.028 else 'South' for lat in lats]
})

df.to_csv('data/sample_field_ndvi.csv', index=False)
print(f"Generated {n} field data points → data/sample_field_ndvi.csv")
print(df.head())