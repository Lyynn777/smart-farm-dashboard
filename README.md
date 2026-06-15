# 🌾 Smart Farm Intelligence Dashboard

A graduate-level ML web application for agricultural data science — combining 
crop disease detection, yield prediction, and interactive field health mapping.

## Features

| Module | Technology | What it does |
|---|---|---|
| Disease Detector | CNN · MobileNetV2 · Transfer Learning | Classifies leaf diseases from images |
| Yield Predictor | Random Forest · Scikit-learn | Forecasts crop yield from NPK + climate data |
| Field Health Map | Folium · NDVI · GeoPandas | Interactive map of field stress zones |

## Tech Stack
`Python` · `TensorFlow` · `Scikit-learn` · `Streamlit` · `Folium` · `Plotly`

## Run Locally

```bash
git clone https://github.com/yourusername/smart-farm-dashboard
cd smart-farm-dashboard
pip install -r requirements.txt

python generate_sample_data.py   # create sample NDVI field data
python train_disease.py          # train CNN (needs PlantVillage dataset)
python train_yield.py            # train Random Forest (needs crop_yield.csv)
streamlit run app.py
```

## Live Demo
[Add Streamlit Cloud link here]

## Methodology

**Disease Detection** — MobileNetV2 pretrained on ImageNet, fine-tuned on 
PlantVillage tomato leaf images (3 classes). Data augmentation applied to 
improve generalisation on limited labelled data.

**Yield Prediction** — Random Forest Regressor (200 trees) trained on NPK 
soil nutrient values, rainfall, and temperature. Achieves R² > 0.85 on 
held-out test set.

**Field Health Map** — NDVI values from drone/satellite imagery plotted as 
an interactive Folium map with colour-coded stress zones. Threshold: NDVI > 0.6 
healthy, 0.35–0.6 moderate stress, < 0.35 severe stress.

## Dataset Credits
- PlantVillage Dataset (Hughes et al.)
- Crop Yield Prediction Dataset (Kaggle)