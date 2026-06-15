import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import json
import folium
from streamlit_folium import st_folium
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Smart Farm Intelligence",
    page_icon="🌾",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 700;
        color: #1a6b3c; margin-bottom: 0;
    }
    .subtitle {
        font-size: 0.95rem; color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f0f7f2; border-left: 4px solid #2d8a57;
        padding: 12px 16px; border-radius: 6px; margin: 6px 0;
    }
    .healthy { color: #1a6b3c; font-weight: 700; }
    .diseased { color: #c0392b; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Load models (cached) ─────────────────────────────────
@st.cache_resource
def load_disease_model():
    model = tf.keras.models.load_model('model/disease_model.h5')
    with open('model/class_names.json') as f:
        class_indices = json.load(f)
    classes = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]
    return model, classes

@st.cache_resource
def load_yield_model():
    model = pickle.load(open('model/yield_model.pkl', 'rb'))
    le = pickle.load(open('model/crop_encoder.pkl', 'rb'))
    return model, le

# ── Header ────────────────────────────────────────────────
st.markdown('<p class="main-title">🌾 Smart Farm Intelligence Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Crop disease detection · Yield forecasting · Field health mapping</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "🔬 Disease Detector",
    "📈 Yield Predictor",
    "🗺️ Field Health Map"
])

# ════════════════════════════════════════════════════════════
# TAB 1 — Disease Detector
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Crop Leaf Disease Detection")
    st.caption("Upload a leaf image — the CNN model (MobileNetV2 + transfer learning) identifies the disease.")

    disease_model, classes = load_disease_model()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        uploaded = st.file_uploader(
            "Upload leaf image", type=['jpg', 'png', 'jpeg'],
            help="Works best with close-up photos of a single leaf"
        )

    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        img_resized = img.resize((224, 224))

        with col1:
            st.image(img, caption="Uploaded leaf", use_column_width=True)

        arr = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
        preds = disease_model.predict(arr)[0]

        label = classes[np.argmax(preds)]
        confidence = float(np.max(preds)) * 100
        is_healthy = 'healthy' in label.lower()

        with col2:
            status_class = "healthy" if is_healthy else "diseased"
            icon = "✅" if is_healthy else "⚠️"
            st.markdown(f"### {icon} Prediction")
            st.markdown(f'<p class="{status_class}" style="font-size:1.4rem">{label.replace("_", " ")}</p>', unsafe_allow_html=True)
            st.metric("Confidence", f"{confidence:.1f}%")

            st.markdown("---")
            st.markdown("**Class probabilities**")
            fig = px.bar(
                x=[c.replace("_", " ") for c in classes],
                y=[round(float(p) * 100, 2) for p in preds],
                color=[float(p) for p in preds],
                color_continuous_scale='Greens',
                labels={'x': 'Disease class', 'y': 'Confidence (%)'}
            )
            fig.update_layout(
                showlegend=False, coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=10, b=0), height=260
            )
            st.plotly_chart(fig, use_container_width=True)

            if not is_healthy:
                st.warning("Disease detected. Recommend field inspection and treatment.")
    else:
        st.info("Upload a leaf image to get started.")

# ════════════════════════════════════════════════════════════
# TAB 2 — Yield Predictor
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Crop Yield Forecasting")
    st.caption("Input field conditions — Random Forest model predicts expected yield (tonnes/hectare).")

    yield_model, _ = load_yield_model()

    le_crop    = pickle.load(open('model/crop_encoder.pkl', 'rb'))
    le_region  = pickle.load(open('model/region_encoder.pkl', 'rb'))
    le_soil    = pickle.load(open('model/soil_encoder.pkl', 'rb'))
    le_weather = pickle.load(open('model/weather_encoder.pkl', 'rb'))

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown("**Crop & Region**")
        crop   = st.selectbox("Crop type", le_crop.classes_)
        region = st.selectbox("Region", le_region.classes_)
        soil   = st.selectbox("Soil type", le_soil.classes_)

    with col2:
        st.markdown("**Climate**")
        rainfall    = st.slider("Rainfall (mm)", 200, 3000, 1000, step=50)
        temperature = st.slider("Temperature (°C)", 10, 45, 25)
        weather     = st.selectbox("Weather condition", le_weather.classes_)

    with col3:
        st.markdown("**Farm inputs**")
        fertilizer = st.selectbox("Fertilizer used", ["Yes", "No"])
        irrigation = st.selectbox("Irrigation used", ["Yes", "No"])
        days       = st.slider("Days to harvest", 30, 300, 90)

    if st.button("Run Yield Prediction", type="primary", use_container_width=True):
        crop_enc    = le_crop.transform([crop])[0]
        region_enc  = le_region.transform([region])[0]
        soil_enc    = le_soil.transform([soil])[0]
        weather_enc = le_weather.transform([weather])[0]
        fert        = 1 if fertilizer == "Yes" else 0
        irrig       = 1 if irrigation == "Yes" else 0

        features = np.array([[region_enc, soil_enc, crop_enc,
                               rainfall, temperature, fert,
                               irrig, weather_enc, days]])

        prediction = yield_model.predict(features)[0]

        st.success(f"Predicted yield: **{prediction:.2f} tonnes/hectare**")

        low  = prediction * 0.85
        high = prediction * 1.15
        st.caption(f"Expected range: {low:.2f} – {high:.2f} t/ha")

        importance  = yield_model.feature_importances_
        feat_names  = ['Region', 'Soil', 'Crop', 'Rainfall', 'Temperature',
                        'Fertilizer', 'Irrigation', 'Weather', 'Days to Harvest']
        fi_df = pd.DataFrame({'Feature': feat_names, 'Importance': importance})
        fi_df = fi_df.sort_values('Importance', ascending=True)

        fig2 = px.bar(
            fi_df, x='Importance', y='Feature',
            orientation='h', title="What drives this prediction?",
            color='Importance', color_continuous_scale='Greens'
        )
        fig2.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=30, b=0), height=300
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Set parameters and click predict.")
# ════════════════════════════════════════════════════════════
# TAB 3 — Field Health Map
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Interactive Field NDVI Health Map")
    st.caption("Visualise crop health zones across a field using NDVI values from drone/satellite imagery.")

    use_sample = st.toggle("Use sample field data", value=True)

    if use_sample:
        try:
            df_map = pd.read_csv('data/sample_field_ndvi.csv')
        except FileNotFoundError:
            st.error("Run `python generate_sample_data.py` first to create sample data.")
            st.stop()
    else:
        csv_file = st.file_uploader(
            "Upload CSV with columns: lat, lon, ndvi",
            type=['csv']
        )
        if csv_file:
            df_map = pd.read_csv(csv_file)
            if not all(c in df_map.columns for c in ['lat', 'lon', 'ndvi']):
                st.error("CSV must have columns: lat, lon, ndvi")
                st.stop()
        else:
            st.info("Upload a CSV or enable sample data.")
            st.stop()

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        def ndvi_color(val):
            if val > 0.6:   return '#27ae60'  # healthy
            elif val > 0.35: return '#f39c12'  # moderate stress
            else:            return '#e74c3c'  # severe stress

        center_lat = df_map['lat'].mean()
        center_lon = df_map['lon'].mean()

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=15,
            tiles='CartoDB positron'
        )

        for _, row in df_map.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=10,
                color=ndvi_color(row['ndvi']),
                fill=True,
                fill_color=ndvi_color(row['ndvi']),
                fill_opacity=0.75,
                popup=folium.Popup(
                    f"<b>NDVI: {row['ndvi']:.3f}</b><br>"
                    f"Status: {'Healthy' if row['ndvi'] > 0.6 else 'Stressed'}",
                    max_width=150
                )
            ).add_to(m)

        # Legend
        legend_html = """
        <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                    background: white; padding: 10px 14px; border-radius: 8px;
                    border: 1px solid #ccc; font-size: 13px;">
            <b>NDVI Health</b><br>
            <span style="color:#27ae60">●</span> Healthy (&gt; 0.6)<br>
            <span style="color:#f39c12">●</span> Moderate (0.35–0.6)<br>
            <span style="color:#e74c3c">●</span> Stressed (&lt; 0.35)
        </div>"""
        m.get_root().html.add_child(folium.Element(legend_html))

        st_folium(m, width=None, height=450, use_container_width=True)

    with col2:
        st.markdown("**Field Summary**")

        healthy = (df_map['ndvi'] > 0.6).sum()
        moderate = ((df_map['ndvi'] > 0.35) & (df_map['ndvi'] <= 0.6)).sum()
        stressed = (df_map['ndvi'] <= 0.35).sum()
        total = len(df_map)

        st.metric("Average NDVI", f"{df_map['ndvi'].mean():.3f}")
        st.metric("🟢 Healthy zones", f"{healthy} points ({healthy/total*100:.0f}%)")
        st.metric("🟡 Moderate stress", f"{moderate} points ({moderate/total*100:.0f}%)")
        st.metric("🔴 Severe stress", f"{stressed} points ({stressed/total*100:.0f}%)")

        st.markdown("---")
        st.markdown("**NDVI Distribution**")
        fig3 = px.histogram(
            df_map, x='ndvi', nbins=20,
            color_discrete_sequence=['#2d8a57'],
            labels={'ndvi': 'NDVI value', 'count': 'Points'}
        )
        fig3.add_vline(x=0.35, line_dash="dash", line_color="#f39c12", annotation_text="Stress threshold")
        fig3.add_vline(x=0.6, line_dash="dash", line_color="#27ae60", annotation_text="Healthy threshold")
        fig3.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=250, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

        st.download_button(
            "Download field report (CSV)",
            data=df_map.to_csv(index=False),
            file_name="field_ndvi_report.csv",
            mime="text/csv",
            use_container_width=True
        )