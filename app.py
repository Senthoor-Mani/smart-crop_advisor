import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

from src.weatherapi_client import search_city, forecast, WeatherAPIError
from src.features import summarize_forecast

st.set_page_config(page_title="Smart Crop Advisor", page_icon="🌾", layout="wide")
st.title("🌾 Smart Crop Recommendation System (XGBoost + WeatherAPI)")
st.caption("Soil NPK + pH + Weather Forecast → Top-5 crop recommendation")

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/crop_model.joblib")
    le = joblib.load("models/label_encoder.joblib")
    return model, le

model, le = load_artifacts()

# --- Sidebar Soil Inputs ---
st.sidebar.header("🧪 Soil Inputs")
N = st.sidebar.slider("N (Nitrogen)", 0, 140, 80, 1)
P = st.sidebar.slider("P (Phosphorus)", 0, 140, 40, 1)
K = st.sidebar.slider("K (Potassium)", 0, 140, 40, 1)
ph = st.sidebar.slider("Soil pH", 0.0, 14.0, 6.5, 0.1)

st.sidebar.header("🌦 Weather Settings")
try_days = st.sidebar.selectbox("Forecast days (try)", [14, 7, 3])
manual_override = st.sidebar.checkbox("Manual override weather", False)

# --- City search + dropdown ---
st.subheader("📍 Select City (India)")
query = st.text_input("Type city name (e.g., Pune, Delhi, Chennai)", "Pune")

city = query.strip()
suggestions = []
if len(query) >= 2:
    try:
        results = search_city(query)
        india = [r for r in results if str(r.get("country","")).lower() == "india"]
        suggestions = india if india else results
    except WeatherAPIError as e:
        st.error(str(e))

if suggestions:
    options = [
        f"{r['name']}, {r.get('region','')}, {r.get('country','')}".replace(" ,", ",")
        for r in suggestions
    ]
    picked = st.selectbox("Suggestions", options)
    city = picked.split(",")[0].strip()

# --- Weather fetch ---
st.subheader("🌦 Weather Forecast")
if st.button("Fetch Weather"):
    try:
        try:
            wjson = forecast(city, days=int(try_days))
        except WeatherAPIError:
            # fallback to 3 days (common for free plans)
            wjson = forecast(city, days=3)

        wf = summarize_forecast(wjson)
        st.session_state["wf"] = wf
        st.success(f"Forecast fetched for {city} (days used: {wf['days_used']})")

    except Exception as e:
        st.error(f"Weather fetch failed: {e}")

wf = st.session_state.get("wf")

if wf:
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Temp (°C)", round(wf["temperature"], 2))
    c2.metric("Avg Humidity (%)", round(wf["humidity"], 2))
    c3.metric("Total Rain (mm)", round(wf["rainfall"], 2))

# --- Values to model ---
if wf and not manual_override:
    temperature = float(wf["temperature"])
    humidity = float(wf["humidity"])
    rainfall = float(wf["rainfall"])
else:
    temperature = st.slider("Temperature (°C)", 0.0, 50.0, float(wf["temperature"] if wf else 25.0), 0.1)
    humidity = st.slider("Humidity (%)", 0.0, 100.0, float(wf["humidity"] if wf else 70.0), 1.0)
    rainfall = st.slider("Rainfall (mm)", 0.0, 500.0, float(wf["rainfall"] if wf else 100.0), 1.0)

# --- Predict Top-5 ---
st.subheader("✅ Crop Recommendations (Top-5)")
X = pd.DataFrame([{
    "N": N, "P": P, "K": K,
    "temperature": temperature,
    "humidity": humidity,
    "ph": ph,
    "rainfall": rainfall
}])

proba = model.predict_proba(X)[0]
class_ids = list(range(len(le.classes_)))

df = pd.DataFrame({"crop_id": class_ids, "probability": proba})
df["crop"] = le.inverse_transform(df["crop_id"].astype(int))

top5 = df.sort_values("probability", ascending=False).head(5)
best = top5.iloc[0]

st.success(f"🌟 Best crop: **{best['crop']}** (confidence {best['probability']:.2%})")

fig = px.bar(top5, x="crop", y="probability", title="Top 5 Crop Suitability")
st.plotly_chart(fig, use_container_width=True)
st.dataframe(top5[["crop","probability"]], use_container_width=True)

st.caption("WeatherAPI uses `key=` and `q=` parameters; forecast days 1–14; app falls back to 3 if needed.")
