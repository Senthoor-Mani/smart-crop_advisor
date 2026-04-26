#   Smart Crop Advisor (XGBoost + WeatherAPI)

## What this project does
- Uses Kaggle Crop Recommendation dataset features:
  N, P, K, temperature, humidity, ph, rainfall → predicts crop
- Uses WeatherAPI city search (/search.json) + forecast (/forecast.json)
- Shows Top-5 crops with confidence + chart

## Run locally
```bash
pip install -r requirements.txt
export WEATHERAPI_KEY=YOUR_KEY
streamlit run app.py
```
