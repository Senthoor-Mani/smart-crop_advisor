def summarize_forecast(forecast_json):
    days = forecast_json["forecast"]["forecastday"]
    avg_temp = sum(d["day"]["avgtemp_c"] for d in days) / len(days)
    avg_hum  = sum(d["day"]["avghumidity"] for d in days) / len(days)
    total_rain = sum(d["day"]["totalprecip_mm"] for d in days)
    return {
        "temperature": float(avg_temp),
        "humidity": float(avg_hum),
        "rainfall": float(total_rain),
        "days_used": int(len(days)),
    }
