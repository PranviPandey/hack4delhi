import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random
import requests
import math
import re
# Page configuration
st.set_page_config(page_title="Air Quality Dashboard", layout="wide", initial_sidebar_state="expanded")

WAQI_TOKEN = "c01d49af5769bc584fc51f7733c6fdcfedf47b3c"   # put your real token here
CITY = "delhi"

WARDS = [
    ("Connaught Place", 28.6315, 77.2167),
    ("Rohini", 28.7400, 77.1200),
    ("Saket", 28.5244, 77.2066),
    ("Laxmi Nagar", 28.6363, 77.2773),
    ("Dwarka", 28.5921, 77.0460),
    ("Karol Bagh", 28.6518, 77.1909),
    ("Janakpuri", 28.6219, 77.0878),
    ("Vasant Kunj", 28.5293, 77.1550),
    ("Pitampura", 28.7033, 77.1310),
    ("Mayur Vihar", 28.6092, 77.2928)
]

@st.cache_data(ttl=300)
def fetch_station_data():
    url = f"https://api.waqi.info/search/?token={WAQI_TOKEN}&keyword={CITY}"
    resp = requests.get(url, timeout=10).json()

    if resp["status"] != "ok":
        return []

    stations = []
    for s in resp["data"]:
        if s.get("aqi") != "-" and "station" in s:
            stations.append({
                "uid": s.get("uid"), # add uid so we can call feed/@uid later
                "lat": s["station"]["geo"][0],
                "lon": s["station"]["geo"][1],
                "aqi": int(s["aqi"])
            })
    return stations
def fetch_station_feed(uid):
    """Fetch IAQI (individual pollutants) for a station uid via WAQI feed"""
    try:
        url = f"https://api.waqi.info/feed/@{uid}/?token={WAQI_TOKEN}"
        resp = requests.get(url, timeout=10).json()
        if resp.get("status") != "ok":
            return {}
        iaqi = resp["data"].get("iaqi", {})
        out = {}
        if "pm25" in iaqi and "v" in iaqi["pm25"]:
            out["PM2.5"] = iaqi["pm25"]["v"]
        if "pm10" in iaqi and "v" in iaqi["pm10"]:
            out["PM10"] = iaqi["pm10"]["v"]
        if "no2" in iaqi and "v" in iaqi["no2"]:
            out["NO2"] = iaqi["no2"]["v"]
        if "so2" in iaqi and "v" in iaqi["so2"]:
            out["SO2"] = iaqi["so2"]["v"]
        if "co" in iaqi and "v" in iaqi["co"]:
            out["CO"] = iaqi["co"]["v"]
        if "o3" in iaqi and "v" in iaqi["o3"]:
            out["O3"] = iaqi["o3"]["v"]
        return out
    except Exception:
        return {}

def pm25_to_aqi(c):
    """Convert PM2.5 concentration (μg/m³) to US EPA AQI (integer)."""
    # breakpoints: (C_low, C_high, I_low, I_high)
    bps = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    if c is None:
        return None
    c = float(c)
    for cl, ch, il, ih in bps:
        if cl <= c <= ch:
            aqi = (ih - il) / (ch - cl) * (c - cl) + il
            return int(round(aqi))
    return 500 if c > 500.4 else 0

def fetch_hourly_pm25(lat, lon, hours=24, radius=5000):
    """
    Fetch PM2.5 measurements from OpenAQ around (lat,lon) for the last `hours`.
    Returns list of length `hours` with averages per hour (oldest -> newest), None if no data for hour.
    """
    end = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=hours - 1)
    url = (
        "https://api.openaq.org/v2/measurements"
        f"?coordinates={lat},{lon}&radius={radius}&parameter=pm25"
        f"&date_from={start.isoformat()}Z&date_to={(end + timedelta(hours=1)).isoformat()}Z&limit=1000"
    )
    try:
        resp = requests.get(url, timeout=10).json()
        results = resp.get("results", [])
    except Exception:
        results = []

    # bucket by UTC hour
    buckets = {}
    for r in results:
        date_utc = r.get("date", {}).get("utc")
        if not date_utc:
            continue
        try:
            dt = datetime.fromisoformat(date_utc.replace("Z", "+00:00"))
        except Exception:
            continue
        hour = dt.replace(minute=0, second=0, microsecond=0)
        buckets.setdefault(hour, []).append(r.get("value"))

    series = []
    for i in range(hours - 1, -1, -1):
        t = end - timedelta(hours=i)
        vals = buckets.get(t)
        series.append(float(np.mean(vals)) if vals else None)
    return series

def distance(lat1, lon1, lat2, lon2):
    return math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)


def compute_ward_aqi(ward_lat, ward_lon, stations):
    total, weight = 0, 0
    for s in stations:
        d = distance(ward_lat, ward_lon, s["lat"], s["lon"])
        w = 1 / (d + 0.0001)
        total += s["aqi"] * w
        weight += w
    return round(total / weight) if weight > 0 else random.randint(80, 300)

# Generate mock real-time data
@st.cache_data(ttl=300)
def generate_ward_data():
    wards = WARDS

    stations = fetch_station_data()
    data = []

    for ward, lat, lon in wards:
        aqi = compute_ward_aqi(lat, lon, stations)

        if aqi <= 50:
            category, color = "Good", "#00E400"
        elif aqi <= 100:
            category, color = "Moderate", "#FFFF00"
        elif aqi <= 150:
            category, color = "Unhealthy for Sensitive", "#FF7E00"
        elif aqi <= 200:
            category, color = "Unhealthy", "#FF0000"
        elif aqi <= 300:
            category, color = "Very Unhealthy", "#8F3F97"
        else:
            category, color = "Hazardous", "#7E0023"

        # get nearest station IAQI
        pollutants = {}
        if stations:
            nearest = min(stations, key=lambda s: distance(lat, lon, s["lat"], s["lon"]))
            if nearest.get("uid") is not None:
                pollutants = fetch_station_feed(nearest["uid"])

        pm25 = int(round(pollutants.get("PM2.5", random.randint(20, 150))))
        pm10 = int(round(pollutants.get("PM10", random.randint(30, 200))))
        no2 = int(round(pollutants.get("NO2", random.randint(10, 80))))
        so2 = int(round(pollutants.get("SO2", random.randint(5, 40))))
        co = int(round(pollutants.get("CO", random.randint(1, 15))))
        o3 = int(round(pollutants.get("O3", random.randint(20, 100))))

        data.append({
            "Ward": ward,
            "AQI": aqi,
            "Category": category,
            "Color": color,
            "PM2.5": pm25,
            "PM10": pm10,
            "NO2": no2,
            "SO2": so2,
            "CO": co,
            "O3": o3
        })

    return pd.DataFrame(data)


@st.cache_data(ttl=60)  # shorter TTL to make trend more responsive
def generate_trend_data(ward):
    # find ward coords
    row = next(((n, la, lo) for (n, la, lo) in WARDS if n == ward), None)
    if not row:
        # fallback to random series if ward not found
        dates = [datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)]
        return pd.DataFrame({"Time": dates, "AQI": [random.randint(80, 250) for _ in range(24)]})

    _, lat, lon = row
    pm25_series = fetch_hourly_pm25(lat, lon, hours=24, radius=5000)

    # if OpenAQ returned no data, try the nearest WAQI station current PM2.5 via fetch_station_feed
    if all(v is None for v in pm25_series):
        stations = fetch_station_data()
        if stations:
            nearest = min(stations, key=lambda s: distance(lat, lon, s["lat"], s["lon"]))
            feed = fetch_station_feed(nearest.get("uid")) if nearest.get("uid") else {}
            pm25_val = feed.get("PM2.5")
            if pm25_val is not None:
                # pm25_series = [pm25_val] * 24
                pm25_series = [
                    float(max(0, pm25_val + random.gauss(0, max(1.0, 0.05 * pm25_val))))
                    for _ in range(24)
                ]

            else:
                pm25_series = [random.randint(80, 150) for _ in range(24)]

    end = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    times = [end - timedelta(hours=i) for i in range(23, -1, -1)]
    aqi_values = [pm25_to_aqi(v) if v is not None else None for v in pm25_series]

    # fill missing AQI points by forward/backward fill where possible
    aqi_series = pd.Series(aqi_values).fillna(method="ffill").fillna(method="bfill").fillna(random.randint(80, 150)).tolist()
    return pd.DataFrame({"Time": times, "AQI": aqi_series})


@st.cache_data(ttl=300)
def generate_source_data(pm25, pm10, no2, so2, co, o3):
    """
    Infer source contributions deterministically from pollutant profile.
    """
    weights = {}
    weights["Vehicular Emissions"] = co * 3 + no2 * 2 + pm25 * 1.5
    weights["Industrial Activity"] = so2 * 4 + no2 * 1.5 + pm10 * 0.5
    weights["Construction Dust"] = pm10 * 3 + pm25 * 0.5
    weights["Residential Cooking"] = pm25 * 1 + co * 1
    weights["Waste Burning"] = pm25 * 1 + so2 * 1.5
    weights["Other"] = 5.0

    total = sum(weights.values()) or 1.0
    sources = {k: round((v / total) * 100, 1) for k, v in weights.items()}
    return pd.DataFrame(list(sources.items()), columns=["Source", "Contribution %"])

def get_health_recommendations(aqi):
    """
    Returns:
    - general_message: str
    - vulnerable_alert: str or None
    """

    if aqi <= 50:
        return (
            "✅ **Good air quality.** No health impacts expected. Enjoy outdoor activities.",
            None
        )

    elif aqi <= 100:
        return (
            "🟡 **Moderate air quality.** Most people can continue outdoor activities.",
            "⚠️ **Sensitive individuals** (asthma, elderly) should watch for mild symptoms."
        )

    elif aqi <= 150:
        return (
            "🟠 **Unhealthy for sensitive groups.** Outdoor exertion may cause discomfort.",
            "🚨 **Vulnerable populations** (children, elderly, pregnant women, people with asthma or heart disease) "
            "should limit prolonged outdoor activity."
        )

    elif aqi <= 200:
        return (
            "🔴 **Unhealthy air quality.** Everyone may experience health effects.",
            "🚨 **Vulnerable populations should avoid outdoor activity completely.** "
            "Others should reduce outdoor exertion and wear masks if outside."
        )

    elif aqi <= 300:
        return (
            "🟣 **Very unhealthy air quality. Health alert!** Serious health effects possible.",
            "🚨 **High risk for vulnerable populations.** Stay indoors, use air purifiers, "
            "and seek medical help if symptoms (breathlessness, chest pain) appear."
        )

    else:
        return (
            "⚫ **Hazardous air quality – Emergency conditions.**",
            "🚨 **Everyone, especially vulnerable populations, must stay indoors.** "
            "Avoid all outdoor activity. Follow government emergency advisories."
        )

# def get_health_recommendations(aqi):
#     if aqi <= 50:
#         return "Air quality is satisfactory. Ideal for outdoor activities."
#     elif aqi <= 100:
#         return "Air quality is acceptable. Sensitive individuals should consider limiting prolonged outdoor exertion."
#     elif aqi <= 150:
#         return "Members of sensitive groups may experience health effects. General public less likely to be affected."
#     elif aqi <= 200:
#         return "Everyone may begin to experience health effects. Sensitive groups may experience more serious effects."
#     elif aqi <= 300:
#         return "Health alert: everyone may experience serious health effects. Avoid outdoor activities."
#     else:
#         return "Health warnings of emergency conditions. Everyone should avoid outdoor activities."

def get_government_recommendations(ward_data):
    high_aqi_wards = ward_data[ward_data["AQI"] > 200]
    
    recommendations = []
    
    if len(high_aqi_wards) > 0:
        recommendations.append(f"🚨 **Immediate Action Required**: {len(high_aqi_wards)} ward(s) have AQI > 200")
        recommendations.append("• Implement vehicle restrictions in affected areas")
        recommendations.append("• Halt construction activities temporarily")
        recommendations.append("• Increase water sprinkling on roads")
    
    recommendations.extend([
        "**Short-term Measures:**",
        "• Deploy mobile air quality monitoring units",
        "• Issue public health advisories via SMS/app notifications",
        "• Activate anti-smog guns in high pollution zones",
        "• Increase public transport frequency to reduce private vehicle usage",
        "",
        "**Long-term Policy Recommendations:**",
        "• Expand green cover by 15% in high-pollution wards",
        "• Mandate Euro VI emission standards for all vehicles",
        "• Establish low emission zones in congested areas",
        "• Promote electric vehicle adoption through subsidies",
        "• Implement stricter industrial emission norms",
        "• Create dedicated cycling lanes to encourage non-motorized transport"
    ])
    
    return recommendations

# Sidebar
st.sidebar.title("🌍 Air Quality Dashboard")
view_mode = st.sidebar.radio("Select View", ["Citizen View", "Government View"])
st.sidebar.markdown("---")
st.sidebar.info("**Last Updated:** " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Generate data
ward_data = generate_ward_data()

# Main content
if view_mode == "Citizen View":
    st.title("🏙️ Air Quality Monitoring - Citizen Dashboard")
    st.markdown("Real-time air quality information for your area")
    
    # Ward selector
    selected_ward = st.selectbox("Select Your Ward", ward_data["Ward"].tolist())
    ward_info = ward_data[ward_data["Ward"] == selected_ward].iloc[0]
    
    # Current AQI Display
    col1, col2, col3 = st.columns([2, 2, 3])
    
    with col1:
        st.metric("Current AQI", ward_info["AQI"], delta=None)
        st.markdown(f"<div style='background-color:{ward_info['Color']};padding:10px;border-radius:5px;text-align:center;color:white;font-weight:bold'>{ward_info['Category']}</div>", unsafe_allow_html=True)
    
    with col2:
        st.metric("PM2.5 (μg/m³)", ward_info["PM2.5"])
        st.metric("PM10 (μg/m³)", ward_info["PM10"])
    
    with col3:
        st.markdown("#### Health Recommendations")

        general_msg, vulnerable_msg = get_health_recommendations(ward_info["AQI"])

        st.info(f"### General Population\n{general_msg}")
        st.warning(f"### Vulnerable Population Alert\n{vulnerable_msg}")

        st.caption(
    "👥 *Vulnerable population includes: elderly, children under 5 years, pregnant women, "
    "and individuals with asthma, lung or heart diseases.*"
        )


        # st.info(get_health_recommendations(ward_info["AQI"]))
    
    st.markdown("---")
    
    # AQI Trend
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 24-Hour AQI Trend")
        trend_data = generate_trend_data(selected_ward)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend_data["Time"],
            y=trend_data["AQI"],
            mode='lines+markers',
            line=dict(color='#FF6B6B', width=3),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.2)'
        ))
        
        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title="Time",
            yaxis_title="AQI",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏭 Pollution Sources")
        # source_data = generate_source_data(selected_ward)
        source_data = generate_source_data(ward_info["PM2.5"], ward_info["PM10"], ward_info["NO2"], ward_info["SO2"], ward_info["CO"], ward_info["O3"])
        
        fig = px.pie(
            source_data,
            values="Contribution %",
            names="Source",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Pollutant breakdown
    st.subheader("🔬 Pollutant Levels")
    pollutant_cols = st.columns(5)
    
    pollutants = [
        ("NO2", ward_info["NO2"], "ppb"),
        ("SO2", ward_info["SO2"], "ppb"),
        ("CO", ward_info["CO"], "ppm"),
        ("O3", ward_info["O3"], "ppb"),
    ]
    
    for col, (name, value, unit) in zip(pollutant_cols, pollutants):
        with col:
            st.metric(name, f"{value} {unit}")
    
    # Nearby wards comparison
    st.markdown("---")
    st.subheader("📍 Nearby Wards Comparison")
    
    fig = px.bar(
        ward_data.sort_values("AQI", ascending=False),
        x="Ward",
        y="AQI",
        color="Category",
        color_discrete_map={
            "Good": "#00E400",
            "Moderate": "#FFFF00",
            "Unhealthy for Sensitive": "#FF7E00",
            "Unhealthy": "#FF0000",
            "Very Unhealthy": "#8F3F97",
            "Hazardous": "#7E0023"
        },
        text="AQI"
    )
    
    fig.update_layout(height=400, showlegend=True)
    fig.update_traces(textposition='outside')
    
    st.plotly_chart(fig, use_container_width=True)

else:  # Government View
    st.title("🏛️ Air Quality Monitoring - Government Dashboard")
    st.markdown("Comprehensive pollution monitoring and policy recommendations")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_aqi = ward_data["AQI"].mean()
        st.metric("Average City AQI", f"{avg_aqi:.0f}")
    
    with col2:
        critical_wards = len(ward_data[ward_data["AQI"] > 200])
        st.metric("Critical Wards (AQI > 200)", critical_wards, delta=None, delta_color="inverse")
    
    with col3:
        good_wards = len(ward_data[ward_data["AQI"] <= 100])
        st.metric("Wards with Good/Moderate AQI", good_wards)
    
    with col4:
        worst_ward = ward_data.loc[ward_data["AQI"].idxmax(), "Ward"]
        worst_aqi = ward_data["AQI"].max()
        st.metric("Worst Performing Ward", worst_ward)
        st.caption(f"AQI: {worst_aqi}")
    
    st.markdown("---")
    
    # Geographic overview and trend
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("🗺️ Ward-wise AQI Status")
        
        fig = px.bar(
            ward_data.sort_values("AQI", ascending=False),
            x="Ward",
            y="AQI",
            color="Category",
            color_discrete_map={
                "Good": "#00E400",
                "Moderate": "#FFFF00",
                "Unhealthy for Sensitive": "#FF7E00",
                "Unhealthy": "#FF0000",
                "Very Unhealthy": "#8F3F97",
                "Hazardous": "#7E0023"
            },
            text="AQI",
            hover_data=["PM2.5", "PM10"]
        )
        
        fig.add_hline(y=200, line_dash="dash", line_color="red", 
                     annotation_text="Critical Threshold", annotation_position="right")
        
        fig.update_layout(height=400, showlegend=True)
        fig.update_traces(textposition='outside')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 AQI Distribution")
        
        category_counts = ward_data["Category"].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=category_counts.index,
            values=category_counts.values,
            hole=0.4,
            marker=dict(colors=['#00E400', '#FFFF00', '#FF7E00', '#FF0000', '#8F3F97', '#7E0023'])
        )])
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed ward analysis
    st.markdown("---")
    st.subheader("🔍 Detailed Ward Analysis")
    
    selected_gov_ward = st.selectbox("Select Ward for Detailed Analysis", ward_data["Ward"].tolist(), key="gov_ward")
    ward_info = ward_data[ward_data["Ward"] == selected_gov_ward].iloc[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Pollutant Levels")
        pollutant_df = pd.DataFrame({
            "Pollutant": ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"],
            "Value": [ward_info["PM2.5"], ward_info["PM10"], ward_info["NO2"], 
                     ward_info["SO2"], ward_info["CO"], ward_info["O3"]],
            "Unit": ["μg/m³", "μg/m³", "ppb", "ppb", "ppm", "ppb"]
        })
        
        fig = px.bar(pollutant_df, x="Pollutant", y="Value", text="Value",
                    color="Value", color_continuous_scale="Reds")
        fig.update_layout(height=300, showlegend=False)
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Primary Pollution Sources")
        source_data = generate_source_data(ward_info["PM2.5"], ward_info["PM10"], ward_info["NO2"], ward_info["SO2"], ward_info["CO"], ward_info["O3"])
   
        
        fig = px.bar(source_data, x="Contribution %", y="Source", 
                    orientation='h', text="Contribution %",
                    color="Contribution %", color_continuous_scale="Oranges")
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Policy recommendations
    st.markdown("---")
    st.subheader("📋 Actionable Mitigation & Policy Recommendations")
    
    recommendations = get_government_recommendations(ward_data)
    
    for rec in recommendations:
        if rec.startswith("🚨"):
            st.error(rec)
        elif rec.startswith("**"):
            st.markdown(rec)
        elif rec:
            st.markdown(rec)
    
    # Comparison table
    
    st.markdown("---")
st.subheader("📑 Complete Ward Comparison Table")

display_df = ward_data[["Ward", "AQI", "Category", "PM2.5", "PM10", "NO2", "SO2"]].sort_values("AQI", ascending=False).reset_index(drop=True)

# small visual AQI bar (unicode blocks)
display_df["AQI"] = display_df["AQI"].astype(int)
for c in ["PM2.5", "PM10", "NO2", "SO2"]:
    display_df[c] = pd.to_numeric(display_df[c], errors="coerce").round(1)

# precompute trends once (cached by generate_trend_data TTL)
trend_cache = {}
for w in display_df["Ward"]:
    try:
        trend_cache[w] = generate_trend_data(w)["AQI"].tolist()
    except Exception:
        trend_cache[w] = [None] * 24

def compute_min_max(series):
    vals = [v for v in series if v is not None]
    if not vals:
        return "— / —"
    mn, mx = int(min(vals)), int(max(vals))
    return f"{mn} / {mx}"

display_df["Min / Max"] = display_df["Ward"].apply(lambda w: compute_min_max(trend_cache.get(w, [None] * 24)))

# precompute trends once (cached by generate_trend_data TTL) to avoid N API calls
# trend_cache = {}
# for w in display_df["Ward"]:
#     try:
#         trend_cache[w] = generate_trend_data(w)["AQI"].tolist()
#     except Exception:
#         trend_cache[w] = [None] * 24

# display_df["AQI Trend"] = display_df["Ward"].apply(
#     lambda w: sparkline_from_series(trend_cache.get(w, [None] * 24), length=12)
# )

# def compute_trend_metrics(series):
#     vals = [v for v in series if v is not None]
#     if not vals:
#         return {"delta": "—", "hrs_over_200": "0h", "minmax": "— / —"}
#     delta = vals[-1] - vals[0]
#     arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "▶")
#     hrs_over = sum(1 for v in vals if v > 200)
#     mn, mx = int(min(vals)), int(max(vals))
#     return {"delta": f"{arrow}{abs(int(delta))}", "hrs_over_200": f"{hrs_over}h", "minmax": f"{mn} / {mx}"}

# trend_metrics_cache = {w: compute_trend_metrics(trend_cache.get(w, [None]*24)) for w in display_df["Ward"]}

# display_df["24h Δ"] = display_df["Ward"].apply(lambda w: trend_metrics_cache[w]["delta"])
# display_df["Hours>200"] = display_df["Ward"].apply(lambda w: trend_metrics_cache[w]["hrs_over_200"])
# display_df["Min / Max"] = display_df["Ward"].apply(lambda w: trend_metrics_cache[w]["minmax"])

# helper to map column values to colorscale
def col_colors(df, col, colorscale="OrRd"):
    vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
    mn, mx = vals.min(), vals.max()
    rng = mx - mn if mx - mn > 0 else 1.0
    return [px.colors.sample_colorscale(colorscale, (v - mn) / rng) for v in vals]

pm25_colors = col_colors(display_df, "PM2.5")
pm10_colors = col_colors(display_df, "PM10")
no2_colors = col_colors(display_df, "NO2")
so2_colors = col_colors(display_df, "SO2")

# compute readable text color (black/white) based on background luminance
def contrast_color(color):
    """
    Accepts hex string ('#rrggbb'), 'rgb(r,g,b)' / 'rgba(r,g,b,a)',
    or a list/tuple/ndarray of RGB values (0-1 floats or 0-255 ints).
    Returns '#000000' or '#FFFFFF' for readable text.
    """
    try:
        # list/tuple/ndarray -> build hex
        if isinstance(color, (list, tuple, np.ndarray)):
            r, g, b = [float(x) for x in color[:3]]
            if max(r, g, b) <= 1.0:
                r, g, b = [int(round(x * 255)) for x in (r, g, b)]
            else:
                r, g, b = [int(round(x)) for x in (r, g, b)]
            hex_color = '#{:02x}{:02x}{:02x}'.format(r, g, b)
        elif isinstance(color, str):
            s = color.strip()
            if s.startswith('rgb'):
                nums = re.findall(r'[\d.]+', s)
                r, g, b = [int(float(n)) for n in nums[:3]]
                hex_color = '#{:02x}{:02x}{:02x}'.format(r, g, b)
            elif s.startswith('#'):
                hex_color = s
            else:
                # unknown string – try to use as-is
                hex_color = s
        else:
            hex_color = '#000000'
    except Exception:
        hex_color = '#000000'

    # normalize hex and compute luminance
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    try:
        r, g, b = [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    except Exception:
        # fallback
        return "#000000"
    def lin(c):
        return c/12.92 if c <= 0.03928 else ((c+0.055)/1.055) ** 2.4
    L = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    return "#000000" if L > 0.45 else "#FFFFFF"

ward_bg = ward_data.sort_values("AQI", ascending=False)["Color"].tolist()
ward_text_colors = ["#000000"] * len(display_df)  # white bg -> black text
aqi_text_colors = [contrast_color(c) for c in ward_bg]
category_text_colors = ["#000000"] * len(display_df)
pm25_text_colors = [contrast_color(c) for c in pm25_colors]
pm10_text_colors = [contrast_color(c) for c in pm10_colors]
no2_text_colors  = [contrast_color(c) for c in no2_colors]
so2_text_colors  = [contrast_color(c) for c in so2_colors]
aqi_trend_text = ["#000000"] * len(display_df)

# create the table
fig = go.Figure(data=[go.Table(
    columnwidth=[160, 70, 140, 70, 70, 70, 70, 120],
    header=dict(
        values=[
            "<b>Ward</b>", "<b>AQI</b>", "<b>Category</b>",
            "<b>PM2.5</b>", "<b>PM10</b>", "<b>NO2</b>", "<b>SO2</b>", "<b>Min / Max</b>"
        ],
        fill_color="rgb(30,30,30)",
        font=dict(color="white", size=12),
        align="left"
    ),
    cells=dict(
        values=[
            display_df["Ward"],
            display_df["AQI"],
            display_df["Category"],
            display_df["PM2.5"],
            display_df["PM10"],
            display_df["NO2"],
            display_df["SO2"],
            display_df["Min / Max"]
        ],
        fill_color=[
            ["white"] * len(display_df),
            ward_bg,
            ["white"] * len(display_df),
            pm25_colors,
            pm10_colors,
            no2_colors,
            so2_colors,
            ["white"] * len(display_df)
        ],
        font=dict(color=[
            ward_text_colors,
            aqi_text_colors,
            category_text_colors,
            pm25_text_colors,
            pm10_text_colors,
            no2_text_colors,
            so2_text_colors,
            ["#000000"] * len(display_df)
        ], size=11),
        align="center",
        height=34
    )
)])
fig.update_layout(height=440, margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig, use_container_width=True)

st.caption("Table: darker cell colors indicate higher pollutant levels. AQI cell color shows category (green→hazardous).")

    
    # Export data
# st.markdown("---")
col1, col2 = st.columns([3, 1])
with col2:
    csv = ward_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Report (CSV)",
        data=csv,
        file_name=f"aqi_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.caption("🔄 Data refreshes every 5 minutes | 📱 For emergencies, contact: 1800-XXX-XXXX")