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
import base64
# Page configuration
st.set_page_config(page_title="Air Quality Dashboard", layout="wide", initial_sidebar_state="expanded")

def set_sidebar_background(image_path: str = "sidebarbg.jpg"):
    """Embed image as base64 and apply CSS to Streamlit sidebar."""
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        css = f"""
        <style>
        [data-testid="stSidebar"] {{
            background-image: url("data:image/jpg;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            position: relative;
        }}
        [data-testid="stSidebar"]::before {{
            content: "";
            position: absolute;
            inset: 0;
            background: rgba(0,0,0,0.35);
            z-index: 0;
        }}
        [data-testid="stSidebar"] > div {{ position: relative; z-index: 1; }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Sidebar background not found: {image_path}")

# add this CSS once (only affects elements with .impact-badge)
st.markdown("""
<style>
.impact-badge, .impact-badge * {
  color: #06203a !important;               /* dark blue */
  -webkit-text-fill-color: #06203a !important;
  font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# call it before any sidebar content
set_sidebar_background("sidebarbg.jpg")
def set_page_background(image_path: str = "bg.jpg"):
    """Embed image as base64 and set as Streamlit app background."""
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        css = f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/jpg;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        /* keep main content readable */
        [data-testid="stAppViewContainer"] > .main {{
            background-color: rgba(255,255,255,0.75);
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Background image not found: {image_path}")

# call before rendering the rest of the UI
set_page_background("bg.jpg")

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
def allocate_budget(ward_data, total_budget_crore):
    df = ward_data.copy()

    # Severity score
    df["Severity"] = df["AQI"].apply(lambda x: max(x - 50, 10))
    df["Total Budget (₹ Cr)"] = (
        df["Severity"] / df["Severity"].sum()
    ) * total_budget_crore

    # Priority tier
    def priority(aqi):
        if aqi >= 250:
            return "🚨 Immediate"
        elif aqi >= 200:
            return "🔴 High"
        else:
            return "🟡 Medium"

    df["Priority"] = df["AQI"].apply(priority)

    # Budget split
    df["Transport Control (₹ Cr)"] = (df["Total Budget (₹ Cr)"] * 0.35).round(2)
    df["Dust Control (₹ Cr)"] = (df["Total Budget (₹ Cr)"] * 0.25).round(2)
    df["Green Cover (₹ Cr)"] = (df["Total Budget (₹ Cr)"] * 0.20).round(2)
    df["Monitoring (₹ Cr)"] = (df["Total Budget (₹ Cr)"] * 0.10).round(2)
    df["Awareness (₹ Cr)"] = (df["Total Budget (₹ Cr)"] * 0.10).round(2)

    # Impact estimate
    df["Expected AQI Reduction"] = (
        df["Total Budget (₹ Cr)"] * 0.25
    ).clip(upper=25).round(1)

    return df.round(2)

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
    CPCB-aligned health + action recommendations.
    Returns:
    - general_advice (str)
    - vulnerable_advice (str)
    """

    vulnerable_steps = (
        "**Specific precautions for vulnerable groups:**\n"
        "- **Asthma / respiratory patients:** Keep inhalers handy, avoid outdoor exposure, "
        "and follow prescribed medication strictly.\n"
        "- **Elderly:** Avoid outdoor movement, especially mornings and evenings; rest adequately.\n"
        "- **Pregnant women:** Limit exposure to polluted air, avoid roadside areas, and stay indoors.\n"
        "- **Children:** Avoid outdoor play; keep indoor air clean using purifiers or ventilation.\n"
    )

    if aqi <= 50:
        return (
            "🟢 **Low risk.** Air quality is good.\n\n"
            "- Enjoy outdoor activities\n"
            "- Walk, cycle, or use public transport\n"
            "- Conserve energy and reduce emissions",
            "No special precautions required for vulnerable populations."
        )

    elif aqi <= 100:
        return (
            "🟡 **Satisfactory air quality.** Minor health risk.\n\n"
            "- Prefer public transport or carpooling\n"
            "- Avoid unnecessary vehicle use\n"
            "- Keep indoor spaces well-ventilated",
            "⚠️ Vulnerable populations may experience minor breathing discomfort.\n"
            "- Avoid prolonged or strenuous outdoor activity\n"
            "- Monitor AQI regularly\n\n"
            + vulnerable_steps
        )

    elif aqi <= 200:
        return (
            "🟠 **Moderate air quality.** Health discomfort possible.\n\n"
            "- Reduce outdoor physical exertion\n"
            "- Avoid burning waste or leaves\n"
            "- Improve indoor air using plants or ventilation",
            "🚨 Vulnerable populations should take extra care.\n"
            "- Avoid prolonged outdoor exposure\n"
            "- Use N95/P100 masks if outdoors\n"
            "- Use air purifiers indoors if available\n\n"
            + vulnerable_steps
        )

    elif aqi <= 300:
        return (
            "🔴 **Poor air quality.** Health impacts likely.\n\n"
            "- Avoid outdoor physical activity\n"
            "- Stay indoors as much as possible\n"
            "- Wear N95/P100 masks if stepping outside\n"
            "- Use HEPA air purifiers",
            "🚨 Vulnerable populations at high risk.\n"
            "- Avoid all outdoor activities\n"
            "- Remain indoors and keep activity levels low\n"
            "- Seek medical advice if symptoms worsen\n\n"
            + vulnerable_steps
        )

    elif aqi <= 400:
        return (
            "🟣 **Very poor air quality.** Serious health risk.\n\n"
            "- Avoid outdoor exposure, especially mornings and evenings\n"
            "- Keep doors and windows closed\n"
            "- Use air purifiers continuously",
            "🚨 Severe risk for vulnerable populations.\n"
            "- Stay indoors at all times\n"
            "- Avoid physical exertion completely\n"
            "- Consult a doctor if breathing issues occur\n\n"
            + vulnerable_steps
        )

    else:
        return (
            "⚫ **Severe air quality – Emergency conditions.**\n\n"
            "- Stay indoors with air purification\n"
            "- Avoid all outdoor activities\n"
            "- Follow government emergency advisories",
            "🚨 **Extreme danger for vulnerable populations.**\n"
            "- Remain indoors strictly\n"
            "- Avoid physical activity\n"
            "- Seek immediate medical attention if breathlessness, chest pain, or dizziness occurs\n\n"
            + vulnerable_steps
        )
def render_ward_map(ward_data, selected_ward=None, mode="citizen"):
    df = ward_data.copy()

    fig = px.scatter_mapbox(
        df,
        lat=df["Ward"].map(lambda w: next(x[1] for x in WARDS if x[0] == w)),
        lon=df["Ward"].map(lambda w: next(x[2] for x in WARDS if x[0] == w)),
        color="AQI",
        size="AQI",
        color_continuous_scale="RdYlGn_r",
        zoom=10,
        height=450,
        hover_name="Ward",
        hover_data={
            "AQI": True,
            "Category": True,
            "PM2.5": True,
            "PM10": True
        }
    )



    # Highlight selected ward for citizen
    if mode == "citizen" and selected_ward:
        sel = df[df["Ward"] == selected_ward]
        fig.add_scattermapbox(
            lat=[next(x[1] for x in WARDS if x[0] == selected_ward)],
            lon=[next(x[2] for x in WARDS if x[0] == selected_ward)],
            mode="markers",
            marker=dict(size=25, color="cyan"),
            name="Your Ward"
        )

    fig.update_layout(
        mapbox_style="carto-positron",
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return fig

# def get_health_recommendations(aqi):
#     """
#     Returns:
#     - general_message: str
#     - vulnerable_alert: str or None
#     """

#     if aqi <= 50:
#         return (
#             "✅ **Good air quality.** No health impacts expected. Enjoy outdoor activities.",
#             None
#         )

#     elif aqi <= 100:
#         return (
#             "🟡 **Moderate air quality.** Most people can continue outdoor activities.",
#             "⚠️ **Sensitive individuals** (asthma, elderly) should watch for mild symptoms."
#         )

#     elif aqi <= 150:
#         return (
#             "🟠 **Unhealthy for sensitive groups.** Outdoor exertion may cause discomfort.",
#             "🚨 **Vulnerable populations** (children, elderly, pregnant women, people with asthma or heart disease) "
#             "should limit prolonged outdoor activity."
#         )

#     elif aqi <= 200:
#         return (
#             "🔴 **Unhealthy air quality.** Everyone may experience health effects.",
#             "🚨 **Vulnerable populations should avoid outdoor activity completely.** "
#             "Others should reduce outdoor exertion and wear masks if outside."
#         )

#     elif aqi <= 300:
#         return (
#             "🟣 **Very unhealthy air quality. Health alert!** Serious health effects possible.",
#             "🚨 **High risk for vulnerable populations.** Stay indoors, use air purifiers, "
#             "and seek medical help if symptoms (breathlessness, chest pain) appear."
#         )

#     else:
#         return (
#             "⚫ **Hazardous air quality – Emergency conditions.**",
#             "🚨 **Everyone, especially vulnerable populations, must stay indoors.** "
#             "Avoid all outdoor activity. Follow government emergency advisories."
#         )

def analyze_trend_direction(ward_data):
    """Analyze if AQI is rising or falling for each ward"""
    trends = {}
    for ward in ward_data["Ward"]:
        try:
            trend_data = generate_trend_data(ward)
            recent = trend_data["AQI"].tail(6).mean()  # last 6 hours
            older = trend_data["AQI"].head(6).mean()   # first 6 hours
            change = recent - older
            
            if change > 20:
                trends[ward] = {"direction": "rising", "change": change}
            elif change < -20:
                trends[ward] = {"direction": "falling", "change": change}
            else:
                trends[ward] = {"direction": "stable", "change": change}
        except:
            trends[ward] = {"direction": "stable", "change": 0}
    
    return trends

def get_dominant_pollutant(ward_info):
    """Identify the most problematic pollutant"""
    pollutants = {
        "PM2.5": ward_info["PM2.5"] / 35.4,  # normalized to moderate threshold
        "PM10": ward_info["PM10"] / 154,
        "NO2": ward_info["NO2"] / 100,
        "SO2": ward_info["SO2"] / 75,
        "CO": ward_info["CO"] / 9,
        "O3": ward_info["O3"] / 100
    }
    return max(pollutants, key=pollutants.get)

def get_dominant_source(source_data):
    """Get the primary pollution source"""
    return source_data.loc[source_data["Contribution %"].idxmax(), "Source"]

def generate_dynamic_recommendations(ward_data, selected_ward=None, filters=None):
    """
    Generate dynamic, prioritized recommendations based on real-time data.
    
    Parameters:
    - ward_data: DataFrame with all ward information
    - selected_ward: Optional specific ward for targeted recommendations
    - filters: Dict with keys like 'urgency', 'timeframe', 'cost'
    """
    
    recommendations = {
        "immediate": [],
        "short_term": [],
        "medium_term": [],
        "long_term": [],
        "alerts": []
    }
    
    # Analyze overall situation
    avg_aqi = ward_data["AQI"].mean()
    critical_wards = ward_data[ward_data["AQI"] > 200]
    unhealthy_wards = ward_data[ward_data["AQI"] > 150]
    moderate_wards = ward_data[(ward_data["AQI"] > 100) & (ward_data["AQI"] <= 150)]
    
    trends = analyze_trend_direction(ward_data)
    rising_wards = [w for w, t in trends.items() if t["direction"] == "rising"]
    # Ward-targeted recommendations for selected ward (citizen-focused)
    if selected_ward:
        try:
            ward_row = ward_data[ward_data["Ward"] == selected_ward].iloc[0]
            actions_added = set()
            def add_rec(cat, rec):
                if rec["action"] not in actions_added:
                    recommendations[cat].append(rec)
                    actions_added.add(rec["action"])

            dom_pollutant = get_dominant_pollutant(ward_row)
            src_df = generate_source_data(ward_row["PM2.5"], ward_row["PM10"], ward_row["NO2"], ward_row["SO2"], ward_row["CO"], ward_row["O3"])
            dom_source = get_dominant_source(src_df)

            # Personal / household immediate measures
            if ward_row["AQI"] > 200:
                add_rec("immediate", {
                    "action": "Stay Indoors & Use High-efficiency Filters (HEPA)",
                    "reason": f"Hazardous AQI in {selected_ward} ({ward_row['AQI']}). Protect vulnerable members at home.",
                    "impact": "Critical",
                    "timeframe": "Immediate",
                    "estimated_reduction": "N/A (exposure reduction)"
                })
                add_rec("immediate", {
                    "action": "Wear N95/N99 masks if going outside",
                    "reason": "Very high particulate levels increase risk of respiratory issues",
                    "impact": "High",
                    "timeframe": "Immediate",
                    "estimated_reduction": "N/A"
                })
            elif ward_row["AQI"] > 150:
                add_rec("immediate", {
                    "action": "Limit outdoor exertion; wear masks during necessary travel",
                    "reason": f"Unhealthy AQI in {selected_ward} ({ward_row['AQI']})",
                    "impact": "High",
                    "timeframe": "Immediate",
                    "estimated_reduction": "N/A"
                })
            elif ward_row["AQI"] > 100:
                add_rec("short_term", {
                    "action": "Sensitive individuals should reduce prolonged outdoor activity",
                    "reason": f"Moderate AQI in {selected_ward} ({ward_row['AQI']})",
                    "impact": "Medium",
                    "timeframe": "Today",
                    "estimated_reduction": "N/A"
                })
            else:
                add_rec("medium_term", {
                    "action": "Adopt cleaner cooking/fuel options at household level",
                    "reason": "Proactive action to keep local pollution low",
                    "impact": "Low-Medium",
                    "timeframe": "1-4 weeks",
                    "estimated_reduction": "5-10 AQI points (local)"
                })

            # Pollutant-specific local measures (short-term)
            if dom_pollutant == "PM2.5":
                add_rec("short_term", {
                    "action": f"Intensify road cleaning and dust suppression in {selected_ward}",
                    "reason": f"PM2.5 is dominant pollutant ({ward_row['PM2.5']} μg/m³)",
                    "impact": "Medium",
                    "timeframe": "4-12 hours",
                    "estimated_reduction": "8-15 AQI points"
                })
            elif dom_pollutant == "NO2":
                add_rec("short_term", {
                    "action": f"Encourage staggered work hours and carpooling in {selected_ward}",
                    "reason": "High NO2 indicates vehicular emissions as key contributor",
                    "impact": "Medium",
                    "timeframe": "1-2 days",
                    "estimated_reduction": "5-12 AQI points"
                })

            # Source-specific community measures
            if dom_source == "Vehicular Emissions":
                add_rec("short_term", {
                    "action": f"Promote public transport and carpooling campaigns in {selected_ward}",
                    "reason": "Vehicular traffic is principal source in this ward",
                    "impact": "Medium",
                    "timeframe": "24-72 hours",
                    "estimated_reduction": "8-15 AQI points"
                })
            elif dom_source == "Construction Dust":
                add_rec("short_term", {
                    "action": f"Enforce dust control at construction sites near {selected_ward}",
                    "reason": "Construction dust contributes significantly to PM levels",
                    "impact": "Medium",
                    "timeframe": "4-24 hours",
                    "estimated_reduction": "10-18 AQI points"
                })
        except Exception:
            # keep function robust if selected_ward lookup fails
            pass

    # SMART ALERTS
    if len(critical_wards) > 0:
        recommendations["alerts"].append({
            "message": f"🚨 **CRITICAL ALERT**: {len(critical_wards)} ward(s) have hazardous air quality (AQI > 200)",
            "severity": "critical",
            "wards": critical_wards["Ward"].tolist()
        })
    
    if len(critical_wards) >= 5:
        recommendations["alerts"].append({
            "message": f"⚠️ **CITY-WIDE EMERGENCY**: {len(critical_wards)} wards critically affected. Declare public health emergency.",
            "severity": "emergency",
            "wards": []
        })
    
    if len(rising_wards) >= 3:
        recommendations["alerts"].append({
            "message": f"📈 **TREND ALERT**: AQI rapidly rising in {len(rising_wards)} wards. Immediate intervention needed.",
            "severity": "warning",
            "wards": rising_wards
        })
    
    # Vulnerable zone alerts (simulated - you can add actual school/hospital data)
    high_risk_zones = ["Connaught Place", "Karol Bagh", "Laxmi Nagar"]
    vulnerable_affected = [w for w in critical_wards["Ward"].tolist() if w in high_risk_zones]
    if vulnerable_affected:
        recommendations["alerts"].append({
            "message": f"👶 **VULNERABLE ZONE ALERT**: High-density residential/commercial areas affected: {', '.join(vulnerable_affected)}",
            "severity": "warning",
            "wards": vulnerable_affected
        })
    
    # IMMEDIATE ACTIONS (AQI > 200)
    if len(critical_wards) > 0:
        # Analyze dominant pollutants in critical wards
        critical_pm25 = critical_wards["PM2.5"].mean()
        critical_pm10 = critical_wards["PM10"].mean()
        critical_no2 = critical_wards["NO2"].mean()
        
        recommendations["immediate"].append({
            "action": "Implement Graded Response Action Plan (GRAP) Stage IV",
            "reason": f"{len(critical_wards)} wards in hazardous category",
            "impact": "High",
            "timeframe": "0-2 hours",
            "estimated_reduction": "30-50 AQI points"
        })
        
        if critical_pm25 > 150 or critical_pm10 > 250:
            recommendations["immediate"].append({
                "action": "Ban all construction and demolition activities",
                "reason": "Extremely high particulate matter levels",
                "impact": "High",
                "timeframe": "Immediate",
                "estimated_reduction": "20-30 AQI points"
            })
            
            recommendations["immediate"].append({
                "action": "Deploy water sprinklers and anti-smog guns in all critical wards",
                "reason": f"PM2.5: {critical_pm25:.0f} μg/m³, PM10: {critical_pm10:.0f} μg/m³",
                "impact": "Medium",
                "timeframe": "0-4 hours",
                "estimated_reduction": "15-25 AQI points"
            })
        
        if critical_no2 > 80:
            recommendations["immediate"].append({
                "action": "Implement odd-even vehicle scheme or complete traffic ban in critical zones",
                "reason": f"High NO2 levels ({critical_no2:.0f} ppb) indicate vehicular pollution",
                "impact": "High",
                "timeframe": "0-6 hours",
                "estimated_reduction": "25-40 AQI points"
            })
        
        # Source-specific interventions for critical wards
        for _, ward in critical_wards.iterrows():
            source_data = generate_source_data(
                ward["PM2.5"], ward["PM10"], ward["NO2"], 
                ward["SO2"], ward["CO"], ward["O3"]
            )
            dominant_source = get_dominant_source(source_data)
            
            if dominant_source == "Vehicular Emissions" and ward["NO2"] > 70:
                recommendations["immediate"].append({
                    "action": f"Emergency traffic diversion in {ward['Ward']}",
                    "reason": f"Vehicular emissions account for major pollution source (NO2: {ward['NO2']} ppb)",
                    "impact": "High",
                    "timeframe": "2-4 hours",
                    "estimated_reduction": "20-35 AQI points"
                })
            
            elif dominant_source == "Industrial Activity" and ward["SO2"] > 30:
                recommendations["immediate"].append({
                    "action": f"Temporary shutdown of non-essential industries in {ward['Ward']}",
                    "reason": f"Industrial emissions dominant (SO2: {ward['SO2']} ppb)",
                    "impact": "High",
                    "timeframe": "4-8 hours",
                    "estimated_reduction": "25-40 AQI points"
                })
            
            elif dominant_source == "Construction Dust":
                recommendations["immediate"].append({
                    "action": f"Halt all construction in {ward['Ward']} and nearby areas",
                    "reason": "Construction dust is primary contributor",
                    "impact": "Medium-High",
                    "timeframe": "1-2 hours",
                    "estimated_reduction": "15-30 AQI points"
                })
        
        recommendations["immediate"].append({
            "action": "Issue emergency health advisory via SMS, TV, radio, and mobile apps",
            "reason": "Protect public health during hazardous conditions",
            "impact": "Critical",
            "timeframe": "Immediate",
            "estimated_reduction": "N/A (Health protection)"
        })
        
        recommendations["immediate"].append({
            "action": "Close schools and advise work-from-home in affected wards",
            "reason": "Protect vulnerable populations",
            "impact": "Critical",
            "timeframe": "0-12 hours",
            "estimated_reduction": "N/A (Exposure reduction)"
        })
    
    # SHORT-TERM MEASURES (AQI 150-200)
    if len(unhealthy_wards) > 0:
        avg_unhealthy_aqi = unhealthy_wards["AQI"].mean()
        
        recommendations["short_term"].append({
            "action": "Increase public transport frequency by 30%",
            "reason": f"{len(unhealthy_wards)} wards have unhealthy air quality",
            "impact": "Medium",
            "timeframe": "4-8 hours",
            "estimated_reduction": "10-20 AQI points"
        })
        
        recommendations["short_term"].append({
            "action": "Implement strict parking enforcement and congestion charges in hotspot areas",
            "reason": "Reduce private vehicle usage",
            "impact": "Medium",
            "timeframe": "6-12 hours",
            "estimated_reduction": "8-15 AQI points"
        })
        
        # Analyze dominant pollutants
        for _, ward in unhealthy_wards.iterrows():
            dominant = get_dominant_pollutant(ward)
            
            if dominant == "PM2.5":
                recommendations["short_term"].append({
                    "action": f"Intensify road cleaning and dust suppression in {ward['Ward']}",
                    "reason": f"PM2.5 is dominant pollutant ({ward['PM2.5']} μg/m³)",
                    "impact": "Medium",
                    "timeframe": "4-8 hours",
                    "estimated_reduction": "10-18 AQI points"
                })
            
            elif dominant == "NO2":
                recommendations["short_term"].append({
                    "action": f"Optimize traffic signals for better flow in {ward['Ward']}",
                    "reason": f"High NO2 from vehicular emissions ({ward['NO2']} ppb)",
                    "impact": "Low-Medium",
                    "timeframe": "6-12 hours",
                    "estimated_reduction": "5-12 AQI points"
                })
        
        recommendations["short_term"].append({
            "action": "Conduct industrial emission audits in high-pollution wards",
            "reason": "Identify and penalize violators",
            "impact": "Medium",
            "timeframe": "12-24 hours",
            "estimated_reduction": "12-20 AQI points"
        })
        
        recommendations["short_term"].append({
            "action": "Deploy mobile air quality monitoring units to pollution hotspots",
            "reason": "Better real-time data for targeted interventions",
            "impact": "Low (monitoring)",
            "timeframe": "4-8 hours",
            "estimated_reduction": "N/A (Data collection)"
        })
    
    # MEDIUM-TERM STRATEGIES (AQI 100-150)
    if len(moderate_wards) > 0 or avg_aqi > 100:
        recommendations["medium_term"].append({
            "action": "Enforce PUC (Pollution Under Control) checks at all major intersections",
            "reason": "Prevent deterioration from moderate to unhealthy levels",
            "impact": "Medium",
            "timeframe": "1-3 days",
            "estimated_reduction": "8-15 AQI points"
        })
        
        recommendations["medium_term"].append({
            "action": "Restrict entry of heavy commercial vehicles during peak hours",
            "reason": "Reduce emissions from diesel vehicles",
            "impact": "Medium",
            "timeframe": "1-2 days",
            "estimated_reduction": "10-18 AQI points"
        })
        
        recommendations["medium_term"].append({
            "action": "Launch public awareness campaigns on reducing personal emissions",
            "reason": "Engage citizens in pollution control",
            "impact": "Low-Medium",
            "timeframe": "3-7 days",
            "estimated_reduction": "5-10 AQI points"
        })
        
        # Source-specific medium-term actions
        high_cooking_wards = []
        high_waste_burning = []
        
        for _, ward in ward_data.iterrows():
            source_data = generate_source_data(
                ward["PM2.5"], ward["PM10"], ward["NO2"], 
                ward["SO2"], ward["CO"], ward["O3"]
            )
            
            cooking_pct = source_data[source_data["Source"] == "Residential Cooking"]["Contribution %"].values
            waste_pct = source_data[source_data["Source"] == "Waste Burning"]["Contribution %"].values
            
            if len(cooking_pct) > 0 and cooking_pct[0] > 20:
                high_cooking_wards.append(ward["Ward"])
            if len(waste_pct) > 0 and waste_pct[0] > 15:
                high_waste_burning.append(ward["Ward"])
        
        if high_cooking_wards:
            recommendations["medium_term"].append({
                "action": f"Distribute LPG subsidies and promote electric cooking in: {', '.join(high_cooking_wards[:3])}",
                "reason": "Residential cooking is significant contributor",
                "impact": "Medium",
                "timeframe": "7-14 days",
                "estimated_reduction": "8-12 AQI points"
            })
        
        if high_waste_burning:
            recommendations["medium_term"].append({
                "action": f"Increase waste collection frequency and enforcement patrols in: {', '.join(high_waste_burning[:3])}",
                "reason": "Waste burning contributing to pollution",
                "impact": "Medium",
                "timeframe": "3-7 days",
                "estimated_reduction": "6-10 AQI points"
            })
    
    # LONG-TERM POLICIES (Always applicable)
    recommendations["long_term"].append({
        "action": "Expand metro network and dedicated bus corridors by 25%",
        "reason": "Reduce dependency on private vehicles",
        "impact": "High",
        "timeframe": "6-18 months",
        "estimated_reduction": "30-50 AQI points (sustained)"
    })
    
    recommendations["long_term"].append({
        "action": f"Plant 50,000 trees in wards with AQI > {avg_aqi:.0f}",
        "reason": "Natural air purification and carbon sequestration",
        "impact": "Medium-High",
        "timeframe": "3-12 months",
        "estimated_reduction": "15-25 AQI points (long-term)"
    })
    
    recommendations["long_term"].append({
        "action": "Mandate rooftop solar panels for all new commercial buildings",
        "reason": "Reduce coal-based power generation",
        "impact": "Medium",
        "timeframe": "12-24 months",
        "estimated_reduction": "10-20 AQI points"
    })
    
    recommendations["long_term"].append({
        "action": "Establish low-emission zones in high-traffic areas",
        "reason": "Restrict polluting vehicles permanently",
        "impact": "High",
        "timeframe": "6-12 months",
        "estimated_reduction": "20-35 AQI points"
    })
    
    recommendations["long_term"].append({
        "action": "Provide 50% subsidy on electric vehicle purchases for residents of high-pollution wards",
        "reason": "Accelerate EV adoption",
        "impact": "High",
        "timeframe": "12-36 months",
        "estimated_reduction": "25-40 AQI points (phased)"
    })
    
    # Trend-based recommendations
    if len(rising_wards) > 0:
        recommendations["short_term"].append({
            "action": f"Emergency intervention in rapidly deteriorating wards: {', '.join(rising_wards[:5])}",
            "reason": "AQI showing upward trend",
            "impact": "High",
            "timeframe": "2-6 hours",
            "estimated_reduction": "15-30 AQI points"
        })
    
    # Apply filters if provided
    if filters:
        for category in recommendations:
            if category == "alerts":
                continue
            filtered_recs = []
            for rec in recommendations[category]:
                include = True
                
                if filters.get("urgency"):
                    urgency_map = {"immediate": 4, "short_term": 3, "medium_term": 2, "long_term": 1}
                    if urgency_map.get(category, 0) < filters["urgency"]:
                        include = False
                
                if filters.get("min_impact"):
                    impact_map = {"Low": 1, "Low-Medium": 2, "Medium": 3, "Medium-High": 4, "High": 5, "Critical": 6}
                    rec_impact = impact_map.get(rec.get("impact", "Low"), 1)
                    if rec_impact < filters["min_impact"]:
                        include = False
                
                if include:
                    filtered_recs.append(rec)
            
            recommendations[category] = filtered_recs
    
    return recommendations
def display_recommendations(recommendations, show_filters=True):
    """Display recommendations with interactive filters"""
    # Display alerts first
    if recommendations["alerts"]:
        st.markdown("### 🚨 Critical Alerts")
        for alert in recommendations["alerts"]:
            if alert["severity"] == "critical":
                st.error(f"**{alert['message']}**")
                if alert["wards"]:
                    st.caption(f"Affected wards: {', '.join(alert['wards'])}")
            elif alert["severity"] == "emergency":
                st.error(f"🆘 **{alert['message']}**")
            else:
                st.warning(f"{alert['message']}")
                if alert["wards"]:
                    st.caption(f"Affected areas: {', '.join(alert['wards'])}")
        st.markdown("---")
    
    # Interactive filters
    urgency_filter = "All"
    impact_filter = "All"
    show_impact_prediction = True
    
    if show_filters:
        st.markdown("### 🎛️ Filter Recommendations")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            urgency_filter = st.selectbox(
                "Minimum Urgency",
                ["All", "Long-term", "Medium-term", "Short-term", "Immediate"],
                index=0,
                key="urgency_filter"
            )
        
        with col2:
            impact_filter = st.selectbox(
                "Minimum Impact",
                ["All", "Low", "Low-Medium", "Medium", "Medium-High", "High", "Critical"],
                index=0,
                key="impact_filter"
            )
        
        with col3:
            show_impact_prediction = st.checkbox("Show AQI Reduction Estimates", value=True, key="impact_pred")
        
        st.markdown("---")
  
    # Apply filters to recommendations
    urgency_map = {
        "All": 0,
        "Long-term": 1,
        "Medium-term": 2,
        "Short-term": 3,
        "Immediate": 4
    }
    
    impact_map = {
        "All": 0,
        "Low": 1,
        "Low-Medium": 2,
        "Medium": 3,
        "Medium-High": 4,
        "High": 5,
        "Critical": 6
    }
    
    selected_urgency = urgency_map.get(urgency_filter, 0)
    selected_impact = impact_map.get(impact_filter, 0)
    
    # Filter function (inclusive minimum semantics)
    def should_show_recommendation(rec, category, urgency_threshold, impact_threshold):
        # Map category -> urgency level (same scale as urgency_map but numeric)
        category_urgency = {
            "long_term": 1,
            "medium_term": 2,
            "short_term": 3,
            "immediate": 4
        }
        
        # If user selected a minimum urgency (>0), exclude less urgent categories
        if urgency_threshold > 0 and category_urgency.get(category, 0) < urgency_threshold:
            return False
        
        # If user selected a minimum impact (>0), exclude lower-impact recs
        rec_impact = rec.get("impact", "Low")
        rec_impact_value = impact_map.get(rec_impact, 1)
        if impact_threshold > 0 and rec_impact_value < impact_threshold:
            return False
        
        return True
    
    # Display recommendations by category
    categories = [
        ("immediate", "🚨 Immediate Actions (0-8 hours)", "error"),
        ("short_term", "⚡ Short-term Measures (8 hours - 3 days)", "warning"),
        ("medium_term", "🎯 Medium-term Strategies (3 days - 2 weeks)", "info"),
        ("long_term", "🌱 Long-term Policies (1 month+)", "success")
    ]
    total_filtered_count = 0
    
    for cat_key, cat_title, cat_style in categories:
        # Filter recommendations for this category
        filtered_recs = [
            rec for rec in recommendations[cat_key]
            if should_show_recommendation(rec, cat_key, selected_urgency, selected_impact)
        ]
        
        if filtered_recs:
            st.markdown(f"### {cat_title}")
            
            for i, rec in enumerate(filtered_recs, 1):
                # Create expandable sections for each recommendation
                with st.expander(f"**{i}. {rec['action']}**", expanded=(cat_key == "immediate" and i <= 2)):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Reason:** {rec['reason']}")
                        st.markdown(f"**Implementation Time:** {rec['timeframe']}")
                        
                        if show_impact_prediction and rec.get('estimated_reduction'):
                            if "N/A" not in str(rec['estimated_reduction']):
                                st.markdown(f"**Expected AQI Reduction:** {rec['estimated_reduction']}")
                            else:
                                st.markdown(f"**Expected Impact:** {rec['estimated_reduction']}")
                    
                    
                    with col2:
                        impact = rec.get('impact', 'Medium')
                        if impact in ['High', 'Critical']:
                            st.markdown(f"<div class='impact-badge' style='background-color:#d4edda;padding:10px;border-radius:5px;text-align:center'><b>Impact: {impact}</b></div>", unsafe_allow_html=True)
                        elif impact in ['Medium', 'Medium-High']:
                            st.markdown(f"<div class='impact-badge' style='background-color:#fff3cd;padding:10px;border-radius:5px;text-align:center'><b>Impact: {impact}</b></div>", unsafe_allow_html=True)                        
                        else:
                            st.markdown(f"<div class='impact-badge' style='background-color:#f8d7da;padding:10px;border-radius:5px;text-align:center'><b>Impact: {impact}</b></div>", unsafe_allow_html=True)
            st.markdown("")
        elif cat_key == "immediate" and selected_urgency <= 4:
            # Show message if immediate actions exist but are filtered out
            if recommendations[cat_key] and not filtered_recs:
                st.markdown(f"### {cat_title}")
                st.info(f"ℹ️ {len(recommendations[cat_key])} immediate action(s) available. Adjust filters to view.")

    
    # Show helpful message if filters excluded everything
    if total_filtered_count == 0 and (selected_urgency > 0 or selected_impact > 0):
        st.info("ℹ️ No recommendations match your current filter criteria. Try adjusting the filters above.")
    
    # Show count of filtered recommendations
    if show_filters and total_filtered_count > 0:
        total_available = sum(len(recommendations[cat]) for cat in ["immediate", "short_term", "medium_term", "long_term"])
        st.caption(f"📊 Showing {total_filtered_count} of {total_available} total recommendations")
    """Display recommendations with interactive filters"""
    
    
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

# def get_government_recommendations(ward_data):
#     high_aqi_wards = ward_data[ward_data["AQI"] > 200]
    
#     recommendations = []
    
#     if len(high_aqi_wards) > 0:
#         recommendations.append(f"🚨 **Immediate Action Required**: {len(high_aqi_wards)} ward(s) have AQI > 200")
#         recommendations.append("• Implement vehicle restrictions in affected areas")
#         recommendations.append("• Halt construction activities temporarily")
#         recommendations.append("• Increase water sprinkling on roads")
    
#     recommendations.extend([
#         "**Short-term Measures:**",
#         "• Deploy mobile air quality monitoring units",
#         "• Issue public health advisories via SMS/app notifications",
#         "• Activate anti-smog guns in high pollution zones",
#         "• Increase public transport frequency to reduce private vehicle usage",
#         "",
#         "**Long-term Policy Recommendations:**",
#         "• Expand green cover by 15% in high-pollution wards",
#         "• Mandate Euro VI emission standards for all vehicles",
#         "• Establish low emission zones in congested areas",
#         "• Promote electric vehicle adoption through subsidies",
#         "• Implement stricter industrial emission norms",
#         "• Create dedicated cycling lanes to encourage non-motorized transport"
#     ])
    
#     return recommendations

# Sidebar
st.sidebar.title("🌍 Air Quality Dashboard")
view_mode = st.sidebar.radio("Select View", ["Citizen View", "Government View"])
st.sidebar.markdown("### ⏱️ Live Status")
st.sidebar.info(
    f"""
    **City:** Delhi  
    **Last Updated:** {datetime.now().strftime('%H:%M:%S')}  
    **Data Source:** WAQI + OpenAQ  
    """
)

# ---- AQI Legend (Optional, collapsible) ----
with st.sidebar.expander("🎨 AQI Categories"):
    st.markdown("""
    🟢 **Good:** 0–50  
    🟡 **Moderate:** 51–100  
    🟠 **Unhealthy (Sensitive):** 101–150  
    🔴 **Unhealthy:** 151–200  
    🟣 **Very Unhealthy:** 201–300  
    ⚫ **Hazardous:** 300+  
    """)

# Generate data
ward_data = generate_ward_data()

# Main content
if view_mode == "Citizen View":
    st.title("🏙️ Air Quality Monitoring - Citizen Dashboard")
    st.markdown("Real-time air quality information for your area")
    

    # Ward selector
    selected_ward = st.selectbox("Select Your Ward", ward_data["Ward"].tolist())
    ward_info = ward_data[ward_data["Ward"] == selected_ward].iloc[0]

    map_fig = render_ward_map(
        ward_data,
        selected_ward=selected_ward,
        mode="citizen"
    )
    st.plotly_chart(map_fig, use_container_width=True)
    # Current AQI Display
    col1, col2, col3 = st.columns([2, 2, 3])
    
    with col1:
        st.metric("Current AQI", ward_info["AQI"], delta=None)
        st.markdown(f"<div style='background-color:{ward_info['Color']};padding:10px;border-radius:5px;text-align:center;color:white;font-weight:bold'>{ward_info['Category']}</div>", unsafe_allow_html=True)
    
    with col2:
        st.metric("PM2.5 (μg/m³)", ward_info["PM2.5"])
        st.metric("PM10 (μg/m³)", ward_info["PM10"])


    with col3:
        st.markdown("#### 🔍 AQI Summary")
        st.markdown(
        f"""
        • **Dominant Pollutant:** {get_dominant_pollutant(ward_info)}  
        • **Primary Source:** {get_dominant_source(
            generate_source_data(
                ward_info["PM2.5"], ward_info["PM10"], ward_info["NO2"],
                ward_info["SO2"], ward_info["CO"], ward_info["O3"]
            )
        )}
        """
    )
    
    # with col3:
    #     st.markdown("#### Health Recommendations")

    #     general_msg, vulnerable_msg = get_health_recommendations(ward_info["AQI"])

    #     st.info(f"### General Population\n{general_msg}")
    #     st.warning(f"### Vulnerable Population Alert\n{vulnerable_msg}")

    #     st.caption(
    # "👥 *Vulnerable population includes: elderly, children under 5 years, pregnant women, "
    # "and individuals with asthma, lung or heart diseases.*"
    #     )


        # st.info(get_health_recommendations(ward_info["AQI"]))
    
    # st.markdown("---")
    st.markdown("---")
    st.markdown("## 🩺 Health Recommendations")

    general_msg, vulnerable_msg = get_health_recommendations(ward_info["AQI"])

    # gen_col, vuln_col = st.columns(2)

    #general population
    st.info(
        f"""
        ### 👨‍👩‍👧‍👦 General Population
        {general_msg}
        """
       )

    #vulnerable population
    st.warning(
        f"""
        ### ⚠️ Vulnerable Population
        {vulnerable_msg}
        """
        )

    st.caption(
    "👥 *Vulnerable population includes: elderly, children under 5 years, "
    "pregnant women, and individuals with asthma, lung or heart diseases.*"
    )

    
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

# Ward-specific recommendations for citizens
    st.markdown("---")
    st.subheader("💡 Personalized Recommendations for Your Ward")
    
    # Generate recommendations specific to this ward
    ward_specific_data = ward_data[ward_data["Ward"] == selected_ward]
    citizen_recs = generate_dynamic_recommendations(ward_data, selected_ward=selected_ward)
    
    # Prefer ward-targeted recommendations when available (fall back to city-level otherwise)
    def _ward_filter(recs):
        return [r for r in recs if selected_ward in r.get("reason", "") or selected_ward in r.get("action", "")]

    ward_immediate = _ward_filter(citizen_recs["immediate"])
    ward_short = _ward_filter(citizen_recs["short_term"])
    ward_medium = _ward_filter(citizen_recs["medium_term"])

    # Helper to choose local-first list
    def _choose(local_list, global_list, limit=None):
        if local_list:
            return local_list[:limit] if limit else local_list
        return global_list[:limit] if limit else global_list

    immediate_to_show = _choose(ward_immediate, citizen_recs["immediate"], limit=3)
    if immediate_to_show:
        st.error("🚨 **Immediate Precautions (local-first):**")
        for rec in immediate_to_show:
            st.markdown(f"• **{rec['action']}**")
            st.caption(f"💡 {rec['reason']}")
            if rec.get('estimated_reduction') and "N/A" not in str(rec['estimated_reduction']):
                st.caption(f"📉 Expected improvement: {rec['estimated_reduction']}")
        st.markdown("")
    
    short_to_show = _choose(ward_short, citizen_recs["short_term"], limit=4)
    if short_to_show:
        st.warning("⚡ **Short-term Actions (local-first):**")
        for rec in short_to_show:
            st.markdown(f"• **{rec['action']}**")
            st.caption(f"💡 {rec['reason']}")
        st.markdown("")
 
    medium_to_show = _choose(ward_medium, citizen_recs["medium_term"], limit=3)
    if medium_to_show:
        st.info("🎯 **Medium-term Improvements (local-first):**")
        for rec in medium_to_show:
            st.markdown(f"• **{rec['action']}**")
            st.caption(f"💡 {rec['reason']}")
        st.markdown("")
    # Personal protective measures based on AQI
    st.markdown("#### 🛡️ Personal Protection Measures")
    
    if ward_info["AQI"] > 200:
        st.error("""
        **Immediate Actions for You:**
        - 😷 Wear N95/N99 masks when outdoors
        - 🏠 Stay indoors with windows closed
        - 💨 Use air purifiers if available
        - 🚫 Avoid all outdoor exercise
        - 💊 Keep prescribed medications handy if you have respiratory conditions
        """)
    elif ward_info["AQI"] > 150:
        st.warning("""
        **Recommended Precautions:**
        - 😷 Wear masks when going outside
        - 🏃 Limit outdoor physical activities
        - 🪟 Keep windows closed during peak pollution hours
        - 💨 Use air purifiers indoors
        - 🚌 Use public transport instead of walking/cycling
        """)
    elif ward_info["AQI"] > 100:
        st.info("""
        **Suggested Measures:**
        - 😷 Sensitive individuals should wear masks outdoors
        - 🏃 Reduce prolonged outdoor exertion
        - 🌳 Exercise in parks with more greenery
        - 🚶 Avoid high-traffic areas during peak hours
        """)
    else:
        st.success("""
        **Enjoy the Good Air Quality:**
        - ✅ Safe for all outdoor activities
        - 🏃 Great day for exercise and sports
        - 🌳 Good time to spend time in parks
        """)

else:  # Government View
    st.title("🏛️ Air Quality Monitoring - Government Dashboard")
    st.markdown("## 🗺️ City-wide AQI Heatmap")

    gov_map = render_ward_map(
        ward_data,
        mode="government"
    )
    st.plotly_chart(gov_map, use_container_width=True)

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
            marker = dict(colors=['#8F3F97','#FF0000','#FF7E00','#FFFF00','#00E400'])
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
        st.markdown("---")
    st.subheader("💰 Ward-wise Pollution Mitigation Budget Allocation")

    TOTAL_BUDGET = 500
    budget_df = allocate_budget(ward_data, TOTAL_BUDGET)

    st.caption("Priority-based budget allocation with action-level breakup and estimated impact")
    st.caption(f"**Total Available Budget:** ₹ {TOTAL_BUDGET} Cr (Illustrative)")

    st.dataframe(
        budget_df[
            [
                "Ward", "AQI", "Priority",
                "Total Budget (₹ Cr)",
                "Transport Control (₹ Cr)",
                "Dust Control (₹ Cr)",
                "Green Cover (₹ Cr)",
                "Expected AQI Reduction"
            ]
        ].sort_values("Priority"),
        use_container_width=True
    )
    # Policy recommendations
    st.markdown("---")
    st.subheader("📋 Actionable Mitigation & Policy Recommendations")
    # Generate dynamic recommendations
    recommendations = generate_dynamic_recommendations(ward_data, selected_ward=selected_gov_ward)

    # Display with interactive filters
    # diagnostic: attempt to call and capture NameError / other failures
    display_recommendations(recommendations, show_filters=True)
   
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
