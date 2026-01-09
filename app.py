import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random
import requests
import math

# Page configuration
st.set_page_config(page_title="Air Quality Dashboard", layout="wide", initial_sidebar_state="expanded")

WAQI_TOKEN = "c01d49af5769bc584fc51f7733c6fdcfedf47b3c"   # put your real token here
CITY = "delhi"

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
                "lat": s["station"]["geo"][0],
                "lon": s["station"]["geo"][1],
                "aqi": int(s["aqi"])
            })
    return stations


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
    wards = [
    ("Connaught Place", 28.6315, 77.2167),
    ("Rohini", 28.7400, 77.1200),
    ("Saket", 28.5244, 77.2066),
    ("Laxmi Nagar", 28.6363, 77.2773),
    ("Dwarka", 28.5921, 77.0460),
    ("Karol Bagh", 28.6518, 77.1909)    ,
    ("Janakpuri", 28.6219, 77.0878),
    ("Vasant Kunj", 28.5293, 77.1550),
    ("Pitampura", 28.7033, 77.1310),
    ("Mayur Vihar", 28.6092, 77.2928)
]


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

        data.append({
            "Ward": ward,
            "AQI": aqi,
            "Category": category,
            "Color": color,
            "PM2.5": random.randint(20, 150),
            "PM10": random.randint(30, 200),
            "NO2": random.randint(10, 80),
            "SO2": random.randint(5, 40),
            "CO": random.randint(1, 15),
            "O3": random.randint(20, 100)
        })

    return pd.DataFrame(data)


@st.cache_data(ttl=300)
def generate_trend_data(ward):
    dates = [datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)]
    aqi_values = [random.randint(80, 250) for _ in range(24)]
    
    return pd.DataFrame({
        "Time": dates,
        "AQI": aqi_values
    })

@st.cache_data(ttl=300)
def generate_source_data(ward):
    sources = {
        "Vehicular Emissions": random.randint(25, 45),
        "Industrial Activity": random.randint(15, 35),
        "Construction Dust": random.randint(10, 25),
        "Residential Cooking": random.randint(5, 15),
        "Waste Burning": random.randint(5, 20),
        "Other": random.randint(5, 15)
    }
    
    total = sum(sources.values())
    sources = {k: round(v/total * 100, 1) for k, v in sources.items()}
    
    return pd.DataFrame(list(sources.items()), columns=["Source", "Contribution %"])

def get_health_recommendations(aqi):
    if aqi <= 50:
        return "Air quality is satisfactory. Ideal for outdoor activities."
    elif aqi <= 100:
        return "Air quality is acceptable. Sensitive individuals should consider limiting prolonged outdoor exertion."
    elif aqi <= 150:
        return "Members of sensitive groups may experience health effects. General public less likely to be affected."
    elif aqi <= 200:
        return "Everyone may begin to experience health effects. Sensitive groups may experience more serious effects."
    elif aqi <= 300:
        return "Health alert: everyone may experience serious health effects. Avoid outdoor activities."
    else:
        return "Health warnings of emergency conditions. Everyone should avoid outdoor activities."

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
        st.info(get_health_recommendations(ward_info["AQI"]))
    
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
        source_data = generate_source_data(selected_ward)
        
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
        source_data = generate_source_data(selected_gov_ward)
        
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
    
    display_df = ward_data[["Ward", "AQI", "Category", "PM2.5", "PM10", "NO2", "SO2"]].sort_values("AQI", ascending=False)
    
    def highlight_critical(row):
        if row["AQI"] > 200:
            return ['background-color: #ffcccc'] * len(row)
        elif row["AQI"] > 150:
            return ['background-color: #ffe6cc'] * len(row)
        else:
            return [''] * len(row)
    
    st.dataframe(
        display_df.style.apply(highlight_critical, axis=1),
        use_container_width=True,
        height=400
    )
    
    # Export data
    st.markdown("---")
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
