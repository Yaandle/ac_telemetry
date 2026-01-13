# telemetry_analysis_streamlit.py
import streamlit as st
import pandas as pd
import duckdb
from pathlib import Path
import numpy as np
import altair as alt    # Thin wrapper for charts formatting

st.set_page_config(page_title="Telemetry Analysis", layout="wide")
st.title("Telemetry Analysis Dashboard")

uploaded_file = st.file_uploader("Upload Your CSV File", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV file to proceed.")
    st.stop()

df = pd.read_csv(uploaded_file)
con = duckdb.connect()
con.register("telemetry_data", df)

#======= Session Summary =======#
st.header("Session Summary")
session_summary_query = """
    SELECT 
    COUNT(*) as total_samples,
    MAX(current_lap) as total_laps,
    ROUND(MAX(timestamp) / 60, 2) as duration_min,
    ROUND(MAX(speed_kmh), 2) as top_speed_kmh,
    ROUND(AVG(CASE WHEN speed_kmh > 5 THEN speed_kmh END), 2) as avg_speed_kmh
FROM telemetry_data
"""
summary = con.execute(session_summary_query).fetchdf()
st.dataframe(summary, width="stretch")
st.write("---")

#======= Lap Summary =======#
st.header("Lap-by-Lap Summary")
lap_query = """
SELECT 
    current_lap as lap,
    COUNT(*) as samples,
    ROUND(MAX(speed_kmh), 1) as top_speed,
    ROUND(AVG(speed_kmh), 1) as avg_speed,
    ROUND(AVG(gas) * 100, 1) as avg_throttle,
    ROUND(AVG(brake) * 100, 1) as avg_brake
FROM telemetry_data
GROUP BY current_lap
ORDER BY current_lap
"""
lap_df = con.execute(lap_query).fetchdf()
st.dataframe(lap_df, width="stretch")
st.write("---")

#====== Input Distributions ======#
st.header("Input Distributions")

st.text("🟢 Gas (Throttle) Usage")

throttle_query = """
SELECT 
    CASE 
        WHEN gas > 0.9 THEN 'Full (>90%)'
        WHEN gas > 0.7 THEN 'High (60-90%)'
        WHEN gas > 0.5 THEN 'Medium (30-60%)'
        WHEN gas > 0.2 THEN 'Minimal (0-30%)'
    END AS range,
    COUNT(*) AS samples
FROM telemetry_data
WHERE gas > 0
GROUP BY 1
ORDER BY samples DESC
"""

throttle_df = con.execute(throttle_query).fetchdf()
total_samples = throttle_df["samples"].sum()

lines = []
for _, row in throttle_df.iterrows():
    if row.range is None:
        continue
    pct = (row.samples / total_samples) * 100
    bars = "█" * int(pct / 2)
    lines.append(
        f"{row.range:<18} {row.samples:>7,} ({pct:5.2f}%) {bars}"
    )
st.code("\n".join(lines), language="text")

st.text("🛑 Brake Usage")

brake_query = """
SELECT 
    CASE 
        WHEN brake > 0.9 THEN 'Full (>90%)'
        WHEN brake > 0.7 THEN 'Hard (>70%)'
        WHEN brake > 0.5 THEN 'Medium (60-90%)'
        WHEN brake > 0.0 THEN 'Minimal (<30%)'
    END AS range,
    COUNT(*) AS samples
FROM telemetry_data
WHERE brake > 0
GROUP BY 1
ORDER BY samples DESC
"""

brake_df = con.execute(brake_query).fetchdf()
total_brake = brake_df["samples"].sum()

lines = []
for _, row in brake_df.iterrows():
    if row.range is None:
        continue
    pct = (row.samples / total_brake) * 100
    bars = "█" * int(pct / 2)
    lines.append(
        f"{row.range:<18} {row.samples:>7,} ({pct:5.2f}%) {bars}"
    )
st.code("\n".join(lines), language="text")

st.badge(
    "Brake usage distribution shown excludes all 'Off' (0) inputs. Percentages reflect the share of braking events at each intensity relative to the total number of recorded samples.",
    icon=":material/info:",
    color="blue",
    width="stretch"
)

st.write("---")


#====== Speed Distribution ======#
st.subheader("Speed Distribution")

speed_bins = list(range(80, 301, 10))

df["speed_bin"] = pd.cut(
    df["speed_kmh"],
    bins=speed_bins,
    right=False
)

speed_hist = df.groupby("speed_bin", sort=False)["speed_kmh"].count().reset_index()
speed_hist.columns = ["Speed Range (km/h)", "Samples"]

speed_hist["Speed Range (km/h)"] = speed_hist["Speed Range (km/h)"].apply(
    lambda x: f"{int(x.left)}–{int(x.right)} km/h"
)

speed_hist["sort_key"] = speed_hist["Speed Range (km/h)"].apply(
    lambda x: int(x.split("–")[0])
)
speed_hist = speed_hist.sort_values("sort_key").drop("sort_key", axis=1).reset_index(drop=True)

import altair as alt

speed_chart = alt.Chart(speed_hist).mark_bar().encode(
    x=alt.X("Speed Range (km/h):N", title="Speed Range (km/h)", sort=None),
    y=alt.Y("Samples:Q", title="Samples"),
    tooltip=["Speed Range (km/h)", "Samples"]
).properties(
    width=700,
    height=400,
    title="Speed Distribution"
).configure_axis(
    labelAngle=0
)

st.altair_chart(speed_chart, use_container_width=True)

#====== Wheel Telemetry Analysis ======#
st.header(" Wheel Telemetry Analysis")


st.subheader("Wheel Slip (Average | Max)")

wheel_slip_query = """
SELECT 
    ROUND(AVG(wheel_slip_fl), 4) as avg_fl,
    ROUND(MAX(wheel_slip_fl), 4) as max_fl,
    ROUND(AVG(wheel_slip_fr), 4) as avg_fr,
    ROUND(MAX(wheel_slip_fr), 4) as max_fr,
    ROUND(AVG(wheel_slip_rl), 4) as avg_rl,
    ROUND(MAX(wheel_slip_rl), 4) as max_rl,
    ROUND(AVG(wheel_slip_rr), 4) as avg_rr,
    ROUND(MAX(wheel_slip_rr), 4) as max_rr
FROM telemetry_data
"""

slip_data = con.execute(wheel_slip_query).fetchone()

slip_lines = [
    f"  Front Left:  {slip_data[0]:.4f} | {slip_data[1]:.4f}",
    f"  Front Right: {slip_data[2]:.4f} | {slip_data[3]:.4f}",
    f"  Rear Left:   {slip_data[4]:.4f} | {slip_data[5]:.4f}",
    f"  Rear Right:  {slip_data[6]:.4f} | {slip_data[7]:.4f}"
]

st.code("\n".join(slip_lines), language="text")

st.subheader(" Suspension Travel Analysis")

suspension_query = """
SELECT 
    ROUND(AVG(suspension_travel_fl), 2) as avg_fl,
    ROUND(AVG(suspension_travel_fr), 2) as avg_fr,
    ROUND(AVG(suspension_travel_rl), 2) as avg_rl,
    ROUND(AVG(suspension_travel_rr), 2) as avg_rr,
    ROUND(MAX(suspension_travel_fl), 2) as max_fl,
    ROUND(MAX(suspension_travel_fr), 2) as max_fr,
    ROUND(MAX(suspension_travel_rl), 2) as max_rl,
    ROUND(MAX(suspension_travel_rr), 2) as max_rr,
    ROUND(MIN(suspension_travel_fl), 2) as min_fl,
    ROUND(MIN(suspension_travel_fr), 2) as min_fr,
    ROUND(MIN(suspension_travel_rl), 2) as min_rl,
    ROUND(MIN(suspension_travel_rr), 2) as min_rr
FROM telemetry_data
"""

susp_data = con.execute(suspension_query).fetchone()

susp_lines = [
    "  Average Travel (mm):",
    f"    FL: {susp_data[0]:6.2f}  FR: {susp_data[1]:6.2f}",
    f"    RL: {susp_data[2]:6.2f}  RR: {susp_data[3]:6.2f}",
    "  Max Compression (mm):",
    f"    FL: {susp_data[4]:6.2f}  FR: {susp_data[5]:6.2f}",
    f"    RL: {susp_data[6]:6.2f}  RR: {susp_data[7]:6.2f}",
    "  Max Extension (mm):",
    f"    FL: {susp_data[8]:6.2f}  FR: {susp_data[9]:6.2f}",
    f"    RL: {susp_data[10]:6.2f}  RR: {susp_data[11]:6.2f}"
]

st.code("\n".join(susp_lines), language="text")

st.write("---")

con.close()
