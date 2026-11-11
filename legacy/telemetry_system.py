#!/usr/bin/env python3
"""
ACC Telemetry Analytics Dashboard - Complete Multi-Page Application
Stage 3: System Formation - Comprehensive analysis interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Page configuration
st.set_page_config(
    page_title="ACC Telemetry Analytics",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'df' not in st.session_state:
    st.session_state.df = None
if 'con' not in st.session_state:
    st.session_state.con = None

# Sidebar navigation
st.sidebar.title("🏎️ ACC Telemetry")
st.sidebar.caption("Analytics Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Overview",
        "📈 Descriptive Statistics",
        "� Frequency Analysis",
        "🔗 Relationship Analysis",
        "📉 Time Series & Traces",
        "🎯 Performance Analytics",
        "⚠️ Data Quality & Outliers",
        "💾 SQL Query Interface",
        "📄 Documentation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Data Analytics for Business")
st.sidebar.caption("Stage 3: System Formation")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload CSV Telemetry File", type=["csv"])

if uploaded_file is not None:
    if st.session_state.df is None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.session_state.con = duckdb.connect()
        st.session_state.con.register("telemetry", st.session_state.df)
        st.sidebar.success(f"✅ Loaded {len(st.session_state.df):,} records")
else:
    st.warning("⬆️ Please upload a CSV file to begin analysis")
    st.stop()

df = st.session_state.df
con = st.session_state.con

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================
if page == "📊 Overview":
    st.title("Dashboard Overview")
    st.caption("High-level session summary and KPIs")

    # Session Summary Card
    st.subheader("Session Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Laps", df['current_lap'].max())
    with col2:
        st.metric("Total Samples", f"{len(df):,}")
    with col3:
        st.metric("Duration (min)", f"{df['timestamp'].max()/60:.1f}")
    with col4:
        st.metric("Peak Speed", f"{df['speed_kmh'].max():.1f} km/h")

    # Quick Performance Indicators
    st.subheader("Quick Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        mean_speed = df['speed_kmh'].mean()
        std_speed = df['speed_kmh'].std()
        st.metric("Avg Speed", f"{mean_speed:.1f} km/h", f"±{std_speed:.1f}")

    with col2:
        cv = (std_speed / mean_speed) * 100
        st.metric("Consistency (CV%)", f"{cv:.1f}%")

    with col3:
        avg_throttle = df['gas'].mean() * 100
        st.metric("Avg Throttle", f"{avg_throttle:.1f}%")

    with col4:
        avg_brake = df['brake'].mean() * 100
        st.metric("Avg Brake", f"{avg_brake:.1f}%")

    # Lap Times Overview
    st.subheader("Lap Analysis")
    lap_summary = con.execute("""
        SELECT
            current_lap as lap,
            COUNT(*) as samples,
            ROUND(MAX(speed_kmh), 1) as max_speed,
            ROUND(AVG(speed_kmh), 1) as avg_speed,
            ROUND(AVG(gas) * 100, 1) as avg_throttle,
            ROUND(AVG(brake) * 100, 1) as avg_brake
        FROM telemetry
        GROUP BY current_lap
        ORDER BY current_lap
    """).fetchdf()

    st.dataframe(lap_summary, use_container_width=True)

    # Speed trace
    st.subheader("Speed Profile")
    fig = px.line(df, x='lap_fraction', y='speed_kmh',
                  title='Speed vs Lap Progress',
                  labels={'lap_fraction': 'Lap Fraction', 'speed_kmh': 'Speed (km/h)'})
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 2: DESCRIPTIVE STATISTICS
# ============================================================================
elif page == "📈 Descriptive Statistics":
    st.title("Descriptive Statistics")
    st.caption("Univariate analysis - one variable at a time")

    # Variable selector
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_var = st.selectbox("Select Variable", numerical_cols, index=numerical_cols.index('speed_kmh') if 'speed_kmh' in numerical_cols else 0)

    data = df[selected_var].dropna()

    # Central Location Measures
    st.subheader("Central Location Measures")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean", f"{data.mean():.2f}")
    with col2:
        st.metric("Median", f"{data.median():.2f}")
    with col3:
        mode_val = data.mode()
        st.metric("Mode", f"{mode_val[0]:.2f}" if len(mode_val) > 0 else "N/A")

    # Dispersion Measures
    st.subheader("Dispersion Measures")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Range", f"{data.max() - data.min():.2f}")
    with col2:
        st.metric("Variance", f"{data.var():.2f}")
    with col3:
        st.metric("Std Dev", f"{data.std():.2f}")
    with col4:
        q1, q3 = data.quantile([0.25, 0.75])
        st.metric("IQR", f"{q3 - q1:.2f}")
    with col5:
        cv = (data.std() / data.mean()) * 100 if data.mean() != 0 else 0
        st.metric("CV (%)", f"{cv:.2f}")

    # Standard Error of the Mean
    st.subheader("Estimation Measures")
    col1, col2, col3 = st.columns(3)

    with col1:
        sem = data.sem()
        st.metric("Std Error of Mean (SEM)", f"{sem:.4f}")

    with col2:
        ci_95 = 1.96 * sem
        st.metric("95% CI Margin of Error", f"±{ci_95:.4f}")

    with col3:
        mean_val = data.mean()
        st.metric("95% CI for Mean", f"[{mean_val - ci_95:.2f}, {mean_val + ci_95:.2f}]")

    # Quartiles
    st.subheader("Five-Number Summary & Percentiles")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Min", f"{data.min():.2f}")
    with col2:
        st.metric("Q1 (25%)", f"{data.quantile(0.25):.2f}")
    with col3:
        st.metric("Median (50%)", f"{data.quantile(0.50):.2f}")
    with col4:
        st.metric("Q3 (75%)", f"{data.quantile(0.75):.2f}")
    with col5:
        st.metric("Max", f"{data.max():.2f}")

    # Custom Percentiles
    st.subheader("Custom Percentile Analysis")
    percentiles = st.multiselect("Select Percentiles",
                                options=[1, 5, 10, 25, 50, 75, 90, 95, 99],
                                default=[5, 25, 50, 75, 95])

    percentile_data = []
    for p in sorted(percentiles):
        percentile_data.append({
            'Percentile': f'{p}%',
            'Value': f"{data.quantile(p/100):.2f}"
        })

    percentile_df = pd.DataFrame(percentile_data)
    st.dataframe(percentile_df, use_container_width=True)

    # Shape Measures
    st.subheader("Shape Measures")
    col1, col2 = st.columns(2)
    with col1:
        skew = data.skew()
        st.metric("Skewness", f"{skew:.3f}")
        if abs(skew) < 0.5:
            st.caption("✅ Approximately symmetric")
        elif skew > 0:
            st.caption("📈 Right-skewed (positive)")
        else:
            st.caption("📉 Left-skewed (negative)")

    with col2:
        kurt = data.kurtosis()
        st.metric("Kurtosis", f"{kurt:.3f}")
        if abs(kurt) < 0.5:
            st.caption("✅ Mesokurtic (normal-like)")
        elif kurt > 0:
            st.caption("📊 Leptokurtic (heavy tails)")
        else:
            st.caption("📉 Platykurtic (light tails)")

    # Distribution Visualization
    st.subheader("Distribution Visualization")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(data, x=selected_var, nbins=50,
                          title=f"Histogram of {selected_var}",
                          labels={selected_var: selected_var})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(data, y=selected_var,
                    title=f"Boxplot of {selected_var}",
                    labels={selected_var: selected_var})
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 3: FREQUENCY ANALYSIS
# ============================================================================
elif page == "📊 Frequency Analysis":
    st.title("Frequency Analysis")
    st.caption("Distribution binning, frequency tables, and cumulative analysis")

    # Variable selector
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_var = st.selectbox("Select Variable for Frequency Analysis", numerical_cols,
                               index=numerical_cols.index('speed_kmh') if 'speed_kmh' in numerical_cols else 0)

    data = df[selected_var].dropna()

    # Binning configuration
    col1, col2 = st.columns(2)
    with col1:
        n_bins = st.slider("Number of Bins", 5, 100, 20)
    with col2:
        bin_method = st.selectbox("Binning Method", ["Equal Width", "Equal Frequency (Quantile)"])

    if bin_method == "Equal Width":
        counts, bin_edges = np.histogram(data, bins=n_bins)
    else:  # Equal Frequency
        counts, bin_edges = np.histogram(data, bins=pd.qcut(data, q=n_bins, duplicates='drop'))

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    relative_freq = (counts / counts.sum()) * 100
    cumulative_freq = np.cumsum(counts)
    cumulative_pct = (cumulative_freq / cumulative_freq[-1]) * 100

    # Frequency Table
    st.subheader("Frequency Distribution Table")
    freq_table = pd.DataFrame({
        'Bin Range': [f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)],
        'Frequency': counts,
        'Relative Freq (%)': relative_freq,
        'Cumulative Freq': cumulative_freq,
        'Cumulative % (CDF)': cumulative_pct
    })

    st.dataframe(freq_table, use_container_width=True)

    # Histogram with relative frequency
    st.subheader("Frequency Distribution Visualization")
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bin_centers, y=counts, name='Frequency'))
        fig.update_layout(title=f'Frequency Histogram: {selected_var}',
                         xaxis_title='Value',
                         yaxis_title='Frequency (Count)')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bin_centers, y=relative_freq, name='Relative Frequency'))
        fig.update_layout(title=f'Relative Frequency: {selected_var}',
                         xaxis_title='Value',
                         yaxis_title='Relative Frequency (%)')
        st.plotly_chart(fig, use_container_width=True)

    # Cumulative Distribution Function (CDF)
    st.subheader("Cumulative Distribution Function (CDF)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bin_centers, y=cumulative_pct, mode='lines+markers',
                            name='CDF', line=dict(color='blue', width=2)))
    fig.update_layout(title=f'Cumulative Distribution: {selected_var}',
                     xaxis_title='Value',
                     yaxis_title='Cumulative %',
                     yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig, use_container_width=True)

    # Frequency statistics
    st.subheader("Frequency Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        modal_class = bin_centers[np.argmax(counts)]
        st.metric("Modal Class Center", f"{modal_class:.2f}")

    with col2:
        st.metric("Total Observations", f"{len(data):,}")

    with col3:
        st.metric("Min Bin Count", f"{counts.min()}")

    with col4:
        st.metric("Max Bin Count", f"{counts.max()}")

    # Percentile finder
    st.subheader("Percentile Queries")
    percentile = st.slider("Find this Percentile", 1, 99, 50)
    percentile_value = np.percentile(data, percentile)
    st.metric(f"{percentile}th Percentile Value", f"{percentile_value:.2f}")

    # Speed/RPM/etc zones (categorical binning example)
    st.subheader("Value Zones Breakdown")
    if selected_var == 'speed_kmh':
        zones = {'0-50 km/h': (data < 50).sum(),
                '50-100 km/h': ((data >= 50) & (data < 100)).sum(),
                '100-150 km/h': ((data >= 100) & (data < 150)).sum(),
                '150+ km/h': (data >= 150).sum()}
        zone_df = pd.DataFrame(list(zones.items()), columns=['Zone', 'Count'])
        zone_df['Percentage'] = (zone_df['Count'] / zone_df['Count'].sum() * 100).round(2)

        fig = px.pie(zone_df, values='Count', names='Zone',
                    title=f'{selected_var} Zone Distribution')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(zone_df, use_container_width=True)

# ============================================================================
# PAGE 4: RELATIONSHIP ANALYSIS
# ============================================================================
elif page == "🔗 Relationship Analysis":
    st.title("Relationship Analysis")
    st.caption("Two-variable relationships and correlations")

    # Variable pair selector
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("X-axis Variable", numerical_cols, index=numerical_cols.index('speed_kmh') if 'speed_kmh' in numerical_cols else 0)
    with col2:
        y_var = st.selectbox("Y-axis Variable", numerical_cols, index=numerical_cols.index('accel_lateral') if 'accel_lateral' in numerical_cols else 1)

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    core_vars = ['speed_kmh', 'gas', 'brake', 'steer', 'rpm', 'accel_lateral', 'accel_longitudinal']
    available_vars = [v for v in core_vars if v in df.columns]

    if len(available_vars) > 0:
        corr_matrix = df[available_vars].corr()
        fig = px.imshow(corr_matrix,
                       text_auto='.2f',
                       color_continuous_scale='RdBu_r',
                       title='Correlation Matrix',
                       zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)

    # Scatterplot
    st.subheader(f"Scatterplot: {x_var} vs {y_var}")
    sample_size = min(10000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)

    fig = px.scatter(df_sample, x=x_var, y=y_var,
                    opacity=0.5,
                    trendline="ols",
                    title=f"{x_var} vs {y_var}")
    st.plotly_chart(fig, use_container_width=True)

    # Correlation coefficient
    correlation = df[x_var].corr(df[y_var])
    st.metric("Correlation Coefficient (r)", f"{correlation:.3f}")

    # Common relationships
    st.subheader("Pre-configured Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(["Speed vs G-Force", "Throttle vs Slip", "RPM vs Speed", "Lap Progress"])

    with tab1:
        if 'speed_kmh' in df.columns and 'accel_lateral' in df.columns:
            fig = px.scatter(df_sample, x='speed_kmh', y='accel_lateral',
                           title='Cornering Envelope: Speed vs Lateral G',
                           labels={'speed_kmh': 'Speed (km/h)', 'accel_lateral': 'Lateral Accel (m/s²)'})
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if 'gas' in df.columns and 'wheel_slip_fl' in df.columns:
            fig = px.scatter(df_sample, x='gas', y='wheel_slip_fl',
                           title='Traction Limit: Throttle vs Wheel Slip',
                           labels={'gas': 'Throttle', 'wheel_slip_fl': 'Front Left Slip'})
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if 'rpm' in df.columns and 'speed_kmh' in df.columns:
            fig = px.scatter(df_sample, x='rpm', y='speed_kmh',
                           color='gear' if 'gear' in df.columns else None,
                           title='Gear Analysis: RPM vs Speed',
                           labels={'rpm': 'RPM', 'speed_kmh': 'Speed (km/h)'})
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        if 'lap_fraction' in df.columns and 'speed_kmh' in df.columns:
            fig = px.line(df_sample.sort_values('lap_fraction'), x='lap_fraction', y='speed_kmh',
                         title='Speed Trace',
                         labels={'lap_fraction': 'Lap Progress', 'speed_kmh': 'Speed (km/h)'})
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 4: TIME SERIES & TRACES
# ============================================================================
elif page == "📉 Time Series & Traces":
    st.title("Time Series & Traces")
    st.caption("Sequential data and lap-by-lap evolution")

    # Multi-variable trace
    st.subheader("Multi-Variable Trace")

    variables = st.multiselect(
        "Select variables to plot",
        ['speed_kmh', 'gas', 'brake', 'steer', 'rpm'],
        default=['speed_kmh', 'gas', 'brake']
    )

    if len(variables) > 0:
        # Normalize data for comparison
        df_plot = df[['lap_fraction'] + variables].copy()
        for var in variables:
            df_plot[f'{var}_norm'] = (df_plot[var] - df_plot[var].min()) / (df_plot[var].max() - df_plot[var].min())

        fig = go.Figure()
        for var in variables:
            fig.add_trace(go.Scatter(x=df_plot['lap_fraction'], y=df_plot[f'{var}_norm'],
                                    mode='lines', name=var))

        fig.update_layout(title='Normalized Multi-Variable Trace',
                         xaxis_title='Lap Fraction',
                         yaxis_title='Normalized Value [0-1]')
        st.plotly_chart(fig, use_container_width=True)

    # Lap comparison
    st.subheader("Lap-by-Lap Comparison")
    if 'current_lap' in df.columns:
        laps = sorted(df['current_lap'].unique())
        selected_laps = st.multiselect("Select laps to compare", laps, default=laps[:min(3, len(laps))])

        if len(selected_laps) > 0:
            fig = go.Figure()
            for lap in selected_laps:
                lap_data = df[df['current_lap'] == lap].sort_values('lap_fraction')
                fig.add_trace(go.Scatter(x=lap_data['lap_fraction'], y=lap_data['speed_kmh'],
                                        mode='lines', name=f'Lap {lap}'))

            fig.update_layout(title='Speed Comparison by Lap',
                             xaxis_title='Lap Fraction',
                             yaxis_title='Speed (km/h)')
            st.plotly_chart(fig, use_container_width=True)

    # Wheel dynamics
    st.subheader("Wheel Dynamics")
    wheel_cols = [col for col in df.columns if 'wheel_slip' in col]

    if len(wheel_cols) > 0:
        df_sample = df.sample(n=min(5000, len(df)), random_state=42).sort_values('lap_fraction')

        fig = go.Figure()
        for col in wheel_cols:
            fig.add_trace(go.Scatter(x=df_sample['lap_fraction'], y=df_sample[col],
                                    mode='lines', name=col.replace('wheel_slip_', '').upper()))

        fig.update_layout(title='Wheel Slip by Corner',
                         xaxis_title='Lap Fraction',
                         yaxis_title='Slip Ratio')
        st.plotly_chart(fig, use_container_width=True)

    # Suspension travel
    st.subheader("Suspension Travel")
    susp_cols = [col for col in df.columns if 'suspension_travel' in col]

    if len(susp_cols) > 0:
        df_sample = df.sample(n=min(5000, len(df)), random_state=42).sort_values('lap_fraction')

        fig = go.Figure()
        for col in susp_cols:
            fig.add_trace(go.Scatter(x=df_sample['lap_fraction'], y=df_sample[col],
                                    mode='lines', name=col.replace('suspension_travel_', '').upper()))

        fig.update_layout(title='Suspension Travel (Load Transfer)',
                         xaxis_title='Lap Fraction',
                         yaxis_title='Travel (m)')
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 5: PERFORMANCE ANALYTICS
# ============================================================================
elif page == "🎯 Performance Analytics":
    st.title("Performance Analytics")
    st.caption("KPIs, benchmarks, and improvement tracking")

    # Consistency metrics
    st.subheader("Consistency Metrics")
    col1, col2, col3 = st.columns(3)

    with col1:
        speed_cv = (df['speed_kmh'].std() / df['speed_kmh'].mean()) * 100
        st.metric("Speed Consistency (CV%)", f"{speed_cv:.2f}%")
        st.progress(min(100, int(100 - speed_cv)) / 100)

    with col2:
        throttle_std = df['gas'].std()
        st.metric("Throttle Smoothness", f"{1 - throttle_std:.3f}")
        st.progress(1 - throttle_std)

    with col3:
        brake_std = df['brake'].std()
        st.metric("Brake Smoothness", f"{1 - brake_std:.3f}")
        st.progress(1 - brake_std)

    # Limit analysis
    st.subheader("Limit Analysis")
    col1, col2, col3 = st.columns(3)

    with col1:
        if 'accel_lateral' in df.columns:
            peak_lateral = abs(df['accel_lateral']).max()
            st.metric("Peak Lateral G", f"{peak_lateral:.2f} m/s²")

    with col2:
        peak_speed = df['speed_kmh'].max()
        st.metric("Top Speed", f"{peak_speed:.1f} km/h")

    with col3:
        if 'wheel_slip_fl' in df.columns:
            slip_events = (abs(df['wheel_slip_fl']) > 0.1).sum()
            st.metric("Wheel Slip Events", f"{slip_events:,}")

    # Efficiency scores
    st.subheader("Efficiency Scores")

    # Throttle-to-brake ratio
    total_throttle = df['gas'].sum()
    total_brake = df['brake'].sum()
    ratio = total_throttle / total_brake if total_brake > 0 else 0

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Throttle-to-Brake Ratio", f"{ratio:.2f}")

    with col2:
        full_throttle_pct = (df['gas'] > 0.9).sum() / len(df) * 100
        st.metric("% Time Full Throttle", f"{full_throttle_pct:.1f}%")

    # Delta Analysis (Lap vs Lap Comparison)
    st.subheader("Delta Analysis: Lap vs Lap Comparison")
    if 'current_lap' in df.columns:
        lap_summary = df.groupby('current_lap').agg({
            'speed_kmh': ['mean', 'max', 'std'],
            'gas': 'mean',
            'brake': 'mean',
            'timestamp': 'count'
        }).round(2)

        lap_summary.columns = ['Avg Speed', 'Max Speed', 'Speed SD', 'Avg Throttle', 'Avg Brake', 'Samples']
        lap_summary = lap_summary.reset_index()
        lap_summary.columns = ['Lap', 'Avg Speed', 'Max Speed', 'Speed SD', 'Avg Throttle', 'Avg Brake', 'Samples']

        # Calculate delta from best lap
        best_lap_idx = lap_summary['Avg Speed'].idxmax()
        best_lap = lap_summary.loc[best_lap_idx, 'Lap']
        best_avg_speed = lap_summary.loc[best_lap_idx, 'Avg Speed']

        lap_summary['Δ from Best'] = (lap_summary['Avg Speed'] - best_avg_speed).round(2)
        lap_summary['Δ %'] = ((lap_summary['Avg Speed'] - best_avg_speed) / best_avg_speed * 100).round(2)

        st.dataframe(lap_summary, use_container_width=True)

        # Visualization of lap progression
        col1, col2 = st.columns(2)

        with col1:
            fig = px.line(lap_summary, x='Lap', y=['Avg Speed', 'Max Speed'],
                         title='Speed Evolution by Lap',
                         labels={'Lap': 'Lap Number'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.line(lap_summary, x='Lap', y='Δ from Best',
                         title='Time Delta from Best Lap',
                         labels={'Lap': 'Lap Number', 'Δ from Best': 'Speed Δ (km/h)'})
            st.plotly_chart(fig, use_container_width=True)

    # Diagnostic Analytics - Input Analysis
    st.subheader("Diagnostic Analytics: Input Contribution")

    if 'current_lap' in df.columns:
        best_lap_data = df[df['current_lap'] == best_lap] if 'best_lap' in locals() else df[df['current_lap'] == df['current_lap'].max()]
        avg_lap_data = df[df['current_lap'] != best_lap] if 'best_lap' in locals() else df[df['current_lap'] != df['current_lap'].max()]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Best Lap Avg Speed", f"{best_lap_data['speed_kmh'].mean():.1f} km/h")
            st.metric("Avg Throttle", f"{best_lap_data['gas'].mean()*100:.1f}%")

        with col2:
            st.metric("Other Laps Avg Speed", f"{avg_lap_data['speed_kmh'].mean():.1f} km/h")
            st.metric("Avg Throttle", f"{avg_lap_data['gas'].mean()*100:.1f}%")

        with col3:
            speed_diff = best_lap_data['speed_kmh'].mean() - avg_lap_data['speed_kmh'].mean()
            throttle_diff = (best_lap_data['gas'].mean() - avg_lap_data['gas'].mean()) * 100
            st.metric("Speed Difference", f"{speed_diff:+.1f} km/h")
            st.metric("Throttle Difference", f"{throttle_diff:+.1f}%")

        # Input contribution analysis
        st.subheader("Input Behavior Comparison")

        comparison_vars = [v for v in ['gas', 'brake', 'steer', 'rpm'] if v in df.columns]
        comparison_data = []

        for var in comparison_vars:
            best_mean = best_lap_data[var].mean()
            avg_mean = avg_lap_data[var].mean()
            best_std = best_lap_data[var].std()
            avg_std = avg_lap_data[var].std()

            comparison_data.append({
                'Input': var.upper(),
                'Best Lap Mean': f"{best_mean:.3f}",
                'Best Lap SD': f"{best_std:.3f}",
                'Avg Lap Mean': f"{avg_mean:.3f}",
                'Avg Lap SD': f"{avg_std:.3f}",
                'Δ Mean': f"{best_mean - avg_mean:+.3f}",
                'Smoother?': '✅' if best_std < avg_std else '⚠️'
            })

        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)

    # Progression tracking
    st.subheader("Session Progression")
    if 'current_lap' in df.columns:
        lap_speeds = df.groupby('current_lap')['speed_kmh'].mean().reset_index()

        fig = px.line(lap_speeds, x='current_lap', y='speed_kmh',
                     title='Average Speed by Lap',
                     labels={'current_lap': 'Lap', 'speed_kmh': 'Avg Speed (km/h)'})
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 7: DATA QUALITY & OUTLIER DETECTION
# ============================================================================
elif page == "⚠️ Data Quality & Outliers":
    st.title("Data Quality & Outlier Detection")
    st.caption("Comprehensive data validation and anomaly identification")

    # Data Quality Report
    st.subheader("Data Quality Summary")
    col1, col2, col3, col4, col5 = st.columns(5)

    total_cells = df.shape[0] * df.shape[1]
    total_nulls = df.isnull().sum().sum()
    completeness = ((total_cells - total_nulls) / total_cells) * 100

    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Total Fields", len(df.columns))
    with col3:
        st.metric("Missing Values", f"{total_nulls:,}")
    with col4:
        st.metric("Completeness", f"{completeness:.2f}%")
    with col5:
        st.metric("Data Shape", f"{len(df):,} × {len(df.columns)}")

    # Missing Data Analysis
    st.subheader("Missing Data by Field")
    missing_data = pd.DataFrame({
        'Field': df.columns,
        'Missing Count': df.isnull().sum(),
        'Missing %': (df.isnull().sum() / len(df) * 100).round(2),
        'Data Type': [str(df[col].dtype) for col in df.columns]
    }).sort_values('Missing Count', ascending=False)

    missing_data = missing_data[missing_data['Missing Count'] > 0]
    if len(missing_data) > 0:
        st.dataframe(missing_data, use_container_width=True)

        fig = px.bar(missing_data, x='Field', y='Missing %',
                    title='Missing Data by Field (%)',
                    labels={'Missing %': 'Missing %', 'Field': 'Field'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ No missing data detected!")

    # Outlier Detection - User selection
    st.subheader("Outlier Detection")

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_var = st.selectbox("Select Variable for Outlier Analysis", numerical_cols,
                               index=numerical_cols.index('speed_kmh') if 'speed_kmh' in numerical_cols else 0)

    data = df[selected_var].dropna()

    # Z-score method
    st.subheader("Z-Score Method")
    z_scores = np.abs(stats.zscore(data))
    z_threshold = st.slider("Z-Score Threshold", 1.0, 4.0, 3.0, 0.1)
    z_outliers = (z_scores > z_threshold).sum()
    z_outlier_pct = (z_outliers / len(data)) * 100

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Z-Score Threshold", f"{z_threshold:.1f}σ")
    with col2:
        st.metric("Outliers Detected", f"{z_outliers:,}")
    with col3:
        st.metric("Outlier %", f"{z_outlier_pct:.2f}%")

    # IQR Method
    st.subheader("Interquartile Range (IQR) Method")
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    iqr_multiplier = st.slider("IQR Multiplier", 1.0, 3.0, 1.5, 0.1)

    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR

    iqr_outliers = ((data < lower_bound) | (data > upper_bound)).sum()
    iqr_outlier_pct = (iqr_outliers / len(data)) * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Q1 (25%)", f"{Q1:.2f}")
    with col2:
        st.metric("Q3 (75%)", f"{Q3:.2f}")
    with col3:
        st.metric("IQR", f"{IQR:.2f}")
    with col4:
        st.metric("Lower Bound", f"{lower_bound:.2f}")
    with col5:
        st.metric("Upper Bound", f"{upper_bound:.2f}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("IQR Multiplier", f"{iqr_multiplier:.1f}×")
    with col2:
        st.metric("Outliers Detected", f"{iqr_outliers:,}")
    with col3:
        st.metric("Outlier %", f"{iqr_outlier_pct:.2f}%")

    # Outlier Visualization
    st.subheader("Outlier Visualization")
    col1, col2 = st.columns(2)

    with col1:
        # Boxplot with outliers highlighted
        fig = px.box(df, y=selected_var,
                    title=f'Boxplot: {selected_var} (IQR Method)',
                    labels={selected_var: selected_var})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Distribution with outliers marked
        outlier_mask_z = z_scores > z_threshold
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=data[~outlier_mask_z], name='Normal', opacity=0.7, nbinsx=50))
        fig.add_trace(go.Histogram(x=data[outlier_mask_z], name='Outliers (Z-Score)', opacity=0.7, nbinsx=50))
        fig.update_layout(title=f'Distribution with Z-Score Outliers: {selected_var}',
                         xaxis_title=selected_var,
                         yaxis_title='Frequency',
                         barmode='overlay')
        st.plotly_chart(fig, use_container_width=True)

    # Outlier Details Table
    st.subheader("Outlier Records (Z-Score Method)")
    outlier_indices = np.where(z_scores > z_threshold)[0]

    if len(outlier_indices) > 0:
        outlier_records = df.iloc[outlier_indices].copy()
        outlier_records['z_score'] = z_scores[outlier_indices]
        outlier_records = outlier_records.sort_values('z_score', ascending=False).head(50)

        st.dataframe(outlier_records, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            csv = outlier_records.to_csv(index=False)
            st.download_button("Download Outliers CSV", csv, "outliers.csv", "text/csv")
    else:
        st.info("No outliers detected with current threshold")

    # Data Validation Rules
    st.subheader("Data Validation Rules")

    validation_issues = []

    # Check for zero-filled channels
    for col in numerical_cols:
        zero_count = (df[col] == 0).sum()
        zero_pct = (zero_count / len(df)) * 100
        if zero_pct > 50:  # More than 50% zeros
            validation_issues.append({
                'Field': col,
                'Issue': 'Possible zero-filled channel',
                'Details': f'{zero_count:,} zeros ({zero_pct:.1f}%)',
                'Severity': '⚠️ Warning'
            })

    # Check for negative values where unexpected
    if 'speed_kmh' in df.columns and (df['speed_kmh'] < 0).any():
        validation_issues.append({
            'Field': 'speed_kmh',
            'Issue': 'Negative speed values',
            'Details': f'{(df["speed_kmh"] < 0).sum()} negative values',
            'Severity': '🔴 Error'
        })

    # Check for unrealistic values
    if 'speed_kmh' in df.columns and (df['speed_kmh'] > 300).any():
        validation_issues.append({
            'Field': 'speed_kmh',
            'Issue': 'Extreme speed values',
            'Details': f'{(df["speed_kmh"] > 300).sum()} values > 300 km/h',
            'Severity': '⚠️ Warning'
        })

    if len(validation_issues) > 0:
        validation_df = pd.DataFrame(validation_issues)
        st.dataframe(validation_df, use_container_width=True)
    else:
        st.success("✅ All data validation checks passed!")

# ============================================================================

elif page == "💾 SQL Query Interface":
    st.title("SQL Query Interface")
    st.caption("Terminal-style SQL output for data exploration")

    # Quick queries sidebar
    st.sidebar.subheader("Quick Queries")
    quick_queries = {
        "All Data": "SELECT * FROM telemetry LIMIT 100",
        "Avg Speed by Lap": "SELECT current_lap, AVG(speed_kmh) as avg_speed FROM telemetry GROUP BY current_lap",
        "Gear Distribution": "SELECT gear, COUNT(*) as count FROM telemetry GROUP BY gear ORDER BY gear",
        "Top Speeds": "SELECT current_lap, MAX(speed_kmh) as max_speed FROM telemetry GROUP BY current_lap ORDER BY max_speed DESC",
        "Throttle Analysis": "SELECT AVG(gas) as avg_throttle, MAX(gas) as max_throttle, MIN(gas) as min_throttle FROM telemetry"
    }

    selected_quick = st.sidebar.selectbox("Template Queries", list(quick_queries.keys()))
    if st.sidebar.button("Load Template"):
        st.session_state.query = quick_queries[selected_quick]

    # Query input
    query = st.text_area("SQL Query",
                        value=st.session_state.get('query', "SELECT * FROM telemetry LIMIT 100"),
                        height=150)

    if st.button("Execute Query"):
        try:
            result = con.execute(query).fetchdf()
            st.success(f"Query returned {len(result)} rows")
            st.dataframe(result, use_container_width=True)

            # Export options
            col1, col2 = st.columns(2)
            with col1:
                csv = result.to_csv(index=False)
                st.download_button("Download CSV", csv, "query_result.csv", "text/csv")
            with col2:
                json = result.to_json(orient='records')
                st.download_button("Download JSON", json, "query_result.json", "application/json")

        except Exception as e:
            st.error(f"Query error: {str(e)}")

# ============================================================================
# PAGE 7: DOCUMENTATION
# ============================================================================
elif page == "📄 Documentation":
    st.title("Data Documentation")
    st.caption("Schema reference and metadata")

    # Feature schema
    st.subheader("Feature Schema")

    schema_data = {
        'Field': list(df.columns),
        'Type': [str(df[col].dtype) for col in df.columns],
        'Non-Null': [df[col].notna().sum() for col in df.columns],
        'Null': [df[col].isna().sum() for col in df.columns],
        'Mean': [df[col].mean() if df[col].dtype in ['float64', 'int64'] else None for col in df.columns],
        'Std': [df[col].std() if df[col].dtype in ['float64', 'int64'] else None for col in df.columns]
    }

    schema_df = pd.DataFrame(schema_data)
    st.dataframe(schema_df, use_container_width=True)

    # Data quality report
    st.subheader("Data Quality Report")
    col1, col2, col3 = st.columns(3)

    with col1:
        total_nulls = df.isnull().sum().sum()
        st.metric("Total Missing Values", f"{total_nulls:,}")

    with col2:
        total_cells = df.shape[0] * df.shape[1]
        completeness = ((total_cells - total_nulls) / total_cells) * 100
        st.metric("Data Completeness", f"{completeness:.2f}%")

    with col3:
        st.metric("Total Fields", len(df.columns))

    # Variable distributions (small multiples)
    st.subheader("Variable Distributions Overview")

    numerical_cols = df.select_dtypes(include=[np.number]).columns[:12]  # First 12

    cols_per_row = 4
    for i in range(0, len(numerical_cols), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(numerical_cols[i:i+cols_per_row]):
            with cols[j]:
                fig = px.histogram(df[col].dropna(), nbins=20, title=col)
                fig.update_layout(height=200, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True) 