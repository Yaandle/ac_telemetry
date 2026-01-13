#!/usr/bin/env python3
"""
Assetto Corsa Telemetry Analysis Pipeline
Comprehensive analytics, visualization, and ML-ready dataset preparation
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from feature_schema import SEQUENCE_LENGTH, STRIDE
try:
    import duckdb
except ImportError:
    print("❌ Missing duckdb. Install with: pip install duckdb")
    sys.exit(1)


# ============================================================================
# DATA LOADER
# ============================================================================

class TelemetryLoader:
    """Load and query telemetry data"""
    
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.con = duckdb.connect()
        self.use_pandas = False
        
        # Always load with pandas and register - this is most reliable
        try:
            self.df = pd.read_csv(csv_path)
            self.con.register('telemetry_data', self.df)
            self.use_pandas = True
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            raise
    
    def query(self, sql: str):
        """Execute SQL query"""
        # Always use telemetry_data table name since we register with pandas
        sql = sql.replace(f"read_csv_auto('{self.csv_path}')", "telemetry_data")
        sql = sql.replace(f"read_csv_auto('{self.csv_path}', HEADER=True)", "telemetry_data")
        return self.con.execute(sql).fetchall()
    
    def query_df(self, sql: str):
        """Execute query and return as DataFrame"""
        # Always use telemetry_data table name since we register with pandas
        sql = sql.replace(f"read_csv_auto('{self.csv_path}')", "telemetry_data")
        sql = sql.replace(f"read_csv_auto('{self.csv_path}', HEADER=True)", "telemetry_data")
        return self.con.execute(sql).fetchdf()
    
    def close(self):
        self.con.close()


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    """Feature engineering for ML models"""
    
    @staticmethod
    def create_state_action_pairs(csv_path: Path, downsample: int = 1):
        """
        Extract state-action pairs for supervised learning
        
        State (X): Vehicle state features (17 features total)
        Action (y): Driver inputs [gas, brake, steer]
        """
        df = pd.read_csv(csv_path)
        
        # STATE FEATURES (17 total - no action-derived signals)
        state_cols = [
            'speed_ms', 'accel_longitudinal', 'accel_lateral',
            'gear', 'rpm', 'lap_fraction', 'vx', 'vy', 'vz',
            'wheel_slip_fl', 'wheel_slip_fr', 'wheel_slip_rl', 'wheel_slip_rr',
            'suspension_travel_fl', 'suspension_travel_fr',
            'suspension_travel_rl', 'suspension_travel_rr'
        ]
        
        # ACTION LABELS
        action_cols = ['gas', 'brake', 'steer']
        
        # Check if all columns exist
        missing_cols = [col for col in state_cols + action_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️  Warning: Missing columns: {missing_cols}")
            state_cols = [col for col in state_cols if col in df.columns]
        
        # Extract and downsample
        X = df[state_cols].values[::downsample]
        y = df[action_cols].values[::downsample]
        
        # Remove any NaN/inf values
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)
        X = X[mask]
        y = y[mask]
        
        return X, y
    
    @staticmethod
    def create_sequences(csv_path: Path, sequence_length: int = 10, downsample: int = 1):
        """
        Create sequences for temporal models (LSTM/RNN)
        
        Returns:
            X_seq: (num_sequences, sequence_length, num_features)
            y_seq: (num_sequences, num_actions)
        """
        X, y = FeatureEngineer.create_state_action_pairs(csv_path, downsample)
        
        if len(X) < sequence_length:
            return np.array([]), np.array([])
        
        # Create sliding windows
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X) - sequence_length):
            X_sequences.append(X[i:i+sequence_length])
            y_sequences.append(y[i+sequence_length])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    @staticmethod
    def normalize_features(X, min_vals=None, max_vals=None):
        """Min-max normalization to [0, 1]"""
        original_shape = X.shape
        
        if len(X.shape) == 3:
            X = X.reshape(-1, X.shape[-1])
        
        if min_vals is None or max_vals is None:
            min_vals = X.min(axis=0)
            max_vals = X.max(axis=0)
        
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        
        X_normalized = (X - min_vals) / range_vals
        
        if len(original_shape) == 3:
            X_normalized = X_normalized.reshape(original_shape)
        
        return X_normalized, min_vals, max_vals


# ============================================================================
# SESSION ANALYTICS
# ============================================================================

def analyze_session_summary(csv_path: Path):
    """Comprehensive session statistics"""
    print("\n" + "="*70)
    print("📊 SESSION SUMMARY")
    print("="*70)
    
    loader = TelemetryLoader(csv_path)
    
    query = """
    SELECT 
        COUNT(*) as total_samples,
        MAX(current_lap) as total_laps,
        ROUND(MAX(timestamp), 1) as session_duration_sec,
        ROUND(MAX(timestamp) / 60, 2) as session_duration_min,
        ROUND(MAX(speed_kmh), 2) as top_speed_kmh,
        ROUND(AVG(CASE WHEN speed_kmh > 5 THEN speed_kmh END), 2) as avg_speed_kmh,
        ROUND(MAX(distance_m) / 1000, 2) as total_distance_km,
        ROUND(MAX(gas), 3) as max_throttle,
        ROUND(MAX(brake), 3) as max_brake,
        ROUND(AVG(abs_steer), 4) as avg_abs_steer,
        ROUND(MAX(abs_steer), 4) as max_abs_steer,
        ROUND(MAX(ABS(accel_longitudinal)) / 9.81, 2) as max_accel_g_long,
        ROUND(MAX(ABS(accel_lateral)) / 9.81, 2) as max_accel_g_lat
    FROM telemetry_data
    """
    
    result = loader.query(query)[0]
    loader.close()
    
    print(f"{'Total Samples:':.<30} {result[0]:>10,}")
    print(f"{'Total Laps:':.<30} {result[1]:>10}")
    print(f"{'Duration:':.<30} {result[3]:>10.2f} min")
    print(f"{'Top Speed:':.<30} {result[4]:>10.2f} km/h ({result[4]/1.609:.1f} mph)")
    print(f"{'Average Speed:':.<30} {result[5]:>10.2f} km/h")
    print(f"{'Total Distance:':.<30} {result[6]:>10.2f} km")
    print(f"{'Max Throttle:':.<30} {result[7]:>10.1%}")
    print(f"{'Max Brake:':.<30} {result[8]:>10.1%}")
    print(f"{'Avg |Steering|:':.<30} {result[9]:>10.4f}")
    print(f"{'Max |Steering|:':.<30} {result[10]:>10.4f}")
    print(f"{'Max Longitudinal Accel:':.<30} {result[11]:>10.2f} G")
    print(f"{'Max Lateral Accel:':.<30} {result[12]:>10.2f} G")
    print("="*70)


def analyze_lap_summary(csv_path: Path):
    """Per-lap performance breakdown"""
    print("\n" + "="*70)
    print("📈 LAP-BY-LAP SUMMARY")
    print("="*70)
    
    loader = TelemetryLoader(csv_path)
    
    query = """
    SELECT 
        current_lap as lap,
        COUNT(*) as samples,
        ROUND(MAX(speed_kmh), 1) as top_speed,
        ROUND(AVG(CASE WHEN speed_kmh > 5 THEN speed_kmh END), 1) as avg_speed,
        ROUND(AVG(gas) * 100, 1) as avg_throttle,
        ROUND(AVG(brake) * 100, 1) as avg_brake,
        ROUND(AVG(abs_steer) * 100, 2) as avg_steer,
        ROUND(MAX(timestamp) - MIN(timestamp), 3) as lap_duration
    FROM telemetry_data
    GROUP BY current_lap
    ORDER BY current_lap
    """
    
    results = loader.query(query)
    loader.close()
    
    print(f"{'Lap':>4} │ {'Samples':>8} │ {'Top':>6} │ {'Avg':>6} │ {'Gas%':>5} │ {'Brake%':>6} │ {'Steer%':>6} │ {'Time':>7}")
    print("─"*70)
    
    for row in results:
        lap, samples, top, avg, gas, brake, steer, duration = row
        minutes = int(duration // 60)
        seconds = duration % 60
        time_str = f"{minutes}:{seconds:05.2f}"
        print(f"{lap:>4} │ {samples:>8,} │ {top:>6.1f} │ {avg:>6.1f} │ {gas:>5.1f} │ {brake:>6.1f} │ {steer:>6.2f} │ {time_str:>7}")


def analyze_input_distributions(csv_path: Path):
    """Analyze throttle, brake, and steering distributions"""
    print("\n" + "="*70)
    print("📊 INPUT DISTRIBUTION ANALYSIS")
    print("="*70)
    
    loader = TelemetryLoader(csv_path)
    
    # Throttle distribution
    print("\n🎮 THROTTLE USAGE")
    query = """
    SELECT 
        CASE 
            WHEN gas > 0.9 THEN 'Full (>90%)'
            WHEN gas > 0.7 THEN 'High (70-90%)'
            WHEN gas > 0.5 THEN 'Medium (50-70%)'
            WHEN gas > 0.2 THEN 'Low (20-50%)'
            WHEN gas > 0 THEN 'Minimal (<20%)'
            ELSE 'Off'
        END as throttle_range,
        COUNT(*) as samples,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as pct
    FROM telemetry_data
    GROUP BY 1
    ORDER BY 2 DESC
    """
    
    results = loader.query(query)
    max_samples = max(r[1] for r in results)
    
    for range_name, samples, pct in results:
        bar_len = int((samples / max_samples) * 40)
        bar = '█' * bar_len
        print(f"{range_name:.<20} {samples:>8,} ({pct:>5.2f}%) {bar}")
    
    # Brake distribution
    print("\n🛑 BRAKE USAGE")
    query = """
    SELECT 
        CASE 
            WHEN brake > 0.7 THEN 'Hard (>70%)'
            WHEN brake > 0.5 THEN 'Medium (50-70%)'
            WHEN brake > 0.3 THEN 'Light (30-50%)'
            WHEN brake > 0 THEN 'Minimal (<30%)'
            ELSE 'Off'
        END as brake_range,
        COUNT(*) as samples,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as pct
    FROM telemetry_data
    GROUP BY 1
    ORDER BY 2 DESC
    """
    
    results = loader.query(query)
    max_samples = max(r[1] for r in results)
    
    for range_name, samples, pct in results:
        bar_len = int((samples / max_samples) * 40)
        bar = '█' * bar_len
        print(f"{range_name:.<20} {samples:>8,} ({pct:>5.2f}%) {bar}")
    
    # Speed distribution
    print("\n🏎️  SPEED DISTRIBUTION")
    query = """
    SELECT 
        FLOOR(speed_kmh / 20) * 20 as speed_bin,
        COUNT(*) as samples
    FROM telemetry_data
    WHERE speed_kmh > 0
    GROUP BY 1
    ORDER BY 1
    """
    
    results = loader.query(query)
    loader.close()
    
    max_samples = max(r[1] for r in results)
    
    for speed_bin, samples in results:
        bar_len = int((samples / max_samples) * 40)
        bar = '█' * bar_len
        print(f"{int(speed_bin):>3}-{int(speed_bin)+20:<3} km/h {samples:>8,} {bar}")


def analyze_wheel_telemetry(csv_path: Path):
    """Analyze wheel slip and load distributions"""
    print("\n" + "="*70)
    print("🔧 WHEEL TELEMETRY ANALYSIS")
    print("="*70)
    
    loader = TelemetryLoader(csv_path)
    
    query = """
    SELECT 
        ROUND(AVG(wheel_slip_fl), 4) as avg_slip_fl,
        ROUND(AVG(wheel_slip_fr), 4) as avg_slip_fr,
        ROUND(AVG(wheel_slip_rl), 4) as avg_slip_rl,
        ROUND(AVG(wheel_slip_rr), 4) as avg_slip_rr,
        ROUND(MAX(wheel_slip_fl), 4) as max_slip_fl,
        ROUND(MAX(wheel_slip_fr), 4) as max_slip_fr,
        ROUND(MAX(wheel_slip_rl), 4) as max_slip_rl,
        ROUND(MAX(wheel_slip_rr), 4) as max_slip_rr
    FROM telemetry_data
    """
    
    result = loader.query(query)[0]
    loader.close()
    
    print("\n🔄 WHEEL SLIP (Average | Max)")
    print(f"  Front Left:  {result[0]:.4f} | {result[4]:.4f}")
    print(f"  Front Right: {result[1]:.4f} | {result[5]:.4f}")
    print(f"  Rear Left:   {result[2]:.4f} | {result[6]:.4f}")
    print(f"  Rear Right:  {result[3]:.4f} | {result[7]:.4f}")


def analyze_suspension_telemetry(csv_path: Path):
    """Analyze suspension travel"""
    print("\n" + "="*70)
    print("🔧 SUSPENSION TRAVEL ANALYSIS")
    print("="*70)
    
    loader = TelemetryLoader(csv_path)
    
    query = """
    SELECT 
        ROUND(AVG(suspension_travel_fl) * 1000, 2) as avg_fl,
        ROUND(AVG(suspension_travel_fr) * 1000, 2) as avg_fr,
        ROUND(AVG(suspension_travel_rl) * 1000, 2) as avg_rl,
        ROUND(AVG(suspension_travel_rr) * 1000, 2) as avg_rr,
        ROUND(MAX(suspension_travel_fl) * 1000, 2) as max_fl,
        ROUND(MAX(suspension_travel_fr) * 1000, 2) as max_fr,
        ROUND(MAX(suspension_travel_rl) * 1000, 2) as max_rl,
        ROUND(MAX(suspension_travel_rr) * 1000, 2) as max_rr,
        ROUND(MIN(suspension_travel_fl) * 1000, 2) as min_fl,
        ROUND(MIN(suspension_travel_fr) * 1000, 2) as min_fr,
        ROUND(MIN(suspension_travel_rl) * 1000, 2) as min_rl,
        ROUND(MIN(suspension_travel_rr) * 1000, 2) as min_rr
    FROM telemetry_data
    """
    
    result = loader.query(query)[0]
    loader.close()
    
    print("\n  Average Travel (mm):")
    print(f"    FL: {result[0]:>6.2f}  FR: {result[1]:>6.2f}")
    print(f"    RL: {result[2]:>6.2f}  RR: {result[3]:>6.2f}")
    
    print("\n  Max Compression (mm):")
    print(f"    FL: {result[4]:>6.2f}  FR: {result[5]:>6.2f}")
    print(f"    RL: {result[6]:>6.2f}  RR: {result[7]:>6.2f}")
    
    print("\n  Max Extension (mm):")
    print(f"    FL: {result[8]:>6.2f}  FR: {result[9]:>6.2f}")
    print(f"    RL: {result[10]:>6.2f}  RR: {result[11]:>6.2f}")


# ============================================================================
# VISUALIZATION
# ============================================================================



def plot_control_inputs(csv_path: Path, lap: Optional[int] = None, 
                       output_file: str = "control_inputs.html"):
    """Plot gas, brake, and steering for analysis"""
    print(f"\n🎮 Generating control input visualization (Lap {lap if lap else 'All'})...")
    
    loader = TelemetryLoader(csv_path)
    
    if lap:
        query = f"""
        SELECT timestamp, gas, brake, steer, speed_kmh
        FROM telemetry_data
        WHERE current_lap = {lap}
        ORDER BY timestamp
        """
    else:
        query = """
        SELECT timestamp, gas, brake, steer, speed_kmh
        FROM telemetry_data
        ORDER BY timestamp
        """
    
    data = loader.query(query)
    loader.close()
    
    timestamps = [d[0] for d in data]
    gas = [d[1] for d in data]
    brake = [d[2] for d in data]
    steer = [d[3] for d in data]
    speed = [d[4] for d in data]
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Throttle & Brake', 'Steering', 'Speed'),
        vertical_spacing=0.08
    )
    
    fig.add_trace(go.Scatter(x=timestamps, y=gas, name='Gas', 
                            line=dict(color='#2ecc71')), row=1, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=brake, name='Brake', 
                            line=dict(color='#e74c3c')), row=1, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=steer, name='Steering', 
                            line=dict(color='#9b59b6')), row=2, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=speed, name='Speed', 
                            line=dict(color='#3498db')), row=3, col=1)
    
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Input [0-1]", row=1, col=1)
    fig.update_yaxes(title_text="Steer [-1,1]", row=2, col=1)
    fig.update_yaxes(title_text="Speed (km/h)", row=3, col=1)
    
    fig.update_layout(height=900, showlegend=True, template='plotly_white',
                     title_text=f"Control Inputs - Lap {lap}" if lap else "Control Inputs - Full Session")
    
    fig.write_html(output_file)
    print(f"✅ Saved: {output_file}")


def plot_acceleration_analysis(csv_path: Path, output_file: str = "acceleration.html"):
    """G-G diagram"""
    print("\n🎯 Generating G-G diagram...")
    
    loader = TelemetryLoader(csv_path)
    query = "SELECT accel_longitudinal, accel_lateral, speed_kmh FROM telemetry_data WHERE speed_kmh > 10"
    data = loader.query(query)
    loader.close()
    
    long_g = [d[0] / 9.81 for d in data]
    lat_g = [d[1] / 9.81 for d in data]
    speed = [d[2] for d in data]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lat_g, y=long_g, mode='markers',
        marker=dict(size=2, color=speed, colorscale='Viridis', showscale=True,
                   colorbar=dict(title="Speed (km/h)"))
    ))
    
    fig.update_layout(
        title="G-G Diagram", xaxis_title="Lateral Accel (g)", 
        yaxis_title="Longitudinal Accel (g)", template='plotly_white',
        height=700, width=700, xaxis=dict(scaleanchor="y"), yaxis=dict(scaleanchor="x")
    )
    fig.write_html(output_file)
    print(f"✅ Saved: {output_file}")


# ============================================================================
# ML DATASET PREPARATION
# ============================================================================

def prepare_ml_dataset(csv_path: Path, output_dir: Path = Path("ml_data")):
    """Prepare and save ML-ready datasets"""
    print("\n" + "="*70)
    print("🤖 ML DATASET PREPARATION")
    print("="*70)
    
    output_dir.mkdir(exist_ok=True)
    
    # Frame-by-frame dataset
    print("\n📊 Creating frame-by-frame dataset...")
    X, y = FeatureEngineer.create_state_action_pairs(csv_path, downsample=1)
    
    if len(X) > 0:
        print(f"  State features (X): {X.shape}")
        print(f"  Action labels (y): {y.shape}")
        
        X_norm, min_vals, max_vals = FeatureEngineer.normalize_features(X)
        
        np.save(output_dir / "X_states.npy", X)
        np.save(output_dir / "y_actions.npy", y)
        np.save(output_dir / "X_states_normalized.npy", X_norm)
        np.save(output_dir / "normalization_min.npy", min_vals)
        np.save(output_dir / "normalization_max.npy", max_vals)
        
        print(f"  ✅ Saved to {output_dir}/")
        print(f"     - X_states.npy (raw features)")
        print(f"     - X_states_normalized.npy (normalized)")
        print(f"     - y_actions.npy (gas, brake, steer)")
        print(f"     - normalization_min/max.npy (for inference)")
    
    # Sequence dataset
    print("\n🔁 Creating sequence dataset (for LSTM/RNN)...")
    X_seq, y_seq = FeatureEngineer.create_sequences(
        csv_path,
        sequence_length=SEQUENCE_LENGTH,
        downsample=STRIDE
    )

    if len(X_seq) > 0:
        print(f"  Sequence features (X): {X_seq.shape}")
        print(f"  Next actions (y): {y_seq.shape}")
        
        X_seq_norm, _, _ = FeatureEngineer.normalize_features(X_seq)
        
        np.save(output_dir / "X_sequences.npy", X_seq)
        np.save(output_dir / "X_sequences_normalized.npy", X_seq_norm)
        np.save(output_dir / "y_sequences.npy", y_seq)
        
        print(f"  ✅ Saved sequence data")
    
    # Feature descriptions
    feature_names = [
        "speed_ms", "accel_longitudinal", "accel_lateral",
        "gear", 'rpm', "lap_fraction", "vx", "vy", "vz",
        "wheel_slip_fl", "wheel_slip_fr", "wheel_slip_rl", "wheel_slip_rr",
        "suspension_travel_fl", "suspension_travel_fr", 
        "suspension_travel_rl", "suspension_travel_rr"
    ]
    
    with open(output_dir / "feature_names.txt", "w") as f:
        f.write("STATE FEATURES (X):\n")
        for i, name in enumerate(feature_names):
            f.write(f"  {i}: {name}\n")
        f.write("\nACTION LABELS (y):\n")
        f.write("  0: gas [0-1]\n")
        f.write("  1: brake [0-1]\n")
        f.write("  2: steer [-1, 1]\n")
    
    print("\n📝 Feature descriptions saved to feature_names.txt")
    print("="*70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive telemetry analysis and ML dataset preparation'
    )
    parser.add_argument('csv_file', help='Path to telemetry CSV file')
    parser.add_argument('--ml', action='store_true', 
                       help='Prepare ML-ready datasets')
    parser.add_argument('--lap', type=int, 
                       help='Analyze specific lap')
    parser.add_argument('--plots', action='store_true',
                       help='Generate all visualizations')
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"❌ File not found: {csv_path}")
        sys.exit(1)
    
    print("="*70)
    print("ASSETTO CORSA TELEMETRY ANALYSIS PIPELINE")
    print("="*70)
    print(f"📁 Data: {csv_path}")
    
    # Core analytics
    
    # Core analytics    
    analyze_session_summary(csv_path)
    analyze_lap_summary(csv_path)
    analyze_input_distributions(csv_path)
    analyze_wheel_telemetry(csv_path)
    analyze_suspension_telemetry(csv_path)
    
    # ML dataset preparation
    if args.ml:
        prepare_ml_dataset(csv_path)
    
    # Visualizations
    if args.plots:
        plot_control_inputs(csv_path, lap=args.lap)
        plot_acceleration_analysis(csv_path)
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70)
    
    if not args.ml:
        print("\n💡 Tip: Use --ml to generate ML-ready datasets")
    if not args.plots:
        print("💡 Tip: Use --plots to generate interactive visualizations")

if __name__ == "__main__":
    main()
