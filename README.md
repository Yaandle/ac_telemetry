# AC Telemetry Analytics

A Python-based telemetry capture and analysis tool for Assetto Corsa Competizione that captures real-time driving data at 100Hz and transforms it into insights, built to practice data analytics skills while racing.

## Execution Framework (Fibonacci Ladder)

This project is developed using a **Fibonacci Ladder execution framework**, designed to prioritise completion, iteration, and system-building over novelty.

Each stage represents a shift in **how** the work is approached, not just **what** is built.

---

### Current Position
- **Partial Stage:** 3 — System Formation *(in progress)*
- **Partial Stage:** 4 - Speed + Experimentation *(in progress)*


---

### Stage 1 — Foundational Win ✅

**Goal:** Finish something small, real, and runnable.

- Single language (Python)
- Single toolchain
- Real input → real output
- Fully working and documented

**Outcome:**  
`telemetry_reader.py` captures live ACC telemetry at **100 Hz** and logs structured CSV data.

---

### Stage 2 — Repetition + Variation ✅

**Goal:** Build the same thing twice with small, meaningful variation.

- Reuse ≥70% of code
- Change one dimension at a time
- Learn what matters vs. what doesn’t

**Outcome:**
- Terminal-based analytics
- CSV / SQL-driven analytics
- Visualization + ML dataset preparation from the same raw data

This stage reinforces **iteration over novelty**.

---

### Stage 3 — System Formation 🧩

**Goal:** Turn working scripts into a coherent, reusable system.

This repository intentionally includes early Stage 3 elements, even while Stage 2 work continues.

**Stage 3 characteristics already present:**
- Modular file structure (capture, analysis, ML, export)
- Clear input/output boundaries
- Feature schema as a single source of truth
- Validation layers to prevent silent errors
- Reproducible workflows

**Repo is understandable in:**
- ~1 minute (README)
- ~10 minutes (code)

**What is *not* claimed yet:**
- Full production hardening
- Long-term API stability
- Performance optimisation beyond correctness

This keeps the system usable **without overengineering**.

---

### Stage 4-5 — Speed + Experimentation *(Current)* ⚡

**Goal:** Run many small, controlled experiments quickly to test ideas against reality.

Testing multiple learning paradigms against the same task to understand how definitions of learning shape behaviour

**Planned focus:**
- Linear Regression (baseline)
- MLP architecture 
- Genetic Algorithms
- Deep Q-Learning

**Each experiment:**
- Changes one variable
- Logs results
- Compares against baselines

MLP: Hidden Width (Does more capacity help?), (Does it overfit instantly?
GA: Mutation Rate (Does more simulation explore or chaos?), (How sensitive is it?
DQL: Reward Weight (Does reward shaping dominate?), (When does it collapse?)



## What The System Does (Files)

### 🎮 Data Capture (`telemetry_reader.py`)
Captures real-time telemetry from Assetto Corsa's shared memory at 100Hz:

**Core Telemetry:**
- Speed, inputs: Throttle, brake, steering (raw + smoothed)
- Lap tracking: Current lap, completed laps, lap fraction, distance
- Vehicle kinematics: Velocity vectors (vx, vy, vz), acceleration (ax, ay, az)
- Derived features: Longitudinal/lateral acceleration, lap progress

**Advanced Telemetry:**
- Wheel dynamics: Slip ratios, slip angles, suspension travel (all 4 wheels)
- Smoothed control inputs: Moving average filter for noise reduction
- Real-time session tracking: Automatic lap detection, statistics

**Output:** Timestamped CSV logs with 38 telemetry channels

### 📊 Analytics Pipeline (`telemetry_analysis.py`)
Comprehensive analysis using DuckDB SQL queries:

**Session Analytics:**
- Session summaries: Max/avg speeds, G-forces, total distance
- Lap-by-lap statistics: Performance breakdown per lap
- Input distributions: Throttle/brake/steering usage patterns
- Wheel telemetry: Slip, slip angles, and suspension travel analysis

**Visualizations:**
- Interactive HTML plots: Speed traces, control inputs, G-G diagrams
- Export-ready data: Cleaned datasets for further analysis

**ML Dataset Preparation:**
- State-action pairs: (18 features) → (3 control outputs)
- Temporal sequences: For LSTM/RNN models
- Normalized datasets: Ready for training
- Feature documentation: Complete schema reference

### 🧩 Feature Schema (`feature_schema.py`)
**Single source of truth** for ML feature definitions:

**Purpose:**
- Ensures consistency across all pipeline components
- Prevents feature order mismatches between training and inference
- Validates data integrity with runtime checks
- Eliminates action leakage (actions never included as state inputs)

**What It Provides:**
- `POLICY_FEATURES`: Exact 18-feature state vector definition
- `ACTION_LABELS`: Output labels (gas, brake, steer)
- `build_state_from_row()`: Construct state from CSV data
- `build_state_from_memory()`: Construct state from shared memory
- `validate_state()`: Runtime validation of feature count
- `validate_normalization()`: Check normalization array shapes

**Why It Matters:**
Without a centralized schema, it's easy to introduce bugs where training uses features in one order but inference expects a different order. This file eliminates that entire class of errors by making feature order explicit and validated.

### 🤖 ML Training (`mlp_trainer.py`)
Train baseline models to predict driver inputs from vehicle state:

**Models:**
- Linear Regression (simple baseline)
- MLP (Multi-Layer Perceptron) with various architectures
- Extensible framework for custom models

**Evaluation:**
- Per-action metrics: RMSE and R² for gas, brake, steering
- Prediction visualizations: Overlay plots comparing actual vs predicted
- Model comparison: Automated benchmarking

**Output:** Trained models, performance reports, visualization plots

**Feature Validation:**
- Automatically validates feature count matches schema
- Checks normalization arrays for consistency
- Prevents training with mismatched data

### 📦 Export Tools (`export_file.py`)
Efficiently extract specific columns and rows from large CSV files for spreadsheet analysis without importing entire datasets.

### 🎮 Driver File ('driver.py')
Executes trained or evolving policies in the driving environment and evaluates task-level performance under a fixed interface.

driver.py is the single execution surface for all learning paradigms.
Role in the System:
- Consumes a vehicle state vector (from telemetry or simulator)
- Applies a policy (MLP, GA genome, DQL policy)
- Outputs control actions:
    - Steering
    - Throttle
    - Brake

Computes task completion metrics at episode end


## Why This Pipeline?

### Skills Practiced
- **Data Engineering:** Real-time capture from memory-mapped files, ETL pipelines
- **Analytics:** SQL queries, aggregations, descriptive statistics
- **Feature Engineering:** Transform raw telemetry into ML-ready features
- **Machine Learning:** Supervised learning, model evaluation, prediction
- **Visualization:** Interactive plots, time-series analysis
- **Software Design:** Modular architecture, extensible frameworks

### Robotics Alignment
This pipeline treats the car as a robotic system with direct parallels to real-world robotics:

| Telemetry Feature | Robotics Analogue |
|------------------|-------------------|
| Velocity vector (vx, vy, vz) | Odometry / state estimation |
| Acceleration (ax, ay, az) | IMU measurements |
| Wheel slip/angles | Contact/terrain feedback |
| Gas/brake/steer | Actuator commands |
| Lap fraction | Mission progress tracker |

This enables sim-to-real transfer research and autonomous driving experiments.

## Quick Start

### Prerequisites

**Install Git:**
1. Download from: https://git-scm.com/download/win
2. Run installer with default settings
3. Restart terminal after installation

**Requirements:**
- Python 3.7+
- Assetto Corsa Competizione (running on Windows for shared memory access)

### Setup

**1. Clone repository:**
```bash
git clone https://github.com/Yaandle/ac_telemetry.git
cd ac_telemetry
```

**2. Create virtual environment:**
```bash
python -m venv ac_telemetry_venv
```

**3. Activate virtual environment:**
```bash
ac_telemetry_venv\Scripts\activate
```

**4. Install dependencies:**
```bash
# Core analytics
pip install duckdb plotly pandas

# ML training (optional)
pip install scikit-learn matplotlib joblib
```

## Usage

### 1️⃣ Capture Telemetry

Start Assetto Corsa Competizione, then run:

```bash
python telemetry_reader.py
```

**What happens:**
- ✅ Connects to ACC's shared memory
- 📊 Displays live telemetry in terminal (5-second updates)
- 💾 Logs data to `telemetry_logs/acc_session_TIMESTAMP.csv`
- ⏹️ Press Ctrl+C to stop and view session statistics

**Output example:**
```
Top Speed:        287.3 km/h  (178.5 mph)
Average Speed:    156.8 km/h
Distance:         23.45 km
Max Throttle:     100.0%
Max Brake:        98.7%
Max Accel:        1.85 G long, 2.12 G lat
Wheel Dynamics:
  Avg Slip: FL 0.53 | FR 0.54 | RL 1.06 | RR 1.05
  Suspension: Front 28mm | Rear 38mm (rear-biased)
```

### 2️⃣ Analyze Session

**Basic analytics:**
```bash
python telemetry_analysis.py telemetry_logs/acc_session_20251214_143022.csv
```

**With visualizations:**
```bash
python telemetry_analysis.py your_session.csv --plots
```

**Prepare ML datasets:**
```bash
python telemetry_analysis.py your_session.csv --ml
```

**What you get:**
- 📊 Terminal analytics: Session summary, lap breakdown, distributions
- 📈 HTML visualizations: `speed_trace.html`, `control_inputs.html`, `acceleration.html`
- 🤖 ML datasets: `ml_data/` directory with normalized features

### 3️⃣ Train ML Models

```bash
python ml_trainer.py --data_dir ml_data --output_dir ml_results
```

**Output:**
- Trained models (Linear, MLP variants)
- Performance metrics (RMSE, R² per action)
- Prediction plots: `ml_results/*.png`
- Training report: `ml_results/training_report.txt`

**Example results:**
```
Linear Regression (18 features):
  Gas R²:   0.564
  Brake R²: 0.556
  Steer R²: 0.321
  Overall:  0.481

MLP (128,64,32):
  Gas R²:   0.6-0.7 (expected)
  Brake R²: 0.6-0.7 (expected)
  Steer R²: 0.4-0.5 (expected)
```

**Note:** Wheel dynamics features (slip, angles, suspension) significantly improve prediction accuracy.

### 4️⃣ Export for Spreadsheets

```bash
python export_file.py
```

Extract specific columns and rows for custom analysis in Excel/Google Sheets without loading massive CSV files.

## Data Structure

### CSV Format (38 columns, 100Hz sampling)

**Timing & Position:**
```
timestamp, lap_time_str, completed_laps, current_lap,
distance_m, lap_fraction
```

**Kinematics:**
```
speed_kmh, speed_ms, gear, rpm,
vx, vy, vz,                    # Velocity vector (m/s)
ax, ay, az,                    # Acceleration (m/s²)
accel_longitudinal,            # Forward/backward accel
accel_lateral                  # Side-to-side accel
```

**Control Inputs:**
```
gas, brake, steer, abs_steer,
gas_smooth, brake_smooth, steer_smooth    # Filtered inputs
```

**Wheel Telemetry (FL, FR, RL, RR):**
```
wheel_slip_*, slip_angle_*, suspension_travel_*
```

**Other:**
```
fuel_kg
```

### ML Dataset Format

**State Features (X): 18-dimensional vector**

*Basic state (10 features):*
- `speed_ms`, `accel_longitudinal`, `accel_lateral`, `abs_steer`
- `gear`, `rpm`, `lap_fraction`
- `vx`, `vy`, `vz` (velocity components)

*Wheel dynamics (8 features):*
- `wheel_slip_fl/fr/rl/rr` (slip ratios)
- `suspension_travel_fl/fr/rl/rr` (travel distance)

**Action Labels (y): 3-dimensional vector**
- `gas` [0-1], `brake` [0-1], `steer` [-1, 1]

**Files generated:**
- `X_states.npy` / `X_states_normalized.npy`
- `y_actions.npy`
- `X_sequences.npy` (for temporal models)
- `normalization_min/max.npy` (for inference)
- `feature_names.txt` (documentation)

## Example Workflows

### 🏁 Performance Analysis
```bash
# Capture practice session
python telemetry_reader.py

# Analyze lap times and consistency
python telemetry_analysis.py session.csv --plots

# Review in browser:
# - speed_trace.html (speed over time with lap markers)
# - control_inputs.html (throttle/brake/steer patterns)
# - acceleration.html (G-G diagram)
```

### 🤖 ML Experiment
```bash
# 1. Capture training data (30+ laps recommended)
python telemetry_reader.py

# 2. Prepare datasets
python telemetry_analysis.py session.csv --ml

# 3. Train models
python ml_trainer.py

# 4. Review predictions
# Check ml_results/ for overlay plots and model comparison
```

### 📊 Multi-Session Analysis
```bash
# Combine sessions
cat telemetry_logs/session*.csv > combined.csv

# Analyze aggregate performance
python telemetry_analysis.py combined.csv --ml --plots
```

## Advanced Features

### Custom SQL Queries

```python
import duckdb

con = duckdb.connect()

# Find fastest lap
query = """
SELECT current_lap, 
       MAX(timestamp) - MIN(timestamp) as lap_time
FROM read_csv_auto('session.csv')
GROUP BY current_lap
ORDER BY lap_time
LIMIT 1
"""

# High-speed braking zones
query = """
SELECT timestamp, speed_kmh, brake
FROM read_csv_auto('session.csv')
WHERE brake > 0.8 AND speed_kmh > 150
"""
```

### Extending the Pipeline

**Add new sensors:**
1. Define memory offset in `telemetry_reader.py`
2. Parse in `ACTelemetryReader.read_frame()`
3. Add to `TelemetryFrame` dataclass
4. Update CSV headers
5. Update `feature_schema.py` if used for ML

**Add custom features:**
1. Modify `FeatureEngineer.create_state_action_pairs()` in `telemetry_analysis.py`
2. Update `POLICY_FEATURES` in `feature_schema.py`
3. Regenerate datasets with `--ml` flag

**Add new models:**
1. Extend `BaselineModel` class in `ml_trainer.py`
2. Implement `train()` method
3. Add to models list in `main()`

## Use Cases

- 🏎️ **Driver improvement:** Analyze braking points, corner speeds, consistency
- 🎓 **Research:** Imitation learning, autonomous driving, sim-to-real transfer
- 📊 **Data science practice:** Real-world ETL, analytics, ML pipelines
- 🤖 **Robotics experiments:** Control policy learning from human demonstrations
- 📈 **Performance engineering:** Tire behavior, G-forces, vehicle dynamics

## Project Structure

```
ac_telemetry/
├── telemetry_reader.py      # Real-time data capture (100Hz)
├── telemetry_analysis.py    # Analytics + ML dataset prep
├── ml_trainer.py             # Train baseline models
├── feature_schema.py         # Feature definitions & validation
├── export_file.py            # Extract data for spreadsheets
├── telemetry_logs/           # Captured session CSVs
├── ml_data/                  # ML-ready datasets (generated)
└── ml_results/               # Model outputs (generated)
```

## Requirements

- Python 3.7+
- DuckDB (SQL analytics)
- Plotly (interactive visualizations)
- Pandas (data processing)
- scikit-learn (ML models, optional)
- matplotlib (plotting, optional)
- joblib (model saving, optional)
- Assetto Corsa Competizione (data source)
- Windows (for shared memory access)

## Troubleshooting

**"ACC shared memory not found"**
- Ensure Assetto Corsa Competizione is running before starting telemetry reader

**Models perform poorly (R² < 0.5)**
- Capture more data (30+ laps minimum)
- Ensure consistent driving style
- Check data quality with `--plots`

**"Feature count mismatch" error**
- Regenerate ML datasets: `python telemetry_analysis.py session.csv --ml`
- This ensures dataset matches current `feature_schema.py`

**Large file sizes**
- Use `export_file.py` to extract relevant columns
- Downsample in analysis: `--downsample 2`
- Compress old sessions: `gzip session.csv`

## Future Enhancements

- [ ] Real-time prediction dashboard
- [ ] LSTM/RNN sequence models
- [ ] Track map overlay with telemetry
- [ ] Multi-car comparison tools
- [ ] Cloud storage integration
- [ ] ACC plugin integration (custom apps)

## Contributing

Contributions welcome! Areas of interest:
- Additional ML architectures (transformers, attention)
- Real-time inference during gameplay
- Track-specific model training
- Visualization improvements

## License

MIT License - Free to use, modify, and distribute.

---

Built to practice data engineering, analytics, and ML skills while racing. A small tool that helps maintain consistency and actively apply technical concepts instead of just reading them. 🏁
