# AC Telemetry Analytics

A Python-based telemetry capture and analysis tool for Assetto Corsa Competizione that captures real-time driving data at 100Hz and transforms it into insights, built to practice data analytics skills while racing.

## Execution Framework (Fibonacci Ladder)

This project is developed using a **Fibonacci Ladder execution framework**, designed to prioritise completion, iteration, and system-building over novelty.

Each stage represents a shift in **how** the work is approached, not just **what** is built.

---

### Current Position
-  3 — System Formation *(finished)*
-  4-5 - Speed + Experimentation *(finalising)*
-  5-8 - Iteration with real world *(in progress)*


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
Stage 2 consists of two parallel analytics implementations, both built directly on top of the same telemetry reader output and shared analysis logic.

- Reuse ≥70% of code
- Change one dimension at a time
- Learn what matters vs. what doesn’t

**Outcome:**            
`telemetry_analyse_streamlit.py` provides UI visualisations and analytics, `telemetry_analyse_terminal.py` provides CLI-based analytics and ML dataset preparation.

---

### Stage 3 — System Formation 🧩

**Goal:** Turn working scripts into a coherent, reusable system.

This repository intentionally includes early Stage 3 elements, even while Stage 2 work continues.
It introduces the base 'driver.py' and 'feature_schema.py' files.
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

- MLP: Hidden Width (Does more capacity help?), (Does it overfit instantly?
- GA: Mutation Rate (Does more simulation explore or chaos?), (How sensitive is it?
- DQL: Reward Weight (Does reward shaping dominate?), (When does it collapse?)

### Stage 5–8 — Iteration with Real World *(Current)* ⚡

**Goal:** Touch reality — users, data, constraints.

At this stage, learning is no longer offline-only or theoretical.
The *same system* built in Stage 3 and exercised in Stage 4–5 is now
tested directly inside the simulator, frame-by-frame, under real timing constraints.

No new core abstractions are introduced here.
Instead, assumptions are stressed.

**Focus areas:**
- Real-time execution (100Hz loop stability)
- Noisy and imperfect telemetry
- Action clipping, saturation, and deadzones
- Latency between observation → decision → actuation
- Failure modes (spins, stalls, divergence, unsafe actions)

This stage creates **engineering maturity**, not better training metrics.


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

### 📊 Analytics (`telemetry_analyse_terminal.py / telemetry_analyse_streamlit.py`)
Comprehensive analytics tools for ACC telemetry:

**Core Capabilities:**
Session summaries: Max/avg speeds, lap breakdowns, G-forces
Input distributions: Throttle, brake, steering usage patterns
Wheel telemetry: Slip, suspension travel analysis
ML dataset preparation: State-action pairs, normalised datasets, temporal sequences

**Differences:**
telemetry_analyse_terminal.py – CLI-first, textual summaries, fast iteration
telemetry_analyse_streamlit.py – Web-based, interactive charts and sliders

**Shared Foundation:**
Both use feature_schema.py for consistent feature definitions
Both consume CSV data produced by telemetry_reader.py
Differences limited to interface and visualisation


### 🎮 Driver (`driver.py`)

Executes trained or evolving policies inside the live driving environment.

This file closes the loop between:
**state → policy → action → consequence**

It is the system’s **actuation layer** and a prerequisite for any real-world iteration.

**What it consumes:**
- Live vehicle state from shared memory
- Feature schema for state construction
- Trained policy artifacts (Linear, MLP, GA, DQL)

**What it produces:**
- Real-time control commands (steer, throttle, brake)
- Completed laps (or failures)
- Episode-level performance signals:
  - Distance driven
  - Laps completed
  - Stability (off-track, spins, stalls)
  - Policy collapse or divergence

**Why it matters:**
Offline metrics (R², loss) do not guarantee drivability.
`driver.py` is where policies either *move the car* or fail visibly.

This file is introduced in **Stage 3 (System Formation)** and becomes the
primary tool for **Stage 5–8 (Reality Iteration)**.


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

### 🤖 ML Training (`ml_trainer.py`)
Train baseline models to predict driver inputs from vehicle state:

**Models:**
- Linear Regression (simple baseline)
- MLP (Multi-Layer Perceptron) with various architectures
- Extensible for GA or DQL policies in future experiments

**Evaluation:**
- Per-action metrics: RMSE and R² for gas, brake, steering
- Overlay plots comparing actual vs predicted
- Automated model benchmarking and comparison

**Output:**
- Trained models (Linear, MLP, etc.)
- Performance metrics and plots
- Normalized datasets for inference
- Training reports documenting model performance

**Feature Validation:**
- Checks feature count matches schema
- Validates normalization arrays
- Prevents training with mismatched or corrupted datasets

### 📦 Export Tools (`export_file.py`)
Efficiently extract specific columns and rows from large CSV files for analysis without importing entire datasets.

### 🎮 Driver File ('driver.py')
Executes trained or evolving policies in the driving environment and evaluates task-level performance under a fixed interface.

driver.py is the single execution surface for all learning paradigms.
Role in the System:
- Consumes a vehicle state vector (from telemetry or simulator)
- Applies a policy (linear, MLP, GA genome, DQL policy)
- Outputs control actions:
    - Steering
    - Throttle
    - Brake
- Computes task completion metrics at episode end.



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
python -m venv acc_telemetry_venv
```

**3. Activate virtual environment:**
```bash
acc_telemetry_venv\Scripts\activate
```

**4. Install dependencies:**
```bash
# Core analytics
pip install duckdb plotly pandas

# ML training
pip install scikit-learn matplotlib joblib torch

# Controller interface
pip install vgamepad
```

## Usage

### 1️⃣ Capture Telemetry

Start Assetto Corsa Competizione, then run:

```bash
python telemetry_reader.py
```

**What happens:**
- ✅ Connects to ACC's shared memory
- 📊 Streams telemetry at ~100Hz
- 💾 Logs CSV to `telemetry_logs/acc_session_TIMESTAMP.csv`
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

This data is your ground truth human driving behaviour.


### 2️⃣ Analyse Session + Prepare ALL ML Datasets 
Convert raw telemetry into all datasets needed for every learning method:

**Basic analytics:**
📊 Terminal analytics: Session summary, lap breakdown, distributions
```bash
python telemetry_analysis.py telemetry_logs/acc_session_20251214_143022.csv
```

**With visualisations:**
📈 UI visualisation: Session summary, lap breakdown, distributions
```bash
python telemetry_analysis_streamlit.py
```

**Prepare ML datasets:**
🤖 ML datasets: `ml_data/` directory with normalized features
```bash
python telemetry_analyse_terminal.py telemetry_logs/run2/acc_session_combined_2.csv --ml
```
This creates ALL datasets needed for training:
- ml_data/X_states.npy                     (frame-by-frame for Linear / MLP)
- ml_data/y_actions.npy
- ml_data/X_states_normalized.npy
- ml_data/normalization_min.npy
- ml_data/normalization_max.npy
- ml_data/X_sequences.npy                  (sequences for LSTM – future use)
- ml_data/X_sequences_normalized.npy
- ml_data/y_sequences.npy
- ml_data/episodes_states.npy              (episodes for Genetic Algorithm)
- ml_data/episodes_actions.npy
- ml_data/episodes_states_normalized.npy
- ml_data/episode_lengths.npy
- ml_data/dql_transitions.npy              (transitions for Deep Q-Learning)
- ml_data/feature_names.txt                (feature documentation)


### 3️⃣ Train ML Models
All model training can be handled by a single unified script.

```bash
python ml_trainer.py --all

This creates:
ml_results/linear_regression_*.pkl        (Linear baseline)
ml_results/mlp_64_32_*.pkl                (MLP: 64-32)
ml_results/mlp_128_64_32_*.pkl            (MLP: 128-64-32)
ml_results/mlp_256_128_64_*.pkl           (MLP: 256-128-64)
ml_results/ga_model.pkl                   (Genetic Algorithm)
ml_results/dql_model.pt                   (Deep Q-Learning)
ml_results/*.png                          (visualisations)
ml_results/training_report.txt
```
Generated once, reused everywhere:
- Frame data for Linear / MLP
- Episodes for Genetic Algorithms
- Transitions for Deep Q-Learning
- Shared normalisation + feature schema

## Example Workflows

### 🏁 Performance Analysis
```bash
# Capture practice session
python telemetry_reader.py

# Analyze lap times and consistency
python telemetry_analysis.py session.csv 

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

--lr        Train Linear Regression (baseline)
--mlp       Train 3 MLP architectures (64-32, 128-64-32, 256-128-64)
--ga        Train Genetic Algorithm on episode data
--dql       Train Deep Q-Learning on transitions
--all       Train everything (default if no flags given)


```

### 📊 Multi-Session Analysis
```bash
# Combine sessions
cat telemetry_logs/session*.csv > combined.csv

# Analyze aggregate performance
python telemetry_analysis.py combined.csv --ml --plots
```

## Advanced Features


```



## Project Structure

```
ac_telemetry/
├── telemetry_reader.py       # Real-time data capture (100Hz)
├── telemetry_analysis_terminal.py       # Analytics + ML dataset prep
├── telemetry_analysis_streamlit.py      # Streamlit UI
├── driver.py                 # Real-time driving control
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

## License

MIT License - Free to use, modify, and distribute.

---

Built to practice data engineering, analytics, and ML skills while racing. A small tool that helps maintain consistency and actively apply technical concepts instead of just reading them. 🏁

### Skills Practised
- **Data Engineering:** Real-time capture from memory-mapped files, ETL pipelines
- **Analytics:** SQL queries, aggregations, descriptive statistics
- **Feature Engineering:** Transform raw telemetry into ML-ready features
- **Machine Learning:** Supervised learning, model evaluation, prediction
- **Visualization:** Interactive plots, time-series analysis
- **Software Design:** Modular architecture, extensible frameworks
