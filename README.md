# ac_telemetry

# Assetto Corsa Telemetry Analytics

Terminal-based telemetry data analysis for Assetto Corsa using DuckDB.

## Setup

1. **Create virtual environment:**
   ```bash
   python -m venv ac_telemetry_venv
   ```

2. **Activate virtual environment:**
   ```bash
   ac_telemetry_venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install duckdb
   ```

## Usage

**Run all analyses:**
```bash
python analyze_telemetry.py your_session.csv
```

**Compare two sessions:**
```bash
python analyze_telemetry.py session1.csv -c session2.csv
```

**Interactive SQL mode:**
```bash
python analyze_telemetry.py your_session.csv -i
```

**Export lap summary:**
```bash
python analyze_telemetry.py your_session.csv -e lap_summary.csv
```

## Features

- 📊 Session summary statistics
- 📈 Lap-by-lap performance analysis
- 🏁 Speed distribution histograms
- 🎮 Throttle and brake usage visualization
- ⚖️ Session comparison
- 💻 Interactive SQL query mode

## Requirements

- Python 3.7+
- DuckDB
