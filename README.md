# AC Telemetry Analytics

**Turning distraction into progress. Turning time into capability.**

A Python-based telemetry capture and analysis tool for Assetto Corsa Competizione that captures real-time driving data at 100Hz and transforms it into actionable insights—built to practice data analytics skills while racing.

## The Story

I've been spending way too much time on ACC. It's fun, but it was taking away from study, work, and responsibilities. When entertainment becomes routine, discipline drops—hours disappear, deadlines get tighter, and long-term goals start slipping.

So I redirected that energy into something useful: I built a tool that processes and explores large datasets, helping me work through my Data Analytics for Business subject while still enjoying the game.

## What It Does

### Data Capture (`telemetry_reader.py`)
Accesses Assetto Corsa's shared memory in real-time, capturing telemetry at 100Hz:
- **Speed, inputs**: Throttle, brake, steering angle
- **Lap tracking**: Current lap, completed laps, lap times
- **Vehicle data**: Gear, fuel, distance traveled
- **Real-time logging**: Writes to CSV with session timestamps
- **Live terminal output**: See telemetry as you drive

### Terminal Analytics (`analyse_telemetry.py`)
Processes CSV files using DuckDB with SQL queries to generate:
- **Session summaries**: Total laps, speed stats, distance covered
- **Lap-by-lap statistics**: Performance breakdown per lap
- **Speed distribution histograms**: Visual frequency analysis
- **Throttle/brake analysis**: Driving style patterns
- **ASCII visualizations**: Speed over time graphs
- **Input usage charts**: Gas and brake usage by lap

### Excel Analysis
Open the same CSVs in spreadsheet software for:
- Custom pivot tables and charts
- Advanced modeling and forecasting
- Flexible data manipulation and visualization

## Skills Practiced

- Loading, cleaning, and inspecting large data files
- Exploring variables, types, distributions, and summaries
- Descriptive analytics and aggregation techniques
- Understanding data structures, veracity, and integrity
- SQL queries and aggregations with DuckDB
- Interpreting and communicating findings
- Real-time data capture from memory-mapped files

## Quick Start

### Prerequisites

**Install Git:**
- Download from: https://git-scm.com/download/win
- Run installer with default settings
- Restart your terminal after installation

### Setup

1. **Clone repository:**
   ```bash
   git clone https://github.com/Yaandle/ac_telemetry.git
   cd ac_telemetry
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv ac_telemetry_venv
   ```

3. **Activate virtual environment:**
   ```bash
   ac_telemetry_venv\Scripts\activate
   ```

4. **Install dependencies:**
   ```bash
   pip install duckdb
   ```

## Usage

### 1. Capture Telemetry (While Racing)

Start Assetto Corsa, then run:
```bash
python telemetry_reader.py
```

This will:
- Connect to AC's shared memory
- Display live telemetry in terminal
- Log data to `telemetry_logs/ac_session_TIMESTAMP.csv`
- Press `Ctrl+C` to stop and see session statistics

### 2. Terminal Analysis (Quick Insights)

```bash
python analyse_telemetry.py telemetry_logs\ac_session_20251209_131159.csv
```

Get instant insights with:
- 📊 Session summary statistics
- 📈 Lap-by-lap performance table
- 📊 Speed distribution histogram
- 🔄 Throttle & brake analysis
- 📈 ASCII speed-over-time graph
- 🎮 Gas & brake usage visualization

### 3. Excel Analysis (Deep Modeling)

Simply open any CSV from `telemetry_logs/` in Excel, Google Sheets, or your preferred spreadsheet software for custom analysis.

## CSV Data Structure

```
lap_time_sec,speed_kmh,gas,brake,steer,gear,fuel,lap_time_str,completed_laps,current_lap,distance_m
```

**Captured at 100Hz** (0.01s intervals):
- `lap_time_sec`: Elapsed session time
- `speed_kmh`: Current speed
- `gas`: Throttle input (0.0-1.0)
- `brake`: Brake input (0.0-1.0)
- `steer`: Steering angle (-1.0 to 1.0)
- `gear`: Current gear number
- `fuel`: Fuel remaining (kg)
- `lap_time_str`: Current lap time display
- `completed_laps`: Number of finished laps
- `current_lap`: Current lap number
- `distance_m`: Total distance traveled (meters)

## Two Ways to Learn

### Terminal-Based (Quick Analysis)
Perfect for rapid iteration and immediate feedback. Run SQL-based analytics in seconds to understand performance patterns.

### Excel/Spreadsheet (Deep Modeling)
Leverage familiar tools for custom pivot tables, advanced charts, and flexible data manipulation. Build models, test hypotheses, create presentations.

---

**It's a small tool, but it helps me stay consistent, build technical skills, and actively apply concepts instead of just reading them.**

## Requirements

- Python 3.7+
- DuckDB
- Assetto Corsa Competizione (for data capture)
- Windows (for shared memory access)

## License

MIT
