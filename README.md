# AC Telemetry

Telemetry capture, analysis and autonomous driving for Assetto Corsa Competizione (ACC). It logs your laps at 100 Hz, turns them into graphs and an interactive browser dashboard, and uses the same laps to build a reference line that a controller can drive on its own.

![Dashboard](static/screenshot1.png)

## Quick start

Requires Windows, ACC and Python 3.10+.

```bash
git clone https://github.com/Yaandle/ac_telemetry.git
cd ac_telemetry
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

```bash
python reader.py                                  # record a session (Ctrl+C to stop)
python plot_session.py telemetry_logs/<file>.csv  # PNG graphs in graphs/<file>/
```

Then open `dashboard/index.html` in a browser and drop the CSV onto it.

## What the system does

### Capture (`reader.py`, `acc_memory.py`)
- Reads ACC's shared memory by field name (`acc_memory.py` mirrors the official v1.8.12 layout).
- Logs 66 channels at 100 Hz: inputs, speed, gear, G-forces, world position, track position, lap times, race position and gaps, tyres, brakes, TC/ABS.
- Writes `telemetry_logs/acc_session_<time>_<track>.csv`.

### Analysis (`plot_session.py`, `dashboard/`)
- **`plot_session.py`:** PNG graphs per session: lap times, speed overlay, delta to best, braking points, best-lap inputs, G-G, consistency, tyres, track map, race story.
- **`dashboard/`:** a browser app with no install, organised in five tabs:
  - **Map:** zoomable racing lines per lap, line spread, linked speed / line / delta charts, and full-lap charts below.
  - **Race:** lap times, position, gaps, fuel.
  - **Driving:** G-G, braking zones, input distributions, best-lap inputs, lap consistency.
  - **Car:** tyres, brakes, TC/ABS, slip, suspension.
  - **Data:** session summary and channel stats.

  It can also export the circuit to ML Track Trainer.

### Autonomous driving (`track_line.py`, `pilot.py`, `pilot_sim.py`, `pilot_evolve.py`)
- **`track_line.py`:** builds a reference line from the median of your flying laps. It stores your speed, pedal and gear traces along the line, plus the car's limits (acceleration, braking, grip, steering) fitted from your laps.
- **`pilot.py`:** drives ACC at 100 Hz through a virtual gamepad.
  - `classical`: pure-pursuit steering, with speed set from your trace × `--pace`.
  - `mltt`: a small neural network from ML Track Trainer, with the speed limit kept on as a safety governor.
  - Compensates for ACC scaling gamepad steering down with speed.
- **`pilot_sim.py`:** a stand-in for ACC, fitted to your telemetry, for testing without the game.
- **`pilot_evolve.py`:** neuroevolution (a genetic algorithm over [9, 12, 8, 2] networks) on your real line. `--seed-from` starts from a genome exported from ML Track Trainer.
- **`tracks/`:** reference lines, evolved networks, and the learned steering-gain table.

### Shared definitions (`feature_schema.py`)
The column list and metadata for every logged field, plus the 17-feature state used by the legacy models.

### Utilities
- `export_file.py`: writes a cleaned or column-subset copy of a CSV. It never overwrites the input.
- `basic_reader.py`: a minimal live view while recording.

### Legacy (`legacy/`)
The first attempt at driving, kept for comparison. It trained Linear Regression, MLPs, a Genetic Algorithm and Deep Q-Learning on local car state only, with no track context. None completed a lap. See [legacy/README.md](legacy/README.md).

## Driving a lap

```bash
python track_line.py telemetry_logs/<track laps>.csv --out tracks/<track>.json
python pilot.py --track tracks/<track>.json --sim               # test in the simulator first
python pilot.py --track tracks/<track>.json --pace 0.75         # live in ACC
```

To drive with an evolved network instead:

```bash
python pilot_evolve.py --track tracks/<track>.json --gens 150 --seed-from model.json
python pilot.py --track tracks/<track>.json --policy mltt --weights tracks/<track>.evolved.json
```

**ACC setup:**
- Turn the automatic gearbox on (or pass `--manual-gears`), and keep TC and ABS on.
- Gamepad settings: steer gamma 1.0, deadzones 0.
- Start a practice session and drive onto the track before running. `pilot.py` refuses to start more than 10 m from the line.
- The virtual gamepad needs the ViGEmBus driver, which [vgamepad](https://github.com/yannbouteiller/vgamepad) installs.

## Linked project

[ML Track Trainer](https://github.com/Yaandle/ML-Track-Trainer) is a browser simulator for training small driving networks by neuroevolution.
- It imports circuits from these CSVs, or from the dashboard's export.
- Its **Export best model** file is what `pilot.py --weights` and `pilot_evolve.py --seed-from` read.
- `dashboard/acc-csv.js` is shared by both projects; keep the copies identical.

## Project structure

```
ac_telemetry/
├── acc_memory.py, reader.py      # capture
├── feature_schema.py             # column and feature definitions
├── plot_session.py, dashboard/   # analysis
├── track_line.py                 # reference line + car envelope
├── pilot.py, pilot_sim.py        # driver and simulator
├── pilot_evolve.py               # neuroevolution
├── tracks/                       # lines, networks, steering gains
├── export_file.py, basic_reader.py
├── legacy/                       # first ML attempt
├── docs/                         # examples, execution framework
└── requirements.txt
```

Not in git: `telemetry_logs/`, `training_logs/`, `graphs/`, `ml_data/`, `ml_results/`.

## Examples

**Analysis:** graphs from a 12-minute online race at Mount Panorama.

| | |
|---|---|
| ![Track map](docs/examples/track_map.png) | ![Race story](docs/examples/race_story.png) |
| ![Delta to best](docs/examples/delta_to_best.png) | ![Tyres](docs/examples/tyres.png) |

**Autonomous lap:** using a reference line built from 5 flying laps at Mount Panorama (included in `tracks/`), the classical driver completed a full lap in ACC at 75% pace. Its median distance from the line was 0.3 m, and it never went off track.

In the simulator, at 85% pace:

| Policy | Lap | Max line error |
|---|---|---|
| Classical | 2:23.5 | 1.4 m |
| Network seeded from ML Track Trainer | 2:19.6 | 5.4 m |

## Execution framework

Built in stages using a Fibonacci Ladder: finish something small and real, repeat it with variation, form a system, experiment, then iterate against reality. See [docs/execution_framework.md](docs/execution_framework.md).

## License

MIT
