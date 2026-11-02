# Legacy: the January 2026 system

The first attempt at driving Mount Panorama with machine learning. Kept so the before and after can be compared; the current driver is [`pilot.py`](../pilot.py).

Every model here learned from 17 local-state features (speed, RPM, gear, velocity, accelerations, wheel slip, suspension, and a lap fraction integrated from speed). None of them knew where the road went next, and none completed a lap. Best offline score: Linear Regression, mean R² 0.694 across gas, brake and steer (steer alone 0.505).

| File | What it did |
|---|---|
| `telemetry_analysis_terminal.py` | Terminal analytics; `--ml` builds every dataset in `ml_data/` (frames, sequences, episodes, DQL transitions) |
| `ml_trainer.py` | Trains Linear Regression, three MLPs, a Genetic Algorithm and Deep Q-Learning into `ml_results/` |
| `driver.py` | Runs any of those models live in ACC through a virtual gamepad |
| `ml_performance.py` | Compares the trained models |
| `telemetry_analysis_streamlit.py`, `telemetry_system.py` | Streamlit dashboards, replaced by [`dashboard/`](../dashboard) |
| `reader_offsets_v1.py` | The original logger, reading shared memory by byte offset; replaced by `reader.py` + `acc_memory.py` |
| `QUICK_REFERENCE.txt` | The two-step dataset → training workflow from that time |

Run them from the repository root, so `ml_data/`, `ml_results/` and `feature_schema.py` resolve:

```bash
python legacy/telemetry_analysis_terminal.py telemetry_logs/<file>.csv --ml
python legacy/ml_trainer.py --all
python legacy/driver.py --policy linear
```
