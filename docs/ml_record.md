# ML record: driving Mount Panorama

A running record of the driving system: what each method does, which policy is best,
where its data came from, and every live and training result. Numbers here are checked
against logs in `training_logs/` (not in git) or recomputed from `telemetry_logs/`.
Update it after every live test or training run. The final conclusion is written from it.

Goal: an AI that drives Mount Panorama in ACC near the human 2:01–2:04 laps, then race starts.

---

## Best policy so far

| | |
|---|---|
| **Policy** | `classical` (pure pursuit + speed envelope), `--pace 0.9` |
| **Trained from** | not trained: geometry. The full laps below drove the 1 Nov line: median of 5 flying laps (2:01.3–2:02.2), now `tracks/mount_panorama_20261101.json`. `tracks/mount_panorama.json` has been the 12-lap line since 2 Nov ~18:00 (see Data sources) |
| **Steering** | `SteerComp` with gains from `tracks/acc_steer_gain.json` |
| **Best timed laps** | 2 consecutive laps at pace 0.9, lap 1 ACC 2:19.852 (invalid). `pilot_classical_20261102_132857.csv` |
| **Best live result** | full lap at pace 0.9: 6.17 km in 134.4 s by frame count (≈ 140 s real; see the timing note below), starting 445 m before the line (not an ACC-timed lap). Median 0.29 m from the line, max 5.38 m, 0 s recovering, 2.0% outside the measured road. `pilot_classical_20261102_130945.csv` |
| **Earlier full lap** | pace 0.75: 157.5 s by frame count (≈ 163 s real), median 0.32 m, max 3.07 m, never outside the measured road. `pilot_classical_20261102_105851.csv` |
| **Best network live** | `tracks/mount_panorama.edges_seeded_hotlap.json`, pace 0.75, Hotlap: out-lap + **one full timed lap, ACC 2:38.902, ACC-invalid** (invalidated at 3,494 m, 56.7%). Median 0.52 m, max 8.96 m from the line, **2.99 s of classical recovery steering** (3,477–3,522 m), pedals set by the speed envelope on 40.9% of frames. `pilot_mltt_20261106_191618.csv` (6 Nov) |
| **Before that** | `tracks/mount_panorama.seeded.json`: just under 55% of a lap, twice: 54.8% and 54.6% (`pilot_mltt_20261102_105030.csv`, `..._105613.csv`) |

One network has finished a live lap (6 Nov): invalid, with 3.0 s of classical recovery and the pedals shared with the speed envelope. No network has a valid lap, or a lap without recovery, yet.

---

## Methods in use

| Method | Role | Where |
|---|---|---|
| Reference line | Median path of flying laps, with your speed / pedal / steer / gear stored against distance. Everything else is measured relative to it. | `track_line.py` |
| Classical pure pursuit + speed envelope | Steers at a look-ahead point on the line. Speed target = your speed trace × `--pace`, capped by a braking envelope from the car's fitted deceleration. Also the governor and recovery fallback for networks. | `pilot.py` `Classical` |
| Per-band median steer gain | For each 5 m/s speed band, the median of steer ÷ path curvature over your laps (not a regression). Maps curvature to steer in the sim and the classical controller. | `track_line.car_envelope` |
| SteerComp | ACC scales gamepad steering down with speed. SteerComp learns ACC's gain (measured ÷ requested steer) per 5 m/s band during live runs and divides it out. Saved to `tracks/acc_steer_gain.json`. | `pilot.py` `SteerComp` |
| Look-ahead linear regression | Shadow-mode check: LR on the same look-ahead inputs explains R² 0.52–0.56 of your steering. A baseline, not a driver. | README results |
| Neuroevolution, from scratch | ML Track Trainer's GA (64 drivers, [9, 12, 8, 2] tanh) in the fitted-car sim on the real line. | `pilot_evolve.py` |
| Neuroevolution, seeded | Same GA, starting from a genome exported from ML Track Trainer's browser sim. | `pilot_evolve.py --seed-from` |
| Realistic road edges (2 Nov) | The GA sim's road is now where you actually drove, per point and per side (98th percentile of your lateral offset per 10 m, from 4 Bathurst sessions), + 0.5 m. Wall 3 m beyond. Over the grip limit the car scrubs speed instead of clamping for free. | `track_line.py --edges`, `tracks/mount_panorama.edges.json` |
| January methods (comparison only) | Linear / MLP / GA / DQL on 17 local-state features, no track context. None completed a lap. | `legacy/` |

### January numbers (for comparison)

- 17 features. LR R² 0.6936 is the mean over gas, brake and steer. Steering alone: 0.5054.
- Shadow-mode r (current classical pilot steer vs yours, no training): 0.71 (30 Oct line), 0.73 (1 Nov line). An earlier 0.77 can't be reproduced.
- Look-ahead LR: R² 0.52–0.56. An earlier 0.58 can't be reproduced.

---

## Data sources

| Source | Used for |
|---|---|
| `telemetry_logs/acc_session_20261030_*_mount_panorama.csv` | First line (8 flying laps, 2:01.4–2:04.9), `tracks/mount_panorama_20261030.json` |
| `telemetry_logs/acc_session_20261101_*_mount_panorama.csv` | Line of 1 Nov (5 flying laps, 2:01.3–2:02.2), kept as `tracks/mount_panorama_20261101.json` (+ `.edges.json`). All results before 2 Nov ~18:00 used it |
| `acc_session_20261101_182425` + `acc_session_20261102_161217_mount_panorama.csv` | **Current line from 2 Nov ~18:00**: 12 flying laps (2:01.3–2:03.6; today's 7 at 2:02.2–2:03.6, single player, clean). A median 0.12 m from the 1 Nov line (95% within 1.33 m, max 1.91 m). Edges re-measured from all 5 Bathurst sessions. Classical in the live-like sim: 2:40.0 / 2:13.9 at 0.75 / 0.9 (1 Nov line: 2:39.0 / 2:13.2), same line error |
| All 4 Bathurst sessions above | Road edges, `tracks/mount_panorama.edges.json` |
| ML Track Trainer browser genome, gen 147 | `tracks/mount_panorama.mltt_gen147.json`, seed for `.seeded.json` |
| Evolution sim | `.evolved.json`, `.seeded.json`, `.edges_seeded.json` |
| ML Track Trainer `acc-track-import` (sliding car), gen 419, trained on its import of `acc_session_20261102_161217` | `tracks/mount_panorama.mltt_gen419.json`, seed for `.gen419_hotlap.json` and `.gen419_hotlap_w7.json` (6 Nov) |

Line accuracy: distance along the line matches ACC's spline position with median 0.15 m error, 95% within 0.33 m (`track_line.py`'s line, not the dashboard export).

reader.py logs 66 columns at 100 Hz (median sample gap 0.010 s). Two are `timestamp` and the `lap_time_str` text.

**TC / ABS activity (checked 6 Nov).** Commit `2dab330` (1 Nov 18:40) reads ACC's float `tc` / `abs` fields ("TC in action" / "ABS in action" in ACC's shared-memory doc v1.8.12) instead of the int `tcinAction` / `absInAction`. Before it, `tc_active` and `abs_active` were 0 on every frame (`acc_session_20261101_172854`: 80,041 frames, 11,373 braking; `..._182425`: 77,361 frames, 10,446 braking). The first session recorded after it, `acc_session_20261102_161217_mount_panorama.csv`, per lap (time = ACC's lap time):

| Lap | Time | TC % of lap | ABS % of lap | ABS % of braking frames | ABS % off the brakes | TC % of throttle frames | TC % off the throttle |
|---|---|---|---|---|---|---|---|
| 1 (out-lap) | 2:05.04 | 1.20 | 6.72 | 52.8 | 0.03 | 1.68 | 0.00 |
| 2 | 2:03.39 | 2.16 | 5.33 | 46.3 | 0.02 | 2.70 | 0.00 |
| 3 | 2:02.86 | 1.68 | 5.89 | 49.3 | 0.01 | 2.09 | 0.04 |
| 4 | 2:03.57 | 2.09 | 5.10 | 45.6 | 0.00 | 2.61 | 0.04 |
| 5 | 2:02.62 | 2.18 | 6.09 | 50.7 | 0.01 | 2.71 | 0.00 |
| 6 | 2:03.51 | 1.96 | 5.46 | 48.2 | 0.01 | 2.48 | 0.00 |
| 7 | 2:02.25 | 1.41 | 5.84 | 48.8 | 0.09 | 1.73 | 0.00 |
| 8 | 2:02.90 | 2.00 | 4.60 | 43.2 | 0.02 | 2.52 | 0.00 |

(braking = brake > 0.05, throttle = gas > 0.05; lap 9 was incomplete.) The flags now behave like activity, not a setting: ABS fires on about half the braking frames and almost never off the brakes, TC only on throttle. If the float fields held the TC/ABS level instead, both would read ~100%. The dashboard's per-lap aids chart counts the same flags (frames above 5 km/h), so it should now show these values.

---

## Live results

Fields: distance (fraction of the 6167 m lap), time, median / max distance from the line, time in recovery (`lost`).
Live logs written before 2 Nov 13:00 have no record of `--pace`. Newer ones have a `.json` summary next to the CSV.

**Lap timing changed 2 Nov ~13:25, and the live clock was fixed ~13:40.** Before ~13:40 every live time from `pilot.py` (including the 157.5 s and 134.4 s full laps) counted 10 ms frames. The live loop runs ~3.5–4% under 100 Hz, so those times are short. From distance ÷ speed in each log (which reads 0.3% high against ACC's 2:19.852), the 157.5 s lap at 0.75 was really ≈ 163 s and the 134.4 s lap at 0.9 ≈ 140 s. ACC's own lap times (`acc_s`) were never affected. Runs up to then measure one lap of distance from wherever the car started (not an ACC-timed lap). From then on `pilot.py` times laps start/finish to start/finish: an untimed out-lap to the first crossing, then `--laps` timed laps, each with ACC's own lap time (`iLastTime`) and validity (`isValidLap`, ACC's track-limits flag) live. The sim now starts 150 m before the line (`--start-s`), so sim lap times after this are flying laps with a short run-up, not standing starts. Add per-lap rows below from the `.json` summaries.

| Log | Policy | Distance | Time | Median / max line error | Recovery | Notes |
|---|---|---|---|---|---|---|
| `pilot_classical_20261101_184553` | classical | 22.8% (470 → 1878 m) | 37.9 s | 1.78 / 9.38 m | 2.7 s | Before SteerComp. ACC applied 0.2–0.5× steer above 25 m/s. Spun on the climb to The Cutting. |
| `pilot_classical_20261102_103148` | classical | 42.5% | 91.4 s | 1.75 / 11.78 m | 21.5 s | Before SteerComp |
| `pilot_mltt_20261102_104715` | gen 147 | 7.1% | 32.1 s | 5.79 / 12.84 m | 17.4 s | |
| `pilot_mltt_20261102_104818` | gen 147 | 27.7% | 85.2 s | 4.49 / 22.03 m | 21.0 s | |
| `pilot_mltt_20261102_105030` | seeded | 54.8% | 93.7 s | 1.20 / 7.57 m | 7.9 s | Was given as 54.9%; summing the log's distance gives 3,382 m of 6,167 m = 54.8% (6 Nov) |
| `pilot_mltt_20261102_105613` | seeded | 54.6% | 101.3 s | 1.17 / 6.74 m | 16.4 s | |
| `pilot_classical_20261102_105851` | classical, pace 0.75 | **100%** | 157.5 s | 0.32 / 3.07 m | 0 s | First full lap. Standing start. 0% outside measured road. |
| `pilot_classical_20261102_130945` | classical, pace 0.9 | **100%** | 134.4 s | 0.29 / 5.38 m | 0 s | Started at s = 5722 m. 2.0% outside measured road (worst 2.34 m past). ACC steer ÷ requested 0.95–1.04 at 72–288 km/h. Pace 0.8 and 0.85 were skipped. |
| `pilot_classical_20261102_132614` | classical, pace 0.75 | out-lap + most of lap 1 | 132.4 s (frame count) | 0.30 / 2.33 m | 0 s | First run with start/finish timing. Stopped before the end of lap 1. |
| `pilot_classical_20261102_132857` | classical, pace 0.9, `--laps 3` | out-lap + **2 timed laps** | lap 1 **ACC 2:19.852 (invalid)**; lap 2 ≈ 2:19.5 | lap 1 0.45 / 7.73 m, lap 2 0.44 / 7.76 m | 2.7 s, 2.0 s | First consecutive laps. Stopped 0.9 s into lap 3, before ACC counted lap 2, so lap 2 has no ACC time. The ≈ 2:19.5 is from distance ÷ speed in the log. The pilot printed 2:15.29 / 2:14.53 because it counted frames: the live loop runs ~3.7% under 100 Hz. Fixed: live laps now use real time. |

**2 Nov ~15:40 session (real-time clock):**

| Log | Policy | Result | Notes |
|---|---|---|---|
| `pilot_classical_20261102_153930` | classical 0.95 | lap 1 **2:14.276** (real clock, ACC-invalid), median 0.37 / max 9.8 m, 1.9 s recovering; spun at ~8% of lap 2 | ACC's lap counter only went 0 → 1 (out-lap), so lap 1 has no ACC time |
| `pilot_mltt_20261102_154258`, `..._154359`, `..._154434` | edges_seeded 0.9 | out-lap only, ~6.5 s | Steering swings full lock left/right (±1.0 joystick) from launch, car drifts 2–3 m off line, then ACC resets the session (car back at start, lap counter 0, rpm 0) |
| `pilot_mltt_20261102_154513`, `..._154545` | gen 147 0.75 | ~2–3 s | Same weave and reset |
| `pilot_mltt_20261102_154615` | evolved 0.75 | ~3 s | Same weave and reset |

Conclusion so far: classical is the only policy that drives laps live. Every network weaves at launch with saturated steering. The likely cause is SteerComp multiplying the network's frame-to-frame steering noise by 1/gain (2–5×), which the sim's `--acc-steer` doesn't reproduce. To investigate before more network runs.

**2 Nov ~17:10 session (Hotlap mode, manual gearbox, automatic startup + launch):**

| Log | Policy | Result | Notes |
|---|---|---|---|
| `pilot_classical_20261102_171101` | classical 0.9, `--manual-gears` | lap 1 **2:17.344** (real clock), median 0.43 / max 7.6 m, 0.8 s recovering; ACC invalid (~60%). Lap 2 to 62%, then stopped | ACC counted the lap at 152.56 s, but the pilot took it as the out-lap (Hotlap doesn't count the out-lap), so no ACC time. Fixed: ACC laps now matched to crossings by time. |
| `pilot_mltt_20261102_171536`, `..._171622` | browser gen 19 refined, 0.9 | out-lap only: 16.4 s, 10.4 s | **Stuck in 1st** (no `--manual-gears`): rev limiter at 8,390 RPM, 128–131 km/h. Steering at full lock 83–87% of the time. ACC Hotlap invalidated the lap and reset the car |
| `pilot_mltt_20261102_171727`, `..._171804` | seeded, real edges, 0.9 | 7.1–7.2 s | Same: 1st gear, full lock 57–60% of the time, reset |
| `pilot_mltt_20261102_171940`, `..._172009` | evolved from scratch, 0.9 | 3.4 s | Full lock 61–65% of the time, 4.8 m off the line, invalidated at 3.4 s, stopped |

The network failures are confounded by the gearbox (1st only), but the steering is the main fault: from launch the networks swing between full lock either way, which no gearbox explains. Hotlap mode ends any run at the first invalidation, so networks get a few seconds at most. Fixed in `pilot.py`: manual shifting switches on automatically when ACC's auto gearbox is off, and a reset by ACC stops the run with a message.

**Weaving fixed in `pilot.py` (2 Nov ~17:40).** Cause: input 8 of the network ("current steer") was fed ACC's measured steer. In training it is the car's own steer, which follows the network's target with a 0.10 s lag. Live, ACC's steer answers within a frame, so the networks reacted to their own last output: the steer target changed sign on 42–56% of frames (0.58 per frame on average) and the joystick sat at full lock 60–87% of the time. The 1 Nov run `pilot_mltt_20261102_105030` was smooth (0.02 per frame), so ACC's response probably got faster when the gamepad settings changed. Fix: the network keeps its own lagged steer (0.10 s, as in `pilot_evolve.py`), reads that as input 8 and sends it to ACC. When the classical fallback steers, the internal steer follows it. No retraining needed.

Sim check with live-like steering (0.01 s response, 1-frame delay, ±0.05 noise on measured steer), 0.9 pace, 1 lap:

| Genome | Before (0.01 s, no noise) | After |
|---|---|---|
| Browser gen 19, refined | sign flips 22% of frames, full lock 20% | 1.5% flips, full lock 1%, lap 2:08.3, median 0.39 m |
| Seeded, real edges | 15% flips, full lock 11% | 1.2% flips, full lock 3%, lap 2:09.0, median 0.83 m |
| Seeded | | 1.9% flips, full lock 3%, lap 2:10.1, median 2.11 m |

`pilot.py --sim --steer-lag 0.01` now reproduces a fast ACC steering response.

**2 Nov ~18:25, Hotlap genomes live (0.9, manual gearbox, new line):**

| Log | Genome | Result | What happened |
|---|---|---|---|
| `pilot_mltt_20261102_182532` | seeded, Hotlap | out-lap only, reset at 10.6 s (Murray's) | Braking 0.75–0.86 into Murray's while downshifting (gear 3 → 2 → 0 → 1), the network asked for −0.34 → −0.98 steer (classical uses at most −0.23 here). ACC's steer went the *opposite* way (+0.06 → +0.91) as the car slid; off at +3.5 m |
| `pilot_mltt_20261102_182630` | browser, Hotlap | **cleared the out-lap**, then 26% of lap 1 (first network past the out-lap in Hotlap) | Smooth to 1,500 m (steer within ±0.13, 228 km/h). Drifted to +2.0 m on the brakes into Griffins Bend, then asked for +0.43 → +1.0 steer (classical: at most +0.18), joystick saturated, ACC's steer again swung against it, spun at 1,600 m |

Both failures: the network asks for 3–5× the steer classical uses in the same corner, the car slides (worse with manual downshifts under braking), and ACC's measured steer then diverges from the request (probably ACC's gamepad counter-steer once the car rotates). The sim's car just understeers at the grip limit, so it never punishes over-steering this way. Weaving is fixed; this is a grip/stability gap.

**Grip limit on network steering (2 Nov ~18:45, `pilot.py --grip-limit`, default 2.0).** Cap = limit × steer_gain(v) × lat_max(v) / v², the steer that reaches the car's measured cornering grip at that speed. Classical stays under 0.82–0.92× it 99% of the time; the networks reached 2–3.5× where they crashed (182530, 182630). In the live-like sim at 0.9: 1.1 made things worse (laps slower, more off-road, and with the car needing ×1.25 steer, the cap blocked needed steer: 34–40% of frames capped, crash/stuck); 1.5 still capped 10–27%; 2.0 changes the sim laps by < 0.1 s and caps 1–5% of frames. The sim can't show the benefit (its car never slides), so 2.0 is a safety rail for live, to be judged on live runs.

**ML Track Trainer car rebuilt to slide like ACC (2 Nov ~19:30).** Cause of the live failures: the browser car scaled full steer to the grip limit and never slid, and the imported Bathurst road was 33 m wide. Now (`env.js`, `car-fit.js`, ML-Track-Trainer `acc-track-import`):
- Dynamic bicycle model: front/rear tyres with grip falling past the peak (Pacejka C 1.45), load transfer (CG height 0.36 m), friction circle, ABS 0.95, TC 0.85, brake balance 65% front, gamepad counter-steer 0.7 × body slip.
- Car fitted from the imported session per 5 m/s band (same method as `track_line.car_envelope`; on the 2 Nov session the JS and Python fits agree within 0.3 m/s² for accel/brake and exactly for steer gain). Steering in ACC units: full lock = 0.258 rad road wheel, understeer K 0.00057 from the steer-gain fit. Steady-cornering steer matches telemetry ×0.84–1.14 at 25–50 m/s.
- Checks: braking 0.8 at 85% of grip spins it (78° slip); 3× the grip steer runs the front wide. A driver following your speed trace with pure pursuit: 0.85 → flying 2:25.7, **0.9 → flying 2:18.0 (classical live in ACC at 0.9: 2:17.3)**, 0.95 → 2:11.3 with 11° slip moments (classical spun on lap 2 at 0.95 live). No time off the 12 m road.
- Imported roads are 12 m wide (was 0.0053 × lap length = 33 m). Exported models carry `wall_m`, so `pilot.py` scales the lateral input the same way.
- Setup the telemetry was driven with: `setups/ford_mustang_gt3_mount_panorama_new.json` (TC 4/3, ABS 3, brake bias click 7, ARB 2/4, wing 4, 22 L).

Not changed: `pilot_evolve.py` and `pilot_sim.py` still use the old non-sliding car and a 12 m wall scale, so refining a new browser genome there would undo this. Test new browser exports in ACC directly.

**6 Nov ~19:15 session (Hotlap, pace 0.75, `--manual-gears`, grip limit 2.0, start at 5,720 m from rest):**

| Log | Genome | Distance | Time | Median / max line error | Recovery | Off the measured road (+0.3 m) | ACC invalid from | What ended the run |
|---|---|---|---|---|---|---|---|---|
| `pilot_mltt_20261106_191505` | gen 419 (`--net-steer ramp`) | 312 m (5.1%), out-lap | 11.2 s | 0.95 / 5.01 m | 0 s | 9.0% of frames: left from 6,009 m, up to 2.35 m past | 6,034 m (97.9%) | Hotlap reset in the out-lap |
| `pilot_mltt_20261106_191545` | gen 419 Hotlap, 6.96 m | 187 m (3.0%), out-lap | 7.6 s | 0.22 / 2.75 m | 0 s | 2.6%: right from 5,899 m, up to 0.75 m past | 5,905 m (95.8%) | Hotlap reset in the out-lap |
| `pilot_mltt_20261106_191618` | seeded Hotlap | out-lap + **1 timed lap** (6,611 m) | out-lap 16.2 s, lap **ACC 2:38.902** (own clock 2:38.904), invalid | lap 0.52 / 8.96 m | 2.99 s (3,477–3,522 m) | 3.0%: 4 offs, all at 3,454–3,624 m; worst 6.96 m past on the right at 3,488–3,536 m | 3,494 m (56.7%) | `--laps 1` done |
| `pilot_mltt_20261106_192008` | browser Hotlap | out-lap + 65.5% of lap 1 (4,482 m) | 139.2 s | 0.53 / 10.29 m | 13.3 s (7 spells, 3,348–4,037 m) | 8.7%: right at 2,164 m (1.9 m past), then 6 offs at 3,351–3,800 m, worst 8.29 m past | 3,364 m (54.6%) | Stopped by hand (session paused) |

Pedals set by the speed envelope: 28.4%, 0.0%, 40.9%, 29.1% of frames. Steer target sign flips 3.0%, 1.2%, 1.5%, 0.5% of frames; joystick at full lock 0.0%, 11.8%, 7.1%, 18.9%. Numbers from `python pilot_report.py <log>`.

**`pilot_mltt_20261106_192008.csv` was deleted by mistake** later on 6 Nov, during testing (a cleanup step removed the newest log after a sim test had failed to write one). Its `.json` was rewritten from a full copy printed earlier (marked `restored`). The row above and the notes below were taken from the CSV before then; nothing else from that run can be rechecked.

What happened, from the logs (before changing anything):

- **Run 1, gen 419: predicted, not a sim-vs-live gap.** From ~5,830 m the car drifts left (lat −2.6 → −5.0 m) while the steer target stays within ±0.06 and the heading error stays near 0: it holds a line 3–5 m left of yours into Murray's (what looked like turning in too early). `pilot_check.py --start-s 5720` predicts the first off at 10.3 s; live it was at 10.2 s. Cause: the 12 m training road (see the gen 419 section).
- **Run 2, gen 419 Hotlap 6.96 m: sim-vs-live gap.** From 3.6 s the steer target swings left/right with growing amplitude, ±0.08 → ±0.44 (half-periods 0.41–0.68 s, ~0.9 Hz), at full throttle in 1st–3rd gear, 88–160 km/h; joystick at full lock 11.8% of frames; off right at 5,899 m. ACC's steer followed the joystick within 0–1 frame (correlation 0.90 at zero lag, as in runs 1 and 3), so this is not control-loop delay. `pilot_check.py` from 5,720 m shows no oscillation (max 1.67 m off the line at 3–10 s). Adding car steering lag to `pilot_sim.py` exposes it: from rest at 5,720 m, max distance from the line at 3–10 s:

  | Car steering lag | gen 419 Hotlap 6.96 m | seeded Hotlap | browser Hotlap | gen 419 | classical |
  |---|---|---|---|---|---|
  | 0.01 s (`pilot_check`) | 1.67 m | 1.57 m | 1.30 m | 4.53 m | 0.32 m |
  | 0.05 s | **3.86 m** (target ±0.6) | 1.58 m | 1.30 m | 4.53 m | 0.30 m |
  | 0.10 s | **10.41 m** | 1.65 m | 1.31 m | 4.52 m | 0.27 m |
  | 0.20 s | **10.39 m** | 2.60 m | 1.42 m | 4.53 m | 0.24 m |

  The real car's yaw response is a lag `pilot_sim.py` doesn't have (its heading turns instantly), and this genome has the least margin for it. Probable reason, not proven: its 6.96 m lateral scale makes it 1.72× as sensitive to lateral offset as the 12 m-scale genomes.
- **Run 3, seeded Hotlap: first live network lap; sim-vs-live gap at 3.4 km.** Your laps take 3,400–3,525 m in 1st gear at 100–138 km/h; the pilot also downshifted 2 → 1 at 103 km/h. At 3,415–3,421 m the network asked +0.40 to +0.48 steer (your steer there +0.17 to +0.22) and the joystick saturated. At 3,428–3,440 m it asked +0.67 → +0.99 while the pedals alternated frame to frame between full throttle (network) and 0.12–0.15 brake (envelope), at 5,700–6,600 rpm in 1st. ACC's measured steer then swung from +1.00 to −0.99 with the joystick held at +1.00 (ACC counter-steering a slide), the heading error went from −0.28 to +0.54 rad, and the car crossed from 3.9 m left to 8.96 m right; classical recovery took over at 3,477 m. `pilot_check.py` from 5,720 m put its first off at 103.9 s (live 105.2 s), the same place, but only a small one: its car can't slide. The trainer's sliding car predicted a wall at ~5,400 m, which the car drove through cleanly live, and did not predict this spin. Neither sim has the descent here (no elevation).
- **Run 4, browser Hotlap: same failure, ~80 m earlier.** At 3,265–3,330 m (3rd gear, ~160 km/h, braking 0.08–0.29) it asked +0.25 to +0.41 (your +0.09 to +0.15) with the joystick saturated; ACC's measured steer went from +0.24 to −0.42 with the joystick at +1.00, the heading error rose to +0.29 rad, the network switched to −0.42 to −0.55 and the car spun right: 8.29 m past the edge at 3,351–3,418 m, invalid at 3,364 m. Six more offs and recoveries to 3,800 m (an earlier off at 2,164 m did not invalidate the lap). From ~3,850 m it drove on the line again, within ±0.2 m for the last 3 s at 127–137 km/h. The last 1.9 s of the log are frozen (speed 0.00, steer 0.000, gear −1, heading jumping 151° in one frame at the same spot): ACC's physics page stopping when the session was paused, not a wall. Whether it would have finished the lap is unknown.
- **Classical at 0.9 (`171101`) invalidated at 3,537 m** in the same section, differently: it ran 3.3 m wide right on the exit in 1st at 7,000–8,000 rpm, no slide.

Runs 3 and 4 repeat the 2 Nov 18:25 failures (Murray's, Griffins Bend): in a braking zone the network asks 2–3× your steer, the joystick saturates, the rear steps out and ACC counter-steers. The grip limit (2.0) didn't stop it: it allows 0.44 steer at 158 km/h and 1.00 (full lock) at 90 km/h, against your 0.09–0.22 in that section (1.5 would allow 0.33 / 0.75; 1.2 would allow 0.26 / 0.60). The pedal switching between full throttle and brake in run 3 is `pilot.py`'s governor (network pedal unless over the envelope + 1 m/s), not the network alone.

`pilot_check.py --start-s 5720 --laps 1` (the live start; steer gains saved 6 Nov 19:22, after these runs), x1.0 / x0.8 / x1.25:

| Genome | Lap (s, x1.0) | Median line error | Outside edges | First off | Live (6 Nov) |
|---|---|---|---|---|---|
| gen 419 | 155.4 | 1.34 m | 21.4 / 22.3 / 20.0% | 10.3 s | off at 10.2 s, reset |
| gen 419 Hotlap 6.96 m | 155.8 | 0.38 m | 0.9 / 1.1 / 0.5% | 54.7 s | oscillated from 3.6 s, reset at 7.6 s |
| seeded Hotlap | 155.7 | 0.46 m | 0.9 / 0.9 / 0.9% | 103.9 s | first off at 105.2 s, spun, lap completed (invalid) |
| browser Hotlap | 155.2 | 0.41 m | 3.0 / 3.8 / 2.6% | 53.1 s | off at 74.3 s (2,164 m), spun at 101.7 s |
| classical | 160.0 | 0.11 m | 0.0 / 0.0 / 0.2% | none | |

From this start the x1.25 rows lap (from 150 m before the line, every policy failed at x1.25 at ~35%). The gen 419 tables further up used the steer gains saved 2 Nov 18:29: every live run re-saves `tracks/acc_steer_gain.json`, and the sims read ACC's steering scale from it, so sim laps move by a few tenths between gain files (gen 419 at 0.75: 2:35.89 with the 2 Nov gains, 2:35.37 with the 6 Nov gains). `pilot_check.py` now prints which file it used. A grip limit of 1.5 or 1.2 changes nothing in this check at 0.75 (0.0–0.3% of frames capped, same laps): the sim car never slides, so the networks never escalate there. Its effect can only be judged live.

**From 6 Nov ~20:00 `pilot.py` reports offs live:** when the car goes more than `--edge-margin` (0.3 m) past the measured road edges it prints where, which side and how far, and when it's back on; the status line shows `OFF` and `INVALID`; ACC's `isValidLap` going to 0 prints where. Per lap: offs, % outside, `invalid_at_m`. The log gets a `past_edge_m` column. Driving is unchanged (checked lap for lap against the previous `pilot.py` in the sim). The measured edges are where you drove (98th percentile), not ACC's track limits; ACC's own verdict is `isValidLap`.

Setup runs, not results: `pilot_classical_20261102_130920` (stopped at 4 s), `..._132054` (full throttle at 0 km/h for 8 s: ACC wasn't taking pad input), `..._132133`, `..._132153`, `..._132205` (stopped within 8 s), `..._132325` (drove 350 m, then the car was reset to the start).

`pilot_classical_20261101_172218`, `..._103341`, `..._103408` and `pilot_mltt_20261102_104658` start at s = 10 m (the sim's start point) and have no `tyres_out` values, so they're sim runs, not live.

**`tyres_out` is not usable; `isValidLap` is.** ACC leaves `numberOfTyresOut` at 0 in every human and pilot log, including the gen 147 run that ended 22 m from the line. "No tyres off" is therefore not a measured result. From 2 Nov ~13:30, `pilot.py` records ACC's `isValidLap` per timed lap, which is ACC's own track-limits verdict. Line error against the measured road edges is the other check: the classical full lap never left them (closest 0.42 m inside). The two seeded runs were outside them 12% and 23% of the time.

---

## Training runs

| Run | Sim road | Seed | Result | Log |
|---|---|---|---|---|
| From scratch | flat 7.5 m / 12 m wall, no scrub | random | First lap gen 16, 1:48.47 at gen 285. Exploits the sim: 9% of the lap past 7.5 m, 15% at the grip clamp (these two are not in the log: its CSV predates the off/clamp columns, so they can't be rechecked from it). | `evolve_mount_panorama.evolved.csv` |
| Seeded | flat 7.5 m / 12 m wall, no scrub | gen 147 browser genome | First lap gen 2, 1:58.93 at gen 59 | `evolve_mount_panorama.seeded.csv` |
| Seeded, real edges (2 Nov) | measured edges + 0.5 m, wall + 3 m, scrub 5 m/s² | `.seeded.json` | 300 gens. Laps from gen 0. Best 1:49.4 (gen 292). Best car off-road 10.1% at gen 50, 1.9% at gen 299, and 17.2% at the grip limit (16.7–18.0% from gen 50 on; was given as 18%). Saved `.edges_seeded.json`. | `evolve_mount_panorama.edges_seeded.csv`, `evolve_edges_seeded_run.txt` |

Under `--flat-edges` the new code reproduces the seeded genome's 118.9 s, so the old results still hold. With the measured edges, the seeded genome starts at 129–136 s and 3–4% off-road.

**Sim lap times are not comparable to human laps.** 1:49.4 is ~12 s under your 2:01.3. The sim has no elevation (the Mountain climb), and the car gets the 99th-percentile grip and 90th-percentile acceleration of your laps everywhere. The GA still finds the grip limit worth riding even with scrub. Judge genomes on line error and live runs, not sim time.

| Browser gen 19, refined (2 Nov) | measured edges + 0.5 m, wall + 3 m, scrub 5 m/s² | `mount_panorama.mltt_gen19.json`: ML Track Trainer, 30 generations with the 2-lap episode on the circuit imported from 10 new clean laps; the exported genome is the best-reward one, from gen 19. The run's fastest laps were 2:01.4 standing and 1:58.5 flying (browser sim), but the export's `lap_s` / `flying_lap_s` are the fastest by any driver in the run, so they aren't necessarily this genome's | Gen 0 on the real line: 32% of a lap (browser → real line transfer is weak). First lap gen 10, best 1:48.8 at gen 97, 5.5% off-road. Saved `.browser2.json` | `evolve_mount_panorama.browser2.csv`, `evolve_browser2_run.txt` |

**Sim comparison 2 Nov (`--acc-steer`, `--laps 2`, start 150 m before the line, timed laps only):**

| Pace | Policy | Laps | Median / max line error | Recovery | Outside measured edges |
|---|---|---|---|---|---|
| 0.75 | Classical | 2:39.1, 2:39.0 | 0.17 / 3.80 m | 0 s | 0.4% |
| 0.75 | Seeded, real edges | 2:34.4, 2:34.3 | 0.69 / 4.46 m | 0 s | 11.4% |
| 0.75 | Browser gen 19, raw | stuck at the start | | | |
| 0.75 | Browser gen 19, refined | 2:33.6, 2:33.6 | 0.40 / 5.26 m | 0 s | 12.2% |
| 0.85 | Classical | 2:22.6, 2:20.7 | 0.22 / 8.86 m | 1.2 s | 3.9% |
| 0.85 | Seeded, real edges | 2:16.4, 2:16.3 | 0.79 / 4.56 m | 0 s | 8.3% |
| 0.85 | Browser gen 19, refined | 2:15.8, 2:15.8 | 0.41 / 5.23 m | 0 s | 15.0% |
| 0.9 | Classical | 2:16.8, 2:15.4 | 0.26 / 7.86 m | 5.3 s | 6.7% |
| 0.9 | Seeded, real edges | 2:10.5, 2:10.3 | 0.91 / 7.16 m | 3.8 s | 9.2% |
| 0.9 | Browser gen 19, refined | 2:08.8, 2:08.8 | 0.43 / 5.93 m | 0 s | 17.2% |

The refined browser genome is the fastest network and the most consistent (identical lap pairs, no recovery even at 0.9), with half the seeded genome's median line error. But it leaves the measured road more (12–17%), so it's still cutting corners. The raw browser export can't drive the real line at all. Live test pending; every network weaved live on 2 Nov (see Live results).

**Hotlap retrains (2 Nov ~18:00, new 12-lap line).** `pilot_evolve.py --edge-margin 0.3 --wall-margin 0 --steer-noise 0.25`: leaving the measured road (+0.3 m) ends the run, like Hotlap; each car gets a random steering gain mismatch (x0.8–1.25) and bias. Reason: live at 17:26 both networks steered smoothly after the weave fix but drifted 2.2–3.2 m right on the run to Murray's (right edge 2.0 m there) while steering almost straight, and were reset.

| Run | Seed | Result | Log |
|---|---|---|---|
| Browser, Hotlap | `.browser2.json` | Gen 0: 54% of a lap. First lap ~gen 50. Best 1:51.4 (gen 287). Saved `.browser2_hotlap.json` | `evolve_mount_panorama.browser2_hotlap.csv` |
| Seeded, Hotlap | `.edges_seeded.json` | Gen 0: 24%. First lap gen 25. Best 1:51.4 (gen 281). Saved `.edges_seeded_hotlap.json` | `evolve_mount_panorama.edges_seeded_hotlap.csv` |

Live-like sim check (0.01 s steering response, 1-frame delay, ±0.05 noise on measured steer; steer mismatch x1.0 / x0.8 / x1.25; 2 laps; "outside" = beyond measured edge + 0.3 m in timed laps):

| Pace | Policy | Laps (s) | Median line error | Outside edges (x1.0 / x0.8 / x1.25) | First off |
|---|---|---|---|---|---|
| 0.75 | Classical | 160.2, 160.0 | 0.20 m | 1.4 / 0.7 / 4.0% | 140 / 143 / 62 s |
| 0.75 | Browser, Hotlap | 155.2, 155.2 | 0.41 m | 3.1 / 3.6 / 3.7% | 44 s |
| 0.75 | Seeded, Hotlap | 155.7, 155.7 | 0.45 m | 0.9 / 0.9 / 1.9% | 95 / 95 / 61 s |
| 0.75 | Browser (before) | 154.6, 154.5 | 0.39 m | 9.1 / 10.5 / 6.1% | 44 s |
| 0.9 | Classical | 134.3, 134.4 | 0.34 m | 4.6 / 2.1 / 13.7% | 38 / 50 / 9 s |
| 0.9 | Browser, Hotlap | 129.6, 129.6 | 0.34 m | 2.0 / 2.1 / 8.0% | 77 / 79 / 51 s |
| 0.9 | Seeded, Hotlap | 129.8, 129.7 | 0.36 m | 1.0 / 1.0 / 4.8% | 80 / 80 / 51 s |
| 0.9 | Browser (before) | 129.2, 129.1 | 0.39 m | 9.7 / 11.1 / 10.1% | 38 s |

Hotlap retraining cut time off the measured road from ~10% to 1–4% at the same pace, and both new genomes clear the out-lap (first off ≥ 44 s; the out-lap is ~15 s). At 0.9 both are faster than classical and leave the road less. The seeded Hotlap genome is the most robust.

**Hotlap reset rule, from the logs:** the networks were reset the instant they invalidated during the *out-lap*; classical invalidated lap 1 at ~60% (`171101`) and kept driving. So a run has to get through the out-lap clean; after that an off only invalidates the lap. (6 Nov: the per-frame `acc_valid` in `171101` goes to 0 at 3,535 m, 57% of the lap; in `153930` lap 1 goes invalid at 1,560 m, 25%.)

### ML Track Trainer gen 419: rebuilt sliding car (6 Nov)

**The genome.** `tracks/mount_panorama.mltt_gen419.json`: Export best model from ML Track Trainer `acc-track-import` (commit `c6d4c1c`, the sliding car). Reward 27966.97, gen 419, `wall_m` 6.96, car `fitted`. The 4 Nov and 6 Nov exports hold identical weights; only `flying_lap_s` differs (124.60 → 124.70).

**Its own times, re-run headless in the trainer's own `env.js`** (Node, same track import and car fit). The 2 Nov session `acc_session_20261102_161217` (7 flying laps) reproduces its reward: 27966.971 vs 27966.9709 exported, so that is the track and car it trained on. Two-lap episode from rest:

| | This genome | Export field |
|---|---|---|
| Standing lap | **2:16.9** (136.90 s) | `lap_s` 136.53 (2:16.5): another driver in the run, not this genome |
| Flying lap | **2:04.7** (124.70 s) | `flying_lap_s` 124.70: matches |
| Off its 12 m road | 0 s, no wall hits | |
| Max body slip | 21.5° | |
| Steer above the grip steer (1×) | 10.1% of frames | |

These are trainer sim times (no elevation, its own 12 m road, its own line), not comparable to your 2:01.3–2:03.6 laps. On the 1 Nov session's track and car fit (steer full lock 0.203 rad instead of 0.258) it hits the wall at 87 s, so it is sensitive to the car's steering scale.

**Input layout.** Checked against `env.js observe()` and `brain.js forward()`: the same 9 inputs (speed/90, lateral/wall, heading error, look-ahead at 15/40/80/140/220 m over π/2, own steer), the same signs and weight layout as `pilot.py`. The lateral scale (`wall_m` 6.96) comes from the export. Two differences: the trainer's steer is rate-limited (`rampSteer`: 3/s away from centre, 5/s back) where `pilot.py` used a 0.10 s lag, and the trainer ramps the pedals. Added `pilot.py --net-steer ramp` (default `lag`). It changed nothing measurable: same lap times within 0.1 s and the same line error in every check below.

**Transfer to the real line (`pilot.py`).**
- `pilot.py --sim --acc-steer` (0.10 s car steering, the old default): crashed at 0.75 (64.7 s). At 0.9: 2:42.6 / 2:37.9, median 2.29 / 2.18 m, 14.4 s recovering.
- Live-like check (`pilot_check.py`; see below), `--net-steer lag` and `ramp` the same to 0.1 s:

| Pace | Mismatch | Laps (s) | Median / max line error | Outside measured edges + 0.3 m | First off |
|---|---|---|---|---|---|
| 0.75 | x1.0 / x0.8 | 155.7, 155.7 / 155.4, 155.4 | 1.33–1.34 / 5.4 m | 22.1–22.2% / 21.9% | 2.5–2.7 s |
| 0.75 | x1.25 | crash at 35.5% | | 14.1% | 2.9 s |
| 0.85 | x1.0 / x0.8 / x1.25 | 140.5 / 140.7 / 142.6–142.7 | 1.34–1.44 m | 22.1–23.8% | 2.5–2.9 s |
| 0.9 | x1.0 / x0.8 / x1.25 | 135.3 / 135.7 / 135.0 | 1.36–1.46 m | 21.9–24.2% | 2.5–2.9 s |

It reads the inputs and drives the real line (laps at every pace with no recovery at x1.0 / x0.8, unlike the raw browser exports before it), but at 3× the Hotlap genomes' line error and off the measured road a fifth of the time, first within 3 s. In Hotlap that is a reset during the out-lap.

**Cause: the training road, not a pilot.py or sim-vs-live gap.** Its own trainer drive, mapped back to ACC world coordinates, is outside the measured edges + 0.3 m on **17.6%** of the standing lap and **19.5%** of the flying lap (left side 13–15%, right 4–5%). When outside, it is a median 1.26 m past the edge (95th percentile 3.56 m, max 4.84 m). The live-like check's 22% is the same behaviour. The trainer's road is 12 m wide, symmetric about its own line; the measured road is 4.0–5.2 m wide (98th percentile of where you drove). Worst 100 m stretches in its own drive: 1,800–1,900 m (100% outside), 2,500 m (94%), 2,900 m (88%), 3,400–3,800 m (56–67%), all walled sections in ACC. The trainer's line is also a median 0.83 m from `pilot.py`'s line (95th percentile 1.82 m, max 3.15 m): it rounds its 120 control points to whole pixels at 3.07 m per pixel.

**Retrained Hotlap-style** (`pilot_evolve.py --seed-from tracks/mount_panorama.mltt_gen419.json --edge-margin 0.3 --wall-margin 0 --steer-noise 0.25 --gens 300`). `pilot_evolve.py` scales the lateral input by 12 m, and gen 419 trained on 6.96 m, so the run was also done with the new `--seed-wall` (keeps the seed's 6.96 m; off by default):

| Run | Lateral scale | Gen 0 | First lap | Saved | At the grip limit (saved gen) | Log |
|---|---|---|---|---|---|---|
| `.gen419_hotlap.json` | 12 m | 27.1% of a lap | gen 6 | gen 274, 1:53.4 | 13.5% | `evolve_mount_panorama.gen419_hotlap.csv`, `evolve_gen419_hotlap_run.txt` |
| `.gen419_hotlap_w7.json` | 6.96 m (`--seed-wall`) | 30.2% | gen 14 | gen 281, 1:53.7 | 15.2% | `evolve_mount_panorama.gen419_hotlap_w7.csv`, `evolve_gen419_hotlap_w7_run.txt` |

Live-like check, current line, x1.0 / x0.8 / x1.25:

| Pace | Policy | Laps (s, x1.0) | Median line error | Outside edges (x1.0 / x0.8 / x1.25) | First off | Pedals set by the envelope (x1.0) |
|---|---|---|---|---|---|---|
| 0.75 | gen 419 | 155.7, 155.7 | 1.34 m | 22.2 / 21.9 / crash 35.5% | 2.7 s | 42.6% |
| 0.75 | gen 419 Hotlap, 6.96 m | 156.1, 156.0 | 0.39 m | 2.7 / 1.1 / crash 35.5% | 46.6 s | 55.0% |
| 0.75 | gen 419 Hotlap, 12 m | 157.7, 157.8 | 1.05 m | 6.3 / 2.0 / crash 35.4% | 59.8 s | |
| 0.75 | seeded Hotlap | 157.5, 157.5 | 0.48 m | 2.0 / 0.9 / stuck 35.0% | 60.9 s | 53.9% |
| 0.75 | browser Hotlap | 155.4, 155.4 | 0.40 m | 4.6 / 3.8 / stuck 65.5% | 44.4 s | 55.8% |
| 0.75 | classical | 161.0, 161.2 | 0.10 m | 3.2 / 0.0 / crash 35.4% | 61.5 s | |
| 0.85 | gen 419 Hotlap, 6.96 m | 137.8, 137.8 | 0.44 m | 1.0 / 0.6 / crash 150.5% | 42.0 s | |
| 0.9 | gen 419 Hotlap, 6.96 m | 130.2, 130.2 | 0.47 m | 0.3 / 0.4 / 14.5% (laps, 12.4 s recovering) | 89.1 s | |
| 0.9 | seeded Hotlap | 129.7, 129.7 | 0.39 m | 1.0 / 1.0 / 7.8% | 79.9 s | |
| 0.9 | browser Hotlap | 129.5, 129.6 | 0.35 m | 2.1 / 2.1 / 4.4% | 79.3 s | |
| 0.9 | classical | 133.8, 133.8 | 0.14 m | 0.0 / 0.0 / 8.0% (laps, 10.8 s recovering) | 52.4 s (x1.25) | |

At 0.75 the speed envelope sets the pedals on 43–56% of the timed-lap frames for every network: there a "network lap" is network steering with the pedals shared with the governor. Report that with any live lap.

**Sliding-car check with `pilot.py`'s governor.** `pilot_check.py`'s car can't slide, and the 2 Nov live failures were slides. So each genome was also driven on the trainer's sliding car (Node, same import as above): `pilot.py`'s speed envelope (network pedal unless over pace × your trace + 1 m/s), grip limit 2.0, launch at ≥ 0.7 throttle to 30 km/h, the genome's own lateral scale and steer model, two laps from rest at the line. Not modelled: `pilot.py`'s classical recovery and SteerComp. Scored against the real line and measured edges:

| Pace | Genome | Result | Laps (standing, flying) | Median line error | Outside edges + 0.3 m | Max slip |
|---|---|---|---|---|---|---|
| 0.75 | gen 419 (ramp) | **2 laps** | 166.9, 156.7 | 1.34 m | 20.4% | 2° |
| 0.75 | gen 419 Hotlap, 6.96 m | **2 laps** | 166.2, 156.9 | 0.87 m | 9.2% (first off 14.1 s) | 2° |
| 0.75 | seeded Hotlap | wall at 88% (144.5 s) | | 0.90 m | 1.9% | 51° |
| 0.75 | browser Hotlap | stalled at 88% (153.7 s) | | 1.13 m | 13.3% | 78° |
| 0.75 | gen 419 Hotlap, 12 m | doesn't pull away: asks for 0.75 steer at rest, which the trainer's tyre model won't roll with (an artefact; it launches in `pilot_sim.py`) | | | | |
| 0.85 | gen 419 | **2 laps** | 151.6, 140.5 | 1.32 m | 20.5% | 15° |
| 0.85 | gen 419 Hotlap, 6.96 m | wall at 88% (130.1 s) | | 0.83 m | 9.1% | 77° |
| 0.85 | seeded Hotlap | stalled at 88% | | 0.94 m | 5.4% | 81° |
| 0.85 | browser Hotlap | stalled at 24% | | 1.56 m | 15.3% | 78° |
| 0.9 | gen 419 | wall at 63% (100 s) | | 1.38 m | 23.5% | 59° |
| 0.9 | gen 419 Hotlap, 6.96 m | wall at 33% (54.6 s) | | 0.91 m | 12.8% | 31° |
| 0.9 | seeded Hotlap | wall at 88% (122.9 s) | | 0.87 m | 1.5% | 83° |
| 0.9 | browser Hotlap | wall at 24% (42.8 s) | | 1.48 m | 6.4% | 45° |

Ungoverned (each genome on its own pedal, as `pilot_evolve.py` trains them), every Hotlap retrain, the two 2 Nov ones included, hits the wall within 21 s on the sliding car (slip up to 81°); only gen 419 laps. So the Hotlap retrain fixed the road edges and cost the sliding-car stability gen 419 was trained for, which the governor at 0.75 hides. (This trainer car gave a pure-pursuit driver at 0.9 a 2:18.0 flying lap against classical's 2:17.344 real-clock lap live, ACC-invalid. It has never been checked against a live network run.)

**Ranking for the live tests:** at 0.75 only gen 419 and its 6.96 m Hotlap retrain lap in both sims. The retrain stays on the measured road (2.7% / 9.2% outside in the two sims, first off ≥ 14 s) where gen 419 doesn't (20–22%, first off at 2.7 s in `pilot_check`). Neither sim models elevation, kerbs or ACC's real walls. At 0.9 every network crashes on the sliding car.

**`pilot_check.py` vs the 2 Nov live-like table.** The 2 Nov check was done with a script that isn't in the repo. `pilot_check.py` re-implements it from the description, with SteerComp starting from the learned gains as live (from neutral, the joystick saturates out of the first corner and every policy leaves the road within 5 s). It reproduces the seeded Hotlap rows at x1.0 / x0.8 (0.9: 129.7 s, 0.39 m, 1.0%, first off 80 s, vs 129.8 s, 0.36 m, 1.0%, 80 s), but not the x1.25 rows: at 0.75 every policy, classical included, crashes or sticks at ~35% of the lap at x1.25, where the 2 Nov table has them lapping. Compare policies within one harness; the tables above all use `pilot_check.py` (seed 0).

### Policy sim comparison (current line, `pilot.py --sim`)

| Policy | 75% pace | 85% | 95% | Max line error |
|---|---|---|---|---|
| Classical | 2:41.3 | 2:23.5 | 2:09.5 | 1.1–1.6 m |
| Evolved from scratch | 2:40.8 | 2:23.8 | 2:10.5 | 8.3 m, ~9 s recovering |
| Seeded | 2:36.8 | 2:19.6 | 2:08.9 | 5.4–5.7 m |

Rerun 2 Nov with `--acc-steer` (ACC's speed-scaled steering + SteerComp, learning from neutral), adding line error against the measured road edges:

| Pace | Policy | Lap | Median / max line error | Recovery | Outside measured edges |
|---|---|---|---|---|---|
| 0.75 | Classical | 2:41.5 | 0.19 / 3.91 m | 0 s | 0.8% |
| 0.75 | Seeded | 2:36.3 | 1.39 / 5.78 m | 0 s | 16.2% |
| 0.75 | Seeded, real edges | 2:36.9 | 0.71 / 4.53 m | 0 s | 11.8% |
| 0.85 | Classical | 2:23.9 | 0.27 / 4.94 m | 0 s | 2.2% |
| 0.85 | Seeded | 2:19.3 | 1.86 / 5.36 m | 0 s | 32.6% |
| 0.85 | Seeded, real edges | 2:19.3 | 0.79 / 4.57 m | 0 s | 6.8% |
| 0.95 | Classical | 2:13.1 | 0.50 / 8.27 m | 2.3 s | 12.5% |
| 0.95 | Seeded | 2:09.4 | 2.07 / 6.33 m | 0.4 s | 37.2% |
| 0.95 | Seeded, real edges | 2:07.5 | 1.00 / 7.57 m | 1.9 s | 10.3% |

Retraining on real edges halved the seeded network's median line error at the same lap time, and cut its time off the measured road from 32.6% to 6.8% at 0.85. Classical is still the tightest at 0.75–0.85. At 0.95 the retrained network is fastest and leaves the road slightly less than classical. The old seeded genome's 16% outside at 0.75 matches what it did live (12–23%), so this sim check predicts live behaviour reasonably well.

---

## Sim vs live gaps

| Gap | Effect live | Fix | What changed |
|---|---|---|---|
| ACC scales gamepad steer with speed. Gain ≈ 0.5 at 36–72 km/h, 0.21–0.24 at 144–180, 0.11–0.22 at 180–216. Rechecked 6 Nov (median of ACC steer ÷ last frame's joystick, joystick 0.05–0.95, not recovering) in the two pre-SteerComp logs `184553` / `103148`: 0.88 / 0.95 at 0–36 km/h, 0.54 / 0.54 at 36–72, 0.36 / 0.41 at 72–108, 0.23 / 0.27 at 108–144, 0.17 / 0.20 at 144–180, 0.14 / 0.15 at 180–216. That method gives lower values at 144–180 than the 0.21–0.24 above. | Car ran wide in every fast corner. Classical reached 22% and 43%. | SteerComp (learned per-band gain), `pilot_sim.py --acc-steer` to reproduce it | Measured steer within 1–6% of requested below 180 km/h, 10% above. First full classical lap. |
| GA sim road: flat 7.5 m half-width, free grip clamp | From-scratch genome laps 1:48.5 in sim (faster than any human lap) by cutting corners and riding the grip limit | Measured per-point edges + scrub (2 Nov) | Corner cutting mostly gone (seeded at 0.85: 32.6% → 6.8% off the measured road). Grip-limit riding remains (18%). Live test pending. |
| No elevation, kerbs, weight transfer or oversteer in the sim | Unknown. Possibly the 22% spin on the climb to The Cutting. | none yet | |
| Seeded network leaves the measured road 12–23% of the time live | Ends in recovery around 55% | Retrain on real edges | _pending_ |
| ML Track Trainer road: 12 m wide, symmetric about a line rounded to 3.07 m pixels; the measured road is 4.0–5.2 m | gen 419 is off the measured road 18–20% of its own trainer laps and 22% in the live-like check (6 Nov) | Hotlap retrain in `pilot_evolve.py` with `--seed-wall` | 2.7% off at 0.75 in the live-like check, but the retrain spins on the trainer's sliding car without the governor. Live test pending. Better fix: train in the trainer on the measured edges |
| `pilot_evolve.py` / `pilot_sim.py` car can't slide | Retraining there loses sliding-car stability (every Hotlap retrain hits the wall within 21 s on the trainer car, ungoverned) | Governor at 0.75; sliding-car check before live runs | 6 Nov live: seeded and browser Hotlap spun at 3.3–3.5 km after asking 2–3× your steer; ACC counter-steered. The sliding-car check didn't predict it (it predicted a wall at ~5.4 km instead) |
| `pilot_sim.py`'s heading turns instantly (no yaw response) | gen 419 Hotlap 6.96 m oscillated at ~0.9 Hz from 3.6 s live and was reset (6 Nov) | none yet; adding 0.05 s of car steering lag to the sim reproduces it | _pending_ |

---

## Next tests

0. **Live network runs (planned 6 Nov).** Each at pace 0.75, `--manual-gears` (ACC auto gearbox off, auto-clutch on), grip limit 2.0 (default; no network has run live with it yet), `--laps 1`. Stop at the first failure and read the log before the next run. Predictions are from the two checks above:

| # | Genome | Command | Prediction |
|---|---|---|---|
| 1 | gen 419 | `python pilot.py --policy mltt --weights tracks/mount_panorama.mltt_gen419.json --net-steer ramp --pace 0.75 --manual-gears` | Laps in both sims, but off the measured road 20–22% of the time, first within 3 s: a Hotlap reset in the out-lap |
| 2 | gen 419 Hotlap, 6.96 m | `python pilot.py --policy mltt --weights tracks/mount_panorama.gen419_hotlap_w7.json --pace 0.75 --manual-gears` | Main candidate. Laps in both sims, 2.7% / 9.2% off the measured road, median 0.39 m in the live-like check |
| 3 | seeded Hotlap | `python pilot.py --policy mltt --weights tracks/mount_panorama.edges_seeded_hotlap.json --pace 0.75 --manual-gears` | Laps in the live-like check (2.0% off), hits the wall at 88% on the sliding car. Ran live once at 0.9 without the grip limit: reset at 11.4 s (`182532`) |
| 4 | browser Hotlap | `python pilot.py --policy mltt --weights tracks/mount_panorama.browser2_hotlap.json --pace 0.75 --manual-gears` | Laps in the live-like check (4.6% off), stalls at 88% on the sliding car. Ran live once at 0.9 without the grip limit: 33.7% of distance (`182630`) |

   Session: Hotlap resets the car at the first off during the out-lap, which ends run 1 within seconds if the prediction holds. A Practice session lets a run carry on past an off, with ACC marking the lap invalid (`isValidLap`, recorded per lap). A lap completed that way is a completed lap, not a valid one, and is reported as such.

   Whichever completes a lap: 0.8, then 0.85. Not 0.9 yet: every network hits the wall at 0.9 on the sliding car. A lap with recovery time > 0 had classical steering for those seconds; report it.

   **Done 6 Nov ~19:15** (see Live results): run 1 reset at 10.2 s as predicted; run 2 reset at 7.6 s (oscillation the sim didn't show); **run 3 completed the first live network lap, ACC 2:38.902, invalid, 3.0 s recovering**; run 4 spun at the same section and was stopped at 65.5%.

   Next (planned), all seeded Hotlap at 0.75, Hotlap mode, `--manual-gears`:
   - Repeat run 3 with `--laps 2`: is the lap repeatable, and does it go off at 3.4 km again?
   - Same with `--grip-limit 1.2` (caps the network at 0.60 steer at 90 km/h, 0.26 at 158 km/h; costs nothing in the sim at 0.75). Judged on whether the 3.3–3.5 km section stays clean.
   - The governor's pedal switching (full throttle / brake frame to frame in run 3) could be smoothed with a new default-off option; not done yet.

1. Live: `.edges_seeded.json` vs classical at the same pace (0.75), with SteerComp. Sim prediction: both lap, network median ~0.7 m vs classical ~0.2 m.
2. Pace ladder 0.8 → 0.85 → 0.9 → 0.95. Record which policy completes each step. Classical: 0.75 ✓, 0.9 ✓ (and 2 consecutive timed laps, ACC 2:19.852 for the first; lap 1 invalid). Next 0.95 (sim: 2:13.1, 12.5% outside the road, 2.3 s recovering, so expect trouble).
3. Rolling race starts: pace-car speed cap, wait for green, speed limit from the gap to the car ahead.
