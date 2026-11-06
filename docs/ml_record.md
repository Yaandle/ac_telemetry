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
| **Trained from** | not trained: geometry. Line = median of 5 flying laps, 1 Nov session (2:01.3–2:02.2), `tracks/mount_panorama.json` |
| **Steering** | `SteerComp` with gains from `tracks/acc_steer_gain.json` |
| **Best timed laps** | 2 consecutive laps at pace 0.9, lap 1 ACC 2:19.852 (invalid). `pilot_classical_20261102_132857.csv` |
| **Best live result** | full lap at pace 0.9: 6.17 km in 134.4 s by frame count (≈ 140 s real; see the timing note below), starting 445 m before the line (not an ACC-timed lap). Median 0.29 m from the line, max 5.38 m, 0 s recovering, 2.0% outside the measured road. `pilot_classical_20261102_130945.csv` |
| **Earlier full lap** | pace 0.75: 157.5 s by frame count (≈ 163 s real), median 0.32 m, max 3.07 m, never outside the measured road. `pilot_classical_20261102_105851.csv` |
| **Best network live** | `tracks/mount_panorama.seeded.json`: 55% of a lap, twice (`pilot_mltt_20261102_105030.csv`, `..._105613.csv`) |

No network has finished a live lap yet.

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

Line accuracy: distance along the line matches ACC's spline position with median 0.15 m error, 95% within 0.33 m (`track_line.py`'s line, not the dashboard export).

reader.py logs 66 columns at 100 Hz (median sample gap 0.010 s). Two are `timestamp` and the `lap_time_str` text.

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
| `pilot_mltt_20261102_105030` | seeded | 54.9% | 93.7 s | 1.20 / 7.57 m | 7.9 s | |
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

Setup runs, not results: `pilot_classical_20261102_130920` (stopped at 4 s), `..._132054` (full throttle at 0 km/h for 8 s: ACC wasn't taking pad input), `..._132133`, `..._132153`, `..._132205` (stopped within 8 s), `..._132325` (drove 350 m, then the car was reset to the start).

`pilot_classical_20261101_172218`, `..._103341`, `..._103408` and `pilot_mltt_20261102_104658` start at s = 10 m (the sim's start point) and have no `tyres_out` values, so they're sim runs, not live.

**`tyres_out` is not usable; `isValidLap` is.** ACC leaves `numberOfTyresOut` at 0 in every human and pilot log, including the gen 147 run that ended 22 m from the line. "No tyres off" is therefore not a measured result. From 2 Nov ~13:30, `pilot.py` records ACC's `isValidLap` per timed lap, which is ACC's own track-limits verdict. Line error against the measured road edges is the other check: the classical full lap never left them (closest 0.42 m inside). The two seeded runs were outside them 12% and 23% of the time.

---

## Training runs

| Run | Sim road | Seed | Result | Log |
|---|---|---|---|---|
| From scratch | flat 7.5 m / 12 m wall, no scrub | random | First lap gen 16, 1:48.47 at gen 285. Exploits the sim: 9% of the lap past 7.5 m, 15% at the grip clamp. | `evolve_mount_panorama.evolved.csv` |
| Seeded | flat 7.5 m / 12 m wall, no scrub | gen 147 browser genome | First lap gen 2, 1:58.93 at gen 59 | `evolve_mount_panorama.seeded.csv` |
| Seeded, real edges (2 Nov) | measured edges + 0.5 m, wall + 3 m, scrub 5 m/s² | `.seeded.json` | 300 gens. Laps from gen 0. Best 1:49.4 (gen 292). Best car off-road 10.1% at gen 50, 1.9% at gen 299, but 18% at the grip limit. Saved `.edges_seeded.json`. | `evolve_mount_panorama.edges_seeded.csv`, `evolve_edges_seeded_run.txt` |

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

**Hotlap reset rule, from the logs:** the networks were reset the instant they invalidated during the *out-lap*; classical invalidated lap 1 at ~60% (`171101`) and kept driving. So a run has to get through the out-lap clean; after that an off only invalidates the lap.

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
| ACC scales gamepad steer with speed. Gain ≈ 0.5 at 36–72 km/h, 0.21–0.24 at 144–180, 0.11–0.22 at 180–216. | Car ran wide in every fast corner. Classical reached 22% and 43%. | SteerComp (learned per-band gain), `pilot_sim.py --acc-steer` to reproduce it | Measured steer within 1–6% of requested below 180 km/h, 10% above. First full classical lap. |
| GA sim road: flat 7.5 m half-width, free grip clamp | From-scratch genome laps 1:48.5 in sim (faster than any human lap) by cutting corners and riding the grip limit | Measured per-point edges + scrub (2 Nov) | Corner cutting mostly gone (seeded at 0.85: 32.6% → 6.8% off the measured road). Grip-limit riding remains (18%). Live test pending. |
| No elevation, kerbs, weight transfer or oversteer in the sim | Unknown. Possibly the 22% spin on the climb to The Cutting. | none yet | |
| Seeded network leaves the measured road 12–23% of the time live | Ends in recovery around 55% | Retrain on real edges | _pending_ |

---

## Next tests

1. Live: `.edges_seeded.json` vs classical at the same pace (0.75), with SteerComp. Sim prediction: both lap, network median ~0.7 m vs classical ~0.2 m.
2. Pace ladder 0.8 → 0.85 → 0.9 → 0.95. Record which policy completes each step. Classical: 0.75 ✓, 0.9 ✓ (and 2 consecutive timed laps, ACC 2:19.852 for the first; lap 1 invalid). Next 0.95 (sim: 2:13.1, 12.5% outside the road, 2.3 s recovering, so expect trouble).
3. Rolling race starts: pace-car speed cap, wait for green, speed limit from the gap to the car ahead.
