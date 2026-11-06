#!/usr/bin/env python3
"""
pilot_check.py: the live-like sim check from docs/ml_record.md, as one command.

Drives pilot.py's Pilot in pilot_sim.py with ACC's live steering behaviour:
  - steering answers in 0.01 s (ACC's gamepad), not the sim's default 0.10 s
  - the pilot sees state 1 frame old
  - +-0.05 uniform noise on the measured steer
  - ACC's speed-scaled gamepad steering (tracks/acc_steer_gain.json), and SteerComp starting from
    those learned gains, as it does live (from neutral, the joystick saturates out of the first corner)
  - the car needs x1.0 / x0.8 / x1.25 the fitted steer (--mismatch)

and scores each run against the measured road edges (<track>.edges.json):
  outside   % of timed-lap frames beyond the measured edge + --edge-margin (0.3 m, as Hotlap retraining)
  first off seconds from the start of the run (out-lap included) to the first such frame

  python pilot_check.py --policy mltt --weights tracks/mount_panorama.edges_seeded_hotlap.json --pace 0.75
  python pilot_check.py --policy classical --pace 0.9
"""

import argparse
import csv
import json
import tempfile
from pathlib import Path

import numpy as np

import pilot
from pilot_sim import SimBackend
from track_line import TrackLine


class LiveLikeSim(SimBackend):
    def __init__(self, *a, steer_noise=0.05, seed=0, **kw):
        super().__init__(*a, **kw)
        self.noise = steer_noise
        self.rng = np.random.default_rng(seed)

    def read(self):
        st = dict(super().read())
        st["steer"] += self.rng.uniform(-self.noise, self.noise)
        return st


def check(track, edges, policy_name, weights, pace, mismatch, laps=2, net_steer="lag", grip_limit=2.0,
          edge_margin=0.3, steer_noise=0.05, seed=0, log_path=None):
    track.idx = None
    policy = (pilot.MLTrackTrainer(track, weights, grip_limit=grip_limit, steer_model=net_steer)
              if policy_name == "mltt" else pilot.Classical(track))
    p = pilot.Pilot(track, policy, pace=pace, steer_comp=pilot.SteerComp.load())
    acc = pilot.SteerComp.load()
    backend = LiveLikeSim(track, start_s=track.lap_m - 150, steer_mismatch=mismatch, steer_lag=0.01,
                          delay_frames=1, steer_scale=(acc.V, acc.gain), steer_noise=steer_noise, seed=seed)
    with tempfile.TemporaryDirectory() as d:
        tmp = log_path or Path(d) / "check.csv"
        res = pilot.run(p, backend, laps=laps, log_path=str(tmp), realtime=False, quiet=True)
        with open(tmp, newline="") as f:
            rows = list(csv.DictReader(f))
    t = np.array([float(r["t"]) for r in rows])
    s = np.array([float(r["s"]) for r in rows])
    lat = np.array([float(r["lat"]) for r in rows])
    lap = np.array([int(r["lap"]) for r in rows])
    i = np.round(s / track.grid).astype(int) % len(edges["right_m"])
    out = (lat > np.array(edges["right_m"])[i] + edge_margin) | (-lat > np.array(edges["left_m"])[i] + edge_margin)
    timed = lap > 0
    over = getattr(policy, "capped", 0)
    # Frames where the speed envelope, not the network, set the pedals (pilot.Pilot.step: v > v_target + 1)
    gov = np.array([float(r["speed"]) > float(r["v_target"]) + 1.0 and r["lost"] == "0" for r in rows])
    return {
        "result": res["result"], "progress": round(res["progress"], 3), "time_s": round(res["time_s"], 1),
        "laps": [l["time_s"] for l in res["laps"]],
        "median_lat_m": round(float(np.median(np.abs(lat[timed]))), 2) if timed.any() else None,
        "max_lat_m": round(float(np.abs(lat[timed]).max()), 2) if timed.any() else None,
        "recover_s": round(res["recover_s"], 1),
        "outside_pct": round(100 * float(out[timed].mean()), 1) if timed.any() else None,
        "first_off_s": round(float(t[out][0]), 1) if out.any() else None,
        "grip_capped_pct": round(100 * over / len(rows), 1) if policy_name == "mltt" else None,
        "governed_pct": round(100 * float(gov[timed].mean()), 1) if policy_name == "mltt" and timed.any() else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", default="tracks/mount_panorama.json")
    ap.add_argument("--policy", choices=["classical", "mltt"], default="mltt")
    ap.add_argument("--weights")
    ap.add_argument("--pace", type=float, nargs="+", default=[0.75, 0.9])
    ap.add_argument("--mismatch", type=float, nargs="+", default=[1.0, 0.8, 1.25])
    ap.add_argument("--laps", type=int, default=2)
    ap.add_argument("--net-steer", choices=["lag", "ramp"], default="lag", help="see pilot.py --net-steer")
    ap.add_argument("--grip-limit", type=float, default=2.0)
    ap.add_argument("--edge-margin", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="also write the results here")
    a = ap.parse_args()

    track = TrackLine(a.track)
    edges = json.loads(Path(a.track).with_suffix(".edges.json").read_text())
    results = []
    for pace in a.pace:
        for m in a.mismatch:
            r = check(track, edges, a.policy, a.weights, pace, m, a.laps, a.net_steer, a.grip_limit,
                      a.edge_margin, seed=a.seed)
            r.update(pace=pace, mismatch=m)
            results.append(r)
            laps = ", ".join(f"{x:.1f}" for x in r["laps"]) or "-"
            print(f"pace {pace:.2f}  x{m:<4}  {r['result']:6s} {r['progress']:6.1%}  laps {laps:13s}"
                  f"  median {r['median_lat_m']} / max {r['max_lat_m']} m  recover {r['recover_s']} s"
                  f"  outside {r['outside_pct']}%  first off {r['first_off_s']} s"
                  + (f"  grip-capped {r['grip_capped_pct']}%  pedals governed {r['governed_pct']}%"
                     if r["grip_capped_pct"] is not None else ""))
    if a.json:
        Path(a.json).write_text(json.dumps({"args": vars(a), "results": results}, indent=1))


if __name__ == "__main__":
    main()
