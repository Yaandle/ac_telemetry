#!/usr/bin/env python3
"""
pilot_report.py: summarise pilot.py logs for docs/ml_record.md.

For each log (training_logs/pilot_*.csv, with its .json sidecar when there is one):
  distance driven, time, median / max distance from the line, time recovering (classical steering)
  off-road: frames beyond the measured road edges (<track>.edges.json) + --edge-margin, and where
  ACC: where its lap-validity flag first went to 0 (track limits), and its lap times
  steering: target sign flips per frame, time at full lock, grip-capped frames
  pedals: share of frames where the speed envelope, not the policy, set them (mltt only)
  how it ended, with --tail seconds of the log before the end

  python pilot_report.py training_logs/pilot_mltt_20261106_191618.csv
  python pilot_report.py training_logs/pilot_mltt_20261106_19*.csv --tail 3
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from track_line import TrackLine


def spans(mask, t, s):
    """(t0, t1, s0, s1) of each run of True in mask."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((t[i], t[j], s[i], s[j]))
            i = j + 1
        else:
            i += 1
    return out


def report(path, track, edges, margin, tail):
    d = pd.read_csv(path)
    side = Path(path).with_suffix(".json")
    meta = json.loads(side.read_text()) if side.exists() else {}
    L = track.lap_m
    t, s, lat = d.t.to_numpy(), d.s.to_numpy(), d.lat.to_numpy()
    ds = (np.diff(s) + L / 2) % L - L / 2
    dist = float(ds.sum())
    R, Lf = np.array(edges["right_m"]), np.array(edges["left_m"])
    i = np.round(s / track.grid).astype(int) % len(R)
    past = np.where(lat > 0, lat - R[i], -lat - Lf[i])            # m past the measured edge (+ = outside)
    off = past > margin
    lost = d.lost.to_numpy() == 1
    sd = d.steer_des.to_numpy()
    flips = np.mean(np.sign(sd[1:]) * np.sign(sd[:-1]) < 0)
    print(f"\n== {Path(path).name}  {meta.get('policy', '?')} {Path(meta.get('weights') or '').name} "
          f"pace {meta.get('pace', '?')}  result {meta.get('result', '?')}")
    print(f"start {s[0]:.0f} m ({100 * s[0] / L:.1f}%), {dist:.0f} m driven = {100 * dist / L:.1f}% of a lap, "
          f"{t[-1]:.1f} s; line error median {np.median(np.abs(lat)):.2f} / max {np.abs(lat).max():.2f} m; "
          f"recovering {lost.sum() * np.median(np.diff(t)):.1f} s")
    for t0, t1, s0, s1 in spans(lost, t, s):
        print(f"   recovering {t0:.1f}-{t1:.1f} s at {s0:.0f}-{s1:.0f} m ({100 * s0 / L:.1f}%)")
    print(f"off the measured road (+{margin} m): {100 * off.mean():.1f}% of frames, worst {past.max():.2f} m past the edge")
    for t0, t1, s0, s1 in spans(off, t, s)[:12]:
        k = (t >= t0) & (t <= t1)
        side_ = "left" if np.median(lat[k]) < 0 else "right"
        print(f"   off {side_} {t0:.1f}-{t1:.1f} s at {s0:.0f}-{s1:.0f} m ({100 * s0 / L:.1f}%), up to {past[k].max():.2f} m past")
    if "acc_valid" in d.columns and d.acc_valid.notna().any():
        inv = d[d.acc_valid == 0]
        if len(inv):
            r = inv.iloc[0]
            print(f"ACC invalidated the lap at t {r.t:.1f} s, {r.s:.0f} m ({100 * r.s / L:.1f}%), lat {r.lat:+.2f} m")
        else:
            print("ACC: lap valid throughout")
    for lap in meta.get("laps", []):
        print(f"   lap {lap['lap']}: {lap['time_s']} s, ACC {lap['acc_s']}, valid {lap['acc_valid']}, "
              f"median {lap['median_lat_m']} / max {lap['max_lat_m']} m, recovering {lap['recover_s']} s")
    full = np.mean(np.abs(d.steer_cmd) >= 0.99)
    print(f"steer target sign flips {100 * flips:.1f}% of frames; joystick at full lock {100 * full:.1f}%"
          + (f"; grip-capped {100 * meta['grip_capped_frames'] / len(d):.1f}%" if meta.get("grip_capped_frames") is not None else ""))
    if meta.get("policy") == "mltt":
        gov = (d.speed > d.v_target + 1.0) & (d.lost == 0)
        print(f"pedals set by the speed envelope on {100 * gov.mean():.1f}% of frames")
    if tail:
        print(f"last {tail} s:")
        sub = d[d.t >= t[-1] - tail].iloc[::max(1, int(len(d[d.t >= t[-1] - tail]) / 15))]
        print(sub[["t", "s", "lat", "herr", "speed", "v_target", "gas", "brake", "steer_des", "steer_cmd", "steer_meas",
                   "lost", "gear"] + [c for c in ("acc_valid",) if c in d.columns]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--track", default="tracks/mount_panorama.json")
    ap.add_argument("--edge-margin", type=float, default=0.3)
    ap.add_argument("--tail", type=float, default=0.0, help="print this many seconds of the log before the end")
    a = ap.parse_args()
    track = TrackLine(a.track)
    edges = json.loads(Path(a.track).with_suffix(".edges.json").read_text())
    for p in a.logs:
        report(p, track, edges, a.edge_margin, a.tail)


if __name__ == "__main__":
    main()
