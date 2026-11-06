#!/usr/bin/env python3
"""
Reference line for a circuit, built from reader.py CSVs, and live track-relative state.

ACC doesn't publish the centreline, but each frame has the car's world position
(pos_x, pos_z), heading and spline position (lap_fraction). The median path of
the flying laps becomes the reference line, with the human speed / pedal / gear
profile stored against distance. A controller can then look *ahead* on the line,
which the first ML attempt never could.

Conventions (checked against the Bathurst logs):
  ACC heading h = atan2(-dx, dz) in the (x, z) plane; positive steer (right) raises h.
  Everything returned here is "+ = right": lateral offset right of the line,
  heading error with the car pointing right of the line, look-ahead angle with
  the track bending right, steer right. Same signs as ML Track Trainer's env.js.

Usage:
  python track_line.py telemetry_logs/*mount_panorama.csv --out tracks/mount_panorama.json
  python track_line.py telemetry_logs/*mount_panorama.csv --edges tracks/mount_panorama.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

GRID_M = 2.0                              # reference line spacing
LOOK_M = (15, 40, 80, 140, 220)           # same look-ahead distances as ML Track Trainer
PROFILE = ("speed_kmh", "gas", "brake", "steer", "gear")


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def flying_laps(df: pd.DataFrame):
    """Complete laps with an official time, not the first lap, not in the pit lane."""
    out = []
    first = df.current_lap.iloc[0]
    for lap, g in df.groupby("current_lap"):
        if lap == first or g.in_pit_lane.max() > 0 or g.speed_kmh.min() < 30:
            continue
        nxt = df[df.current_lap == lap + 1]
        if nxt.empty:
            continue
        t = nxt.last_lap_ms.iloc[: 400].replace(2147483647, np.nan).max()
        if np.isnan(t) or g.is_valid_lap.min() == 0:
            continue
        out.append((lap, t / 1000.0, g))
    return out


def lap_distance(g: pd.DataFrame, lap_m: float):
    f = g.lap_fraction.to_numpy().copy()
    t = (g.timestamp - g.timestamp.iloc[0]).to_numpy()
    f[(f > 0.9) & (t < 10)] -= 1.0          # spline wraps a few frames off the lap counter
    f[(f < 0.1) & (t > 30)] += 1.0
    return f * lap_m


def build(csvs, grid_m=GRID_M):
    laps = []
    for p in csvs:
        df = pd.read_csv(p)
        for lap, t, g in flying_laps(df):
            laps.append((Path(p).name, lap, t, g))
    if not laps:
        raise SystemExit("No flying laps with world coordinates found.")
    lap_m = float(np.median([g.distance_m.iloc[-1] - g.distance_m.iloc[0] for *_, g in laps]))
    n = int(lap_m // grid_m)
    d = np.arange(n) * grid_m

    cols = {k: [] for k in ("x", "z") + PROFILE}
    for *_, g in laps:
        s = lap_distance(g, lap_m)
        order = np.argsort(s)
        s = s[order]
        for k, src in (("x", "pos_x"), ("z", "pos_z")) + tuple((c, c) for c in PROFILE):
            cols[k].append(np.interp(d, s, g[src].to_numpy()[order]))
    med = {k: np.median(np.vstack(v), axis=0) for k, v in cols.items()}

    # Light circular smoothing of the path so the tangent and curvature aren't noisy
    x, z = med["x"], med["z"]
    k = np.ones(5) / 5
    x = np.convolve(np.r_[x[-2:], x, x[:2]], k, "valid")
    z = np.convolve(np.r_[z[-2:], z, z[:2]], k, "valid")

    dx = np.roll(x, -1) - np.roll(x, 1)
    dz = np.roll(z, -1) - np.roll(z, 1)
    h = np.arctan2(-dx, dz)
    seg = np.hypot(np.roll(x, -1) - x, np.roll(z, -1) - z)
    curv = wrap(np.roll(h, -1) - np.roll(h, 1)) / (2 * seg.mean())    # 1/m, + = right-hander

    # How far each lap strays from the median line: a lower bound on usable road width
    right = np.c_[-np.cos(h), -np.sin(h)]
    spread = []
    for i, (*_, g) in enumerate(laps):
        px, pz = cols["x"][i], cols["z"][i]
        spread.append((px - x) * right[:, 0] + (pz - z) * right[:, 1])
    spread = np.vstack(spread)

    return {
        "name": Path(csvs[0]).stem.split("_", 4)[-1] if "acc_session" in Path(csvs[0]).stem else "track",
        "lap_m": lap_m,
        "grid_m": grid_m,
        "laps": [{"file": f, "lap": int(l), "time_s": t} for f, l, t, _ in laps],
        "x": x.round(3).tolist(),
        "z": z.round(3).tolist(),
        "heading": h.round(5).tolist(),
        "curvature": curv.round(6).tolist(),
        "line_spread_m": np.abs(spread).max(axis=0).round(2).tolist(),
        **{k: med[k].round(4).tolist() for k in PROFILE},
        "car": car_envelope([g for *_, g in laps]),
    }


def car_envelope(laps, bin_ms=5.0):
    """
    What the car did, per speed band (m/s): best acceleration, braking and cornering
    grip, and how much steer input a given curvature took. Accelerations come from
    dv/dt and v * yaw_rate; the logged accel_* columns have their axes swapped.
    """
    g = pd.concat(laps)
    v = g.speed_ms.to_numpy()
    dvdt = (g.speed_ms.rolling(9, center=True).mean().diff(5) / g.timestamp.diff(5)).to_numpy()
    kappa = g.yaw_rate.to_numpy() * -1 / np.maximum(v, 5)     # path curvature, + = right
    steer, gas, brake = g.steer.to_numpy(), g.gas.to_numpy(), g.brake.to_numpy()
    edges = np.arange(0, 95, bin_ms)
    out = {"v": [], "accel": [], "brake": [], "lat": [], "steer_gain": []}
    for lo in edges:
        m = (v >= lo) & (v < lo + bin_ms) & np.isfinite(dvdt)
        q = lambda sel, p: float(np.percentile(sel, p)) if sel.size > 30 else np.nan
        out["v"].append(lo + bin_ms / 2)
        out["accel"].append(q(dvdt[m & (gas > 0.9)], 90))
        out["brake"].append(-q(dvdt[m & (brake > 0.7)], 10))
        out["lat"].append(q(np.abs(v * v * kappa)[m], 99))
        c = m & (np.abs(kappa) > 0.004)
        out["steer_gain"].append(float(np.median(steer[c] / kappa[c])) if c.sum() > 30 else np.nan)
    # Fill empty bands from their neighbours so lookups never hit NaN
    for k in ("accel", "brake", "lat", "steer_gain"):
        a = np.array(out[k], dtype=float)
        ok = np.isfinite(a)
        out[k] = np.interp(out["v"], np.array(out["v"])[ok], a[ok]).round(3).tolist()
    return out


def road_edges(csvs, track_path, bin_m=10.0, pct=98, lo_m=2.0, hi_m=7.5):
    """
    How far right and left of the reference line you actually drove, per stretch of
    track: the usable road, as far as the logs can show it. ACC's tyres_out stays 0 in
    these logs, so the edges come from where the car went, not where track limits are.
    The pct-th percentile per bin_m bin drops spins; clipped to [lo_m, hi_m], smoothed.
    """
    t = TrackLine(track_path)
    S, L = [], []
    for p in csvs:
        df = pd.read_csv(p, usecols=["pos_x", "pos_z", "speed_kmh", "in_pit_lane", "lap_fraction"])
        df = df[(df.speed_kmh > 20) & (df.in_pit_lane == 0)].iloc[::2]
        t.idx = None
        for x, z, f in zip(df.pos_x, df.pos_z, df.lap_fraction):
            s, lat = t.locate(x, z, f)
            S.append(s)
            L.append(lat)
    S, L = np.array(S), np.array(L)
    nb = int(np.ceil(t.lap_m / bin_m))
    b = np.minimum((S // bin_m).astype(int), nb - 1)
    right, left = np.full(nb, np.nan), np.full(nb, np.nan)
    for i in range(nb):
        m = b == i
        if (m & (L > 0)).sum() > 10:
            right[i] = np.percentile(L[m & (L > 0)], pct)
        if (m & (L < 0)).sum() > 10:
            left[i] = np.percentile(-L[m & (L < 0)], pct)
    centres = (np.arange(nb) + 0.5) * bin_m
    d = np.arange(t.n) * t.grid
    k = np.ones(5) / 5
    out = {}
    for name, e in (("right_m", right), ("left_m", left)):
        ok = np.isfinite(e)
        e = np.interp(centres, centres[ok], e[ok])
        e = np.convolve(np.r_[e[-2:], e, e[:2]], k, "valid")         # circular smoothing
        out[name] = np.clip(np.interp(d, centres, e, period=t.lap_m), lo_m, hi_m).round(2).tolist()
    return {"track": str(track_path), "files": [Path(p).name for p in csvs], "samples": int(len(S)),
            "bin_m": bin_m, "percentile": pct, "clip_m": [lo_m, hi_m], **out}


class TrackLine:
    """Loaded reference line with nearest-point lookup and the live observation."""

    def __init__(self, path):
        t = json.loads(Path(path).read_text())
        self.meta = t
        self.lap_m = t["lap_m"]
        self.grid = t["grid_m"]
        self.x = np.array(t["x"])
        self.z = np.array(t["z"])
        self.h = np.array(t["heading"])
        self.curv = np.array(t["curvature"])
        self.v_ref = np.array(t["speed_kmh"]) / 3.6
        self.gas_ref = np.array(t["gas"])
        self.brake_ref = np.array(t["brake"])
        self.steer_ref = np.array(t["steer"])
        self.gear_ref = np.array(t["gear"])
        self.n = len(self.x)
        self.idx = None

    def at(self, s, arr):
        """Linear interpolation of a profile array at distance s (wraps)."""
        u = (s / self.grid) % self.n
        i = int(u)
        f = u - i
        return arr[i] * (1 - f) + arr[(i + 1) % self.n] * f

    def heading_at(self, s):
        u = (s / self.grid) % self.n
        i = int(u)
        return self.h[i] + wrap(self.h[(i + 1) % self.n] - self.h[i]) * (u - i)

    def locate(self, x, z, hint_frac=None, window=60):
        """Distance along the line and signed lateral offset (+ right) of world point (x, z)."""
        if self.idx is None:
            seed = int((hint_frac or 0.0) * self.n) % self.n
            cand = np.arange(self.n) if hint_frac is None else (seed + np.arange(-200, 201)) % self.n
        else:
            cand = (self.idx + np.arange(-window, window + 1)) % self.n
        d2 = (self.x[cand] - x) ** 2 + (self.z[cand] - z) ** 2
        i = int(cand[np.argmin(d2)])
        # If we lost the car (teleport, reset), fall back to a full search
        if d2.min() > 30.0 ** 2 and len(cand) < self.n:
            self.idx = None
            return self.locate(x, z, None)
        self.idx = i
        j = (i + 1) % self.n
        ex, ez = self.x[j] - self.x[i], self.z[j] - self.z[i]
        L2 = ex * ex + ez * ez or 1.0
        u = ((x - self.x[i]) * ex + (z - self.z[i]) * ez) / L2
        s = (i + u) * self.grid
        h = self.heading_at(s)
        px, pz = x - (self.x[i] + u * ex), z - (self.z[i] + u * ez)
        lat = -(px * np.cos(h) + pz * np.sin(h))
        return s % self.lap_m, lat

    def observe(self, x, z, heading, speed_ms, steer, hint_frac=None):
        """Track-relative state: the ML Track Trainer inputs, in metres / radians."""
        s, lat = self.locate(x, z, hint_frac)
        h_line = self.heading_at(s)
        herr = wrap(heading - h_line)                       # + = car pointing right of the line
        look = [wrap(self.heading_at(s + d) - heading) for d in LOOK_M]   # + = track bends right
        return {"s": s, "lat": lat, "herr": herr, "look": look, "speed": speed_ms, "steer": steer}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default="tracks/track.json")
    ap.add_argument("--edges", metavar="TRACK_JSON",
                    help="instead of building a line, measure road edges around this one from the CSVs "
                         "and save them next to it as <name>.edges.json")
    a = ap.parse_args()
    if a.edges:
        e = road_edges(a.csv, a.edges)
        out = Path(a.edges).with_suffix(".edges.json")
        out.write_text(json.dumps(e))
        for side in ("right_m", "left_m"):
            print(f"{side[:-2]:5s} edge median {np.median(e[side]):.2f} m, "
                  f"10th pct {np.percentile(e[side], 10):.2f} m, 90th {np.percentile(e[side], 90):.2f} m")
        print(f"{e['samples']} samples from {len(e['files'])} files. Saved {out}")
        return
    t = build(a.csv)
    out = Path(a.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(t))
    times = sorted(l["time_s"] for l in t["laps"])
    print(f"{len(t['laps'])} flying laps, lap {t['lap_m']:.0f} m, {len(t['x'])} points every {t['grid_m']} m")
    print(f"Lap times {times[0]:.3f} .. {times[-1]:.3f} s | max line spread {max(t['line_spread_m']):.1f} m")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
