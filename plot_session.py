#!/usr/bin/env python3
"""
Session graphs for a reader.py CSV.

Usage:
    python plot_session.py telemetry_logs/acc_session_YYYYMMDD_HHMMSS.csv

Writes PNGs (and braking_points.csv) to graphs/<csv name>/.
Works on old logs too; graphs that need newer columns (track map, race story,
tyres, TC/ABS) are skipped when those columns are missing.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIN_SPEED_KMH = 5.0       # drop stationary frames (pits, menus)
GRID_STEP_M = 5.0         # distance resolution for lap comparisons
NO_TIME = 2_000_000_000   # ACC uses INT_MAX for "no lap time yet"
WHEELS = ["fl", "fr", "rl", "rr"]
BEST = "#d62728"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.color": "#e5e5e5", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})


def fmt_lap(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:06.3f}"


def parse_lap_time(s: str) -> float:
    """'1:39:875' -> 99.875 seconds; invalid -> NaN."""
    try:
        m, sec, ms = s.split(":")
        return int(m) * 60 + int(sec) + int(ms) / 1000
    except (ValueError, AttributeError):
        return np.nan


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / name, dpi=150)
    plt.close(fig)


# ============================================================================
# LOADING
# ============================================================================

def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["new_format"] = "lap_time_ms" in df.columns
    df["lap_t"] = df["timestamp"] - df.groupby("current_lap")["timestamp"].transform("min")

    if df["new_format"].iloc[0]:
        # normalizedCarPosition can be ~0.99 for a few frames after the lap counter ticks,
        # or ~0.01 just before; unwrap so each lap runs 0 -> 1.
        frac = df["lap_fraction"].copy()
        frac[(frac > 0.9) & (df["lap_t"] < 10)] -= 1.0
        frac[(frac < 0.1) & (df["lap_t"] > 30)] += 1.0
        lap_len = (df.groupby("current_lap")["distance_m"].agg(lambda d: d.max() - d.min())).median()
        df["lap_dist_m"] = frac * lap_len
    else:
        # Old logs: lap_fraction was 0 for lap 1, so use distance into the lap
        df["lap_dist_m"] = df["distance_m"] - df.groupby("current_lap")["distance_m"].transform("min")
    return df


def race_start_time(df: pd.DataFrame) -> float:
    """Green flag time; falls back to the first full-throttle moment above 70 km/h (rolling start)."""
    if "green_flag" in df.columns:
        green = df[(df["green_flag"] == 1) | (df["flag"] == 7)]
        if len(green):
            return green["timestamp"].iloc[0]
    go = df[(df["gas"] > 0.95) & (df["speed_kmh"] > 70)]
    return go["timestamp"].iloc[0] if len(go) else df["timestamp"].iloc[0]


def lap_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per lap; a lap is complete when completed_laps increments after it."""
    max_completed = df["completed_laps"].max()
    new = df["new_format"].iloc[0]
    rows = []
    lap_groups = dict(tuple(df.groupby("current_lap")))
    for lap, g in lap_groups.items():
        complete = lap <= max_completed
        moving = g[g["speed_kmh"] > MIN_SPEED_KMH]
        lap_time = np.nan
        valid = np.nan
        if complete and new and lap + 1 in lap_groups:
            # Official time appears in iLastTime once the next lap has started
            last = lap_groups[lap + 1]["last_lap_ms"]
            last = last[(last > 0) & (last < NO_TIME)]
            if len(last):
                lap_time = last.iloc[-1] / 1000
        if complete and np.isnan(lap_time):
            # Max of the running clock: it resets a few frames before completed_laps ticks over
            lap_time = g["lap_time_str"].map(parse_lap_time).max()
        if new:
            valid = int(g["is_valid_lap"].iloc[max(0, len(g) - 50):].min())  # value just before the line
        nxt = lap_groups.get(lap + 1)
        fuel_used = g["fuel_kg"].iloc[0] - nxt["fuel_kg"].iloc[0] if nxt is not None else np.nan
        rows.append({
            "lap": lap,
            "complete": complete,
            "lap_time_s": lap_time,
            "valid": valid,
            "length_m": g["distance_m"].max() - g["distance_m"].min(),
            "top_speed": g["speed_kmh"].max(),
            "avg_speed": moving["speed_kmh"].mean(),
            "full_throttle_pct": (moving["gas"] > 0.95).mean() * 100,
            "braking_pct": (moving["brake"] > 0.05).mean() * 100,
            "coasting_pct": ((moving["gas"] < 0.05) & (moving["brake"] < 0.05)).mean() * 100,
            "overlap_pct": ((moving["gas"] > 0.1) & (moving["brake"] > 0.1)).mean() * 100,
            "fuel_used_kg": fuel_used,
            "position_end": g["position"].iloc[-1] if new else np.nan,
        })
    return pd.DataFrame(rows)


def flying_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Complete laps excluding the first (rolling start / out-lap) and odd-length laps."""
    full = laps[laps["complete"] & laps["lap_time_s"].notna()]
    full = full[full["lap"] != laps["lap"].min()]
    med = full["length_m"].median()
    return full[(full["length_m"] - med).abs() < 0.03 * med]


def best_lap_of(laps: pd.DataFrame):
    fl = flying_laps(laps)
    pool = fl[fl["valid"] != 0] if fl["valid"].notna().any() and (fl["valid"] == 1).any() else fl
    return int(pool.loc[pool["lap_time_s"].idxmin(), "lap"]) if len(pool) else None


def time_on_grid(g: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Time since lap start at each grid distance."""
    d = np.maximum.accumulate(g["lap_dist_m"].to_numpy())
    return np.interp(grid, d, g["lap_t"].to_numpy())


# ============================================================================
# GRAPHS
# ============================================================================

def plot_lap_times(laps, best_lap, out):
    done = laps[laps["complete"] & laps["lap_time_s"].notna()]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = []
    for _, r in done.iterrows():
        if r["lap"] == best_lap:
            colors.append(BEST)
        elif r["valid"] == 0:
            colors.append("#bbbbbb")
        else:
            colors.append("#4c72b0")
    bars = ax.bar(done["lap"].astype(str), done["lap_time_s"], color=colors)
    for b, (_, r) in zip(bars, done.iterrows()):
        label = fmt_lap(r["lap_time_s"]) + (" ✗" if r["valid"] == 0 else "")
        ax.text(b.get_x() + b.get_width() / 2, r["lap_time_s"], label, ha="center", va="bottom", fontsize=9)
    ax.set_ylim(done["lap_time_s"].min() * 0.97, done["lap_time_s"].max() * 1.01)
    ax.set_xlabel("Lap"); ax.set_ylabel("Lap time (s)")
    title = "Lap times (best in red"
    title += ", invalid in grey ✗)" if done["valid"].notna().any() else ")"
    ax.set_title(title)
    save(fig, out, "lap_times.png")


def plot_speed_overlay(df, laps, best_lap, out):
    fig, ax = plt.subplots(figsize=(13, 5))
    for lap in flying_laps(laps)["lap"]:
        g = df[df["current_lap"] == lap]
        is_best = lap == best_lap
        ax.plot(g["lap_dist_m"], g["speed_kmh"], lw=1.6 if is_best else 0.8,
                alpha=1.0 if is_best else 0.45, color=BEST if is_best else None,
                label=f"Lap {lap}" + (" (best)" if is_best else ""), zorder=3 if is_best else 2)
    ax.set_xlabel("Distance into lap (m)"); ax.set_ylabel("Speed (km/h)")
    ax.set_title("Speed trace by lap")
    ax.legend(ncol=4, fontsize=8, loc="lower right")
    save(fig, out, "speed_overlay.png")


def plot_delta(df, laps, best_lap, out):
    fl = flying_laps(laps)
    best = df[df["current_lap"] == best_lap]
    grid = np.arange(0, best["lap_dist_m"].max(), GRID_STEP_M)
    t_best = time_on_grid(best, grid)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for lap in fl["lap"]:
        if lap == best_lap:
            continue
        g = df[df["current_lap"] == lap]
        ax.plot(grid, time_on_grid(g, grid) - t_best, lw=1, label=f"Lap {lap}")
    ax.axhline(0, color="black", lw=1.2, label=f"Lap {best_lap} (best)")
    ax.set_xlabel("Distance into lap (m)"); ax.set_ylabel("Delta to best (s)\n+ = slower")
    ax.set_title("Time delta to best lap")
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    save(fig, out, "delta_to_best.png")


def plot_best_lap_inputs(df, best_lap, out):
    g = df[df["current_lap"] == best_lap]
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    x = g["lap_dist_m"]
    axes[0].plot(x, g["speed_kmh"], color="#1f77b4", lw=1); axes[0].set_ylabel("Speed\n(km/h)")
    axes[1].fill_between(x, g["gas"], color="#2ca02c", alpha=0.6, lw=0, label="Throttle")
    axes[1].fill_between(x, -g["brake"], color=BEST, alpha=0.6, lw=0, label="Brake")
    if "tc_active" in g.columns:
        axes[1].fill_between(x, 0, 1, where=g["tc_active"] > 0, color="#9467bd", alpha=0.25, lw=0,
                             transform=axes[1].get_xaxis_transform(), label="TC active")
        axes[1].fill_between(x, 0, 1, where=g["abs_active"] > 0, color="#ff7f0e", alpha=0.25, lw=0,
                             transform=axes[1].get_xaxis_transform(), label="ABS active")
    axes[1].set_ylabel("Throttle / Brake"); axes[1].set_ylim(-1.05, 1.05)
    axes[1].legend(fontsize=8, loc="lower right", ncol=2)
    axes[2].plot(x, g["steer"], color="#ff7f0e", lw=1); axes[2].set_ylabel("Steer")
    axes[3].step(x, g["gear"], color="#555555", lw=1, where="post"); axes[3].set_ylabel("Gear")
    axes[3].set_xlabel("Distance into lap (m)")
    fig.suptitle(f"Best lap (lap {best_lap}) – inputs")
    save(fig, out, "best_lap_inputs.png")


def find_brake_zones(g: pd.DataFrame, on=0.2, off=0.05, min_gap_m=150):
    """Distances where braking starts (brake rises past `on` after being below `off`)."""
    zones, armed, last = [], True, -1e9
    for d, b in zip(g["lap_dist_m"], g["brake"]):
        if armed and b > on and d - last > min_gap_m:
            zones.append(d); last = d; armed = False
        elif b < off:
            armed = True
    return zones


def braking_points(df, laps, best_lap, out):
    ref = find_brake_zones(df[df["current_lap"] == best_lap])
    rows = []
    for lap in flying_laps(laps)["lap"]:
        g = df[df["current_lap"] == lap]
        onsets = np.array(find_brake_zones(g))
        for zi, zd in enumerate(ref):
            if not len(onsets):
                continue
            near = onsets[np.argmin(np.abs(onsets - zd))]
            if abs(near - zd) > 150:
                continue
            seg = g[(g["lap_dist_m"] >= near) & (g["lap_dist_m"] <= near + 400)]
            rows.append({"zone": zi + 1, "lap": lap, "brake_start_m": round(near, 1),
                         "entry_kmh": round(seg["speed_kmh"].iloc[0], 1),
                         "min_kmh": round(seg["speed_kmh"].min(), 1)})
    bp = pd.DataFrame(rows)
    if bp.empty:
        return
    bp.to_csv(out / "braking_points.csv", index=False)

    summary = bp.groupby("zone").agg(at_m=("brake_start_m", "median"),
                                     spread_m=("brake_start_m", lambda s: s.max() - s.min()),
                                     min_kmh=("min_kmh", "median"),
                                     min_kmh_spread=("min_kmh", lambda s: s.max() - s.min()))
    print("\nBraking zones (flying laps):")
    print(summary.round(1).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    zones = summary.index
    axes[0].boxplot([bp[bp["zone"] == z]["brake_start_m"] - summary.loc[z, "at_m"] for z in zones],
                    tick_labels=[f"Z{z}\n{summary.loc[z, 'at_m']:.0f} m" for z in zones])
    axes[0].set_ylabel("Brake point vs median (m)\n+ = later"); axes[0].set_title("Brake point consistency")
    axes[1].boxplot([bp[bp["zone"] == z]["min_kmh"] for z in zones], tick_labels=[f"Z{z}" for z in zones])
    axes[1].set_ylabel("Minimum speed (km/h)"); axes[1].set_title("Corner minimum speed")
    save(fig, out, "braking_points.png")


def plot_gg(df, out):
    m = df[df["speed_kmh"] > MIN_SPEED_KMH]
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    sc = ax.scatter(m["accel_lateral"], m["accel_longitudinal"], c=m["speed_kmh"],
                    s=1, alpha=0.3, cmap="viridis", rasterized=True)
    fig.colorbar(sc, ax=ax, label="Speed (km/h)")
    ax.axhline(0, color="#999", lw=0.8); ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlabel("Lateral (G)"); ax.set_ylabel("Longitudinal (G)")
    ax.set_title("G-G diagram")
    # Zoom to the 0.1–99.9th percentile so single-frame glitches don't set the axes
    for col, setter in (("accel_lateral", ax.set_xlim), ("accel_longitudinal", ax.set_ylim)):
        lo, hi = m[col].quantile([0.001, 0.999])
        pad = (hi - lo) * 0.1
        setter(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    save(fig, out, "gg_diagram.png")


def plot_lap_consistency(laps, out):
    full = laps[laps["complete"]]
    panels = [("avg_speed", "Avg speed (km/h)", "#1f77b4"),
              ("full_throttle_pct", "Full throttle (% of time)", "#2ca02c"),
              ("braking_pct", "Braking (% of time)", BEST),
              ("coasting_pct", "Coasting (% of time)", "#7f7f7f"),
              ("overlap_pct", "Throttle+brake overlap (%)", "#9467bd"),
              ("fuel_used_kg", "Fuel used (kg)", "#8c564b")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (col, title, color) in zip(axes.flat, panels):
        ax.plot(full["lap"], full[col], "o-", color=color)
        ax.set_title(title); ax.set_xlabel("Lap"); ax.set_xticks(full["lap"])
    fig.suptitle("Lap-by-lap consistency")
    save(fig, out, "lap_consistency.png")


def plot_wheels(df, out):
    m = df[(df["speed_kmh"] > MIN_SPEED_KMH) & (df["wheel_slip_valid"] == 1)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].boxplot([m[f"wheel_slip_{w}"] for w in WHEELS], tick_labels=[w.upper() for w in WHEELS],
                    showfliers=False)
    axes[0].set_title("Wheel slip by corner (no outliers)")
    axes[1].boxplot([m[f"suspension_travel_{w}"] * 1000 for w in WHEELS], tick_labels=[w.upper() for w in WHEELS],
                    showfliers=False)
    axes[1].set_title("Suspension travel (mm)")
    save(fig, out, "wheels.png")


def plot_tyres(df, out):
    m = df[df["speed_kmh"] > MIN_SPEED_KMH]
    per_lap = m.groupby("current_lap")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for metric, ax, title in (("tyre_temp", axes[0], "Tyre core temp (°C)"),
                              ("tyre_press", axes[1], "Tyre pressure (psi)"),
                              ("brake_temp", axes[2], "Brake disc temp (°C)")):
        for w in WHEELS:
            s = per_lap[f"{metric}_{w}"].mean()
            ax.plot(s.index, s.values, "o-", label=w.upper(), ms=4)
        ax.set_title(title + " – lap mean"); ax.set_xlabel("Lap")
        ax.legend(fontsize=8, ncol=2)
    save(fig, out, "tyres.png")


def plot_track_map(df, best_lap, out):
    g = df[(df["current_lap"] == best_lap) & ~((df["pos_x"] == 0) & (df["pos_z"] == 0))]
    if len(g) < 100:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    sc = axes[0].scatter(g["pos_x"], g["pos_z"], c=g["speed_kmh"], s=3, cmap="viridis")
    fig.colorbar(sc, ax=axes[0], label="Speed (km/h)", shrink=0.7)
    axes[0].set_title(f"Track map – speed (lap {best_lap})")
    pedal = g["gas"] - g["brake"]
    sc2 = axes[1].scatter(g["pos_x"], g["pos_z"], c=pedal, s=3, cmap="RdYlGn", vmin=-1, vmax=1)
    fig.colorbar(sc2, ax=axes[1], label="Brake (−1) … Throttle (+1)", shrink=0.7)
    axes[1].set_title("Track map – pedals")
    # Mark distance every 1 km so other graphs can be read against the map
    for km in range(1, int(g["lap_dist_m"].max() // 1000) + 1):
        row = g.iloc[(g["lap_dist_m"] - km * 1000).abs().argmin()]
        for ax in axes:
            ax.annotate(f"{km} km", (row["pos_x"], row["pos_z"]), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
    for ax in axes:
        ax.set_aspect("equal"); ax.axis("off")
    save(fig, out, "track_map.png")


def plot_race_story(df, start_t, out):
    r = df[(df["timestamp"] >= start_t) & (df["position"] > 0)]
    if r.empty:
        return
    minutes = (r["timestamp"] - start_t) / 60
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].step(minutes, r["position"], where="post", color="#1f77b4")
    axes[0].invert_yaxis(); axes[0].set_ylabel("Position")
    axes[0].yaxis.get_major_locator().set_params(integer=True)
    axes[0].set_title("Race story")
    ahead = r["gap_ahead_ms"].where((r["gap_ahead_ms"] > 0) & (r["gap_ahead_ms"] < 60000)) / 1000
    gb = r["gap_behind_ms"].abs()   # ACC reports gapBehind as a negative number
    behind = gb.where((gb > 0) & (gb < 60000)) / 1000
    axes[1].plot(minutes, ahead, color=BEST, lw=1, label="Gap to car ahead")
    axes[1].plot(minutes, -behind, color="#2ca02c", lw=1, label="Gap to car behind (shown negative)")
    axes[1].axhline(0, color="#999", lw=0.8)
    axes[1].set_ylabel("Gap (s)"); axes[1].set_xlabel("Minutes since green")
    axes[1].legend(fontsize=8)
    for lap_start in r.groupby("current_lap")["timestamp"].min().iloc[1:]:
        for ax in axes:
            ax.axvline((lap_start - start_t) / 60, color="#ddd", lw=0.8, zorder=0)
    save(fig, out, "race_story.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(1)
    csv_path = Path(sys.argv[1])
    out = Path("graphs") / csv_path.stem
    out.mkdir(parents=True, exist_ok=True)

    df = load(csv_path)
    new = df["new_format"].iloc[0]
    start_t = race_start_time(df)
    laps = lap_table(df)
    best_lap = best_lap_of(laps)

    print(f"Format: {'new (race fields)' if new else 'old'} | race start at {start_t:.1f} s | best lap: {best_lap}")
    print(laps.round(2).to_string(index=False))

    plot_lap_times(laps, best_lap, out)
    plot_gg(df, out)
    plot_lap_consistency(laps, out)
    plot_wheels(df, out)
    if best_lap is not None:
        plot_speed_overlay(df, laps, best_lap, out)
        plot_delta(df, laps, best_lap, out)
        plot_best_lap_inputs(df, best_lap, out)
        braking_points(df, laps, best_lap, out)
    if new:
        plot_tyres(df, out)
        plot_race_story(df, start_t, out)
        if best_lap is not None:
            plot_track_map(df, best_lap, out)
    laps.to_csv(out / "laps.csv", index=False)
    print(f"\nSaved graphs to {out}/")


if __name__ == "__main__":
    main()
