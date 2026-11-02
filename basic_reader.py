#!/usr/bin/env python3
"""
Minimal Live Telemetry Dash for Assetto Corsa Competizione

Purpose:
- Visually validate reader.py
- No terminal spam
- Clean white UI
- Lap-level summary only
"""

import time
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from reader import ACTelemetryReader, SAMPLING_HZ

# =========================
# CONFIG
# =========================
WINDOW_SEC = 3.0
MAX_SAMPLES = int(WINDOW_SEC * SAMPLING_HZ)

# =========================
# STYLE (LIGHT MODE)
# =========================
BG_COLOR = "white"
GRID_COLOR = "#dddddd"
TEXT_COLOR = "#111111"

SPEED_COLOR = "#1f77b4"
THROTTLE_COLOR = "#2ca02c"
BRAKE_COLOR = "#d62728"
STEER_COLOR = "#ff7f0e"
G_LAT_COLOR = "#9467bd"
G_LON_COLOR = "#8c564b"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": BG_COLOR,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "grid.color": GRID_COLOR,
    "font.size": 10
})

# =========================
# BUFFERS
# =========================
speed_buf = deque(maxlen=MAX_SAMPLES)
gas_buf   = deque(maxlen=MAX_SAMPLES)
brake_buf = deque(maxlen=MAX_SAMPLES)
steer_buf = deque(maxlen=MAX_SAMPLES)
alat_buf  = deque(maxlen=MAX_SAMPLES)
along_buf = deque(maxlen=MAX_SAMPLES)

# =========================
# LAP SUMMARY
# =========================
lap_data = defaultdict(lambda: {
    "samples": 0,
    "speed_sum": 0.0,
    "speed_top": 0.0,
    "start": None,
    "end": None
})
current_lap = 1

# =========================
# FIGURE
# =========================
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

def style(ax, title, ylabel=None):
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(True, alpha=0.6)
    if ylabel:
        ax.set_ylabel(ylabel)

# --- Speed
ax_speed = fig.add_subplot(gs[0, 0])
style(ax_speed, "Speed", "km/h")
speed_line, = ax_speed.plot([], [], lw=2, color=SPEED_COLOR)
ax_speed.set_ylim(0, 350)
ax_speed.set_xlim(0, MAX_SAMPLES)

# --- Inputs (bars)
ax_input = fig.add_subplot(gs[0, 1])
ax_input.set_title("Inputs", loc="left", fontweight="bold")
ax_input.set_xlim(-1.1, 1.1)
ax_input.set_ylim(-0.5, 2.5)
ax_input.axis("off")

throttle_bar = ax_input.barh(2, 0, height=0.4, color=THROTTLE_COLOR)
brake_bar    = ax_input.barh(1, 0, height=0.4, color=BRAKE_COLOR)
steer_neg    = ax_input.barh(0, 0, height=0.4, color=STEER_COLOR)
steer_pos    = ax_input.barh(0, 0, height=0.4, color=STEER_COLOR)

ax_input.text(-1.05, 2, "Throttle")
ax_input.text(-1.05, 1, "Brake")
ax_input.text(-1.05, 0, "Steer")

# --- G Forces
ax_g = fig.add_subplot(gs[1, 0])
style(ax_g, "Acceleration", "G")
g_lat_line, = ax_g.plot([], [], label="Lateral", color=G_LAT_COLOR)
g_lon_line, = ax_g.plot([], [], label="Longitudinal", color=G_LON_COLOR)
ax_g.set_ylim(-3, 3)
ax_g.set_xlim(0, MAX_SAMPLES)
ax_g.axhline(0, lw=1, alpha=0.3)
ax_g.legend()

# --- Brake over time (replaces slip)
ax_brake = fig.add_subplot(gs[1, 1])
style(ax_brake, "Brake (Over Time)", "Brake %")
brake_line, = ax_brake.plot([], [], lw=2, color=BRAKE_COLOR)
ax_brake.set_ylim(0, 1.0)
ax_brake.set_xlim(0, MAX_SAMPLES)

# =========================
# TELEMETRY
# =========================
reader = ACTelemetryReader()
if not reader.connect():
    raise RuntimeError("ACC shared memory not found")

# =========================
# UPDATE LOOP
# =========================
def update(_):
    global current_lap
    frame = reader.read_frame()
    if frame is None:
        return

    # Buffers
    speed_buf.append(frame.speed_kmh)
    gas_buf.append(frame.gas)
    brake_buf.append(frame.brake)
    steer_buf.append(frame.steer)
    alat_buf.append(frame.accel_lateral / 9.81)
    along_buf.append(frame.accel_longitudinal / 9.81)

    # Lap stats
    lap = lap_data[current_lap]
    lap["samples"] += 1
    lap["speed_sum"] += frame.speed_kmh
    lap["speed_top"] = max(lap["speed_top"], frame.speed_kmh)

    t = time.time()
    lap["start"] = lap["start"] or t
    lap["end"] = t

    # X axis
    x = range(len(speed_buf))

    # Plots
    speed_line.set_data(x, speed_buf)
    g_lat_line.set_data(x, alat_buf)
    g_lon_line.set_data(x, along_buf)
    brake_line.set_data(x, brake_buf)

    # Bars
    throttle_bar[0].set_width(gas_buf[-1])
    brake_bar[0].set_width(brake_buf[-1])

    steer = steer_buf[-1]
    if steer >= 0:
        steer_pos[0].set_width(steer)
        steer_neg[0].set_width(0)
    else:
        steer_neg[0].set_width(-steer)
        steer_pos[0].set_width(0)

# =========================
# RUN
# =========================
ani = FuncAnimation(fig, update, interval=1000 / SAMPLING_HZ)
plt.show()

# =========================
# LAP SUMMARY (ON EXIT)
# =========================
print("\nLap Summary")
print("-" * 40)
for lap, d in lap_data.items():
    if d["samples"] == 0:
        continue
    avg = d["speed_sum"] / d["samples"]
    dur = d["end"] - d["start"]
    print(f"Lap {lap}: {dur:6.1f}s | Avg {avg:6.1f} km/h | Top {d['speed_top']:6.1f}")

reader.close()
