#!/usr/bin/env python3
"""
Assetto Corsa Competizione Telemetry Reader - Robotics-Ready Data Ingestion
High-frequency telemetry capture with derived features and real-time logging
Based on ACC Shared Memory v1.8.12 Documentation (layouts in acc_memory.py)
"""

import csv
import time
from dataclasses import dataclass, asdict, fields
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import Optional
from statistics import median

import numpy as np

from acc_memory import (
    SharedMemoryPage, SPageFilePhysics, SPageFileGraphic, SPageFileStatic,
    PHYSICS_SHM, GRAPHICS_SHM, STATIC_SHM, ACC_SESSION_TYPE, player_coordinates,
)

# Configuration
SAMPLING_HZ = 100
DEADZONE_STEER = 0.001
DEADZONE_THROTTLE = 0.001
DEADZONE_BRAKE = 0.001
SMOOTHING_WINDOW = 5        # Samples for moving average

WHEELS = ("fl", "fr", "rl", "rr")

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TelemetryFrame:
    """Single telemetry sample. Field order = CSV column order."""
    # Timing
    timestamp: float
    lap_time_str: str
    completed_laps: int
    current_lap: int

    # State Variables (inputs)
    speed_kmh: float
    speed_ms: float
    gear: int
    rpm: int

    # Velocity (m/s)
    vx: float
    vy: float
    vz: float

    # Acceleration (G, from accG)
    accel_longitudinal: float
    accel_lateral: float

    # Control Inputs (actions)
    gas: float
    brake: float
    steer: float

    # Position & Distance
    distance_m: float
    lap_fraction: float          # ACC normalizedCarPosition (0 = start/finish line)

    # Wheel Data (FL, FR, RL, RR)
    wheel_slip_fl: float
    wheel_slip_fr: float
    wheel_slip_rl: float
    wheel_slip_rr: float
    wheel_slip_valid: bool
    suspension_travel_fl: float  # meters
    suspension_travel_fr: float
    suspension_travel_rl: float
    suspension_travel_rr: float

    fuel_kg: float

    # --- Race context (graphics page) ---
    session_type: str
    status: int                  # 0 off, 1 replay, 2 live, 3 pause
    position: int
    lap_time_ms: int             # iCurrentTime
    last_lap_ms: int             # iLastTime
    best_lap_ms: int             # iBestTime
    is_valid_lap: int
    in_pit_lane: int
    flag: int                    # ACC_FLAG_TYPE (7 = green)
    green_flag: int              # GlobalGreen
    gap_ahead_ms: int
    gap_behind_ms: int
    session_time_left_s: float
    sector: int

    # --- Track position (world coordinates, for track maps) ---
    pos_x: float
    pos_y: float
    pos_z: float
    heading: float
    yaw_rate: float              # localAngularVel[1], rad/s

    # --- Tyres & brakes ---
    tyre_temp_fl: float          # core temp, °C
    tyre_temp_fr: float
    tyre_temp_rl: float
    tyre_temp_rr: float
    tyre_press_fl: float         # psi
    tyre_press_fr: float
    tyre_press_rl: float
    tyre_press_rr: float
    brake_temp_fl: float
    brake_temp_fr: float
    brake_temp_rl: float
    brake_temp_rr: float
    slip_angle_fl: float
    slip_angle_fr: float
    slip_angle_rl: float
    slip_angle_rr: float

    # --- Driver aids ---
    tc_active: int
    abs_active: int
    tyres_out: int               # numberOfTyresOut


class SmoothingFilter:
    """Moving average filter for control inputs"""
    def __init__(self, window_size: int = 5):
        self.gas_buffer = deque(maxlen=window_size)
        self.brake_buffer = deque(maxlen=window_size)
        self.steer_buffer = deque(maxlen=window_size)

    def update(self, gas: float, brake: float, steer: float):
        self.gas_buffer.append(gas)
        self.brake_buffer.append(brake)
        self.steer_buffer.append(steer)
        return (
            sum(self.gas_buffer) / len(self.gas_buffer),
            sum(self.brake_buffer) / len(self.brake_buffer),
            sum(self.steer_buffer) / len(self.steer_buffer)
        )


class WheelSlipFilter:
    """Median-based per-wheel filter with masking and capping."""
    def __init__(self, window: int = 3, speed_threshold_kmh: float = 2.0, cap: float = 3.0):
        self.deques = {w: deque(maxlen=window) for w in WHEELS}
        self.speed_threshold_kmh = speed_threshold_kmh
        self.cap = cap

    def update(self, raw_slips, speed_kmh):
        # If speed is below threshold, return zeros and invalid flag
        if speed_kmh is None or speed_kmh < self.speed_threshold_kmh:
            # still append to history to avoid sudden jumps on restart
            for i, w in enumerate(WHEELS):
                self.deques[w].append(float(raw_slips[i]))
            return [0.0, 0.0, 0.0, 0.0], False

        out = []
        valid = True
        for i, w in enumerate(WHEELS):
            self.deques[w].append(float(raw_slips[i]))
            med = float(median(self.deques[w]))
            if med > self.cap:
                valid = False
                med = self.cap
            out.append(max(0.0, med))
        return out, valid


class DistanceTracker:
    """Track distance traveled via trapezoidal integration"""
    def __init__(self):
        self.total_distance = 0.0
        self.prev_speed_ms = 0.0
        self.prev_time = 0.0

    def update(self, speed_ms: float, elapsed: float) -> float:
        if self.prev_time > 0:
            dt = elapsed - self.prev_time
            self.total_distance += (speed_ms + self.prev_speed_ms) / 2 * dt
        self.prev_speed_ms = speed_ms
        self.prev_time = elapsed
        return self.total_distance


class SessionTracker:
    """Track session statistics and lap boundaries"""
    def __init__(self):
        self.start_time = time.time()
        self.lap_count = 0
        self.sample_count = 0
        self.max_speed = 0.0
        self.max_gas = 0.0
        self.max_brake = 0.0
        self.max_steer = 0.0
        self.max_accel_long = 0.0
        self.max_accel_lat = 0.0
        self.max_rpm = 0
        self.last_lap_number = 0
        self.best_lap_ms = 0
        self.position = 0

    def update(self, frame: TelemetryFrame):
        self.sample_count += 1
        self.max_speed = max(self.max_speed, frame.speed_kmh)
        self.max_gas = max(self.max_gas, frame.gas)
        self.max_brake = max(self.max_brake, frame.brake)
        self.max_steer = max(self.max_steer, abs(frame.steer))
        if frame.speed_kmh > 20:   # resets/pit stops produce single-frame G spikes at low speed
            self.max_accel_long = max(self.max_accel_long, abs(frame.accel_longitudinal))
            self.max_accel_lat = max(self.max_accel_lat, abs(frame.accel_lateral))
        self.max_rpm = max(self.max_rpm, frame.rpm)
        if 0 < frame.best_lap_ms < 2_000_000_000:
            self.best_lap_ms = frame.best_lap_ms
        if frame.position > 0:
            self.position = frame.position

        if frame.current_lap > self.last_lap_number:
            self.last_lap_number = frame.current_lap
            self.lap_count += 1

    def get_summary(self):
        elapsed = time.time() - self.start_time
        return {
            "duration_min": elapsed / 60,
            "samples": self.sample_count,
            "laps": self.lap_count,
            "max_speed_kmh": self.max_speed,
            "max_gas": self.max_gas,
            "max_brake": self.max_brake,
            "max_steer": self.max_steer,
            "max_rpm": self.max_rpm,
            "max_accel_g_long": self.max_accel_long,  # accG is already in G
            "max_accel_g_lat": self.max_accel_lat,
            "best_lap_ms": self.best_lap_ms,
            "position": self.position,
        }

# ============================================================================
# SHARED MEMORY READER
# ============================================================================

class ACTelemetryReader:
    """Read and parse Assetto Corsa Competizione shared memory"""

    def __init__(self):
        self.physics: Optional[SharedMemoryPage] = None
        self.graphics: Optional[SharedMemoryPage] = None
        self.static: Optional[SharedMemoryPage] = None
        self.smoother = SmoothingFilter(window_size=SMOOTHING_WINDOW)
        self.wheel_slip_filter = WheelSlipFilter(window=3, speed_threshold_kmh=2.0, cap=3.0)
        self.distance_tracker = DistanceTracker()
        self.session = SessionTracker()

    def connect(self):
        """Open shared memory connections"""
        try:
            self.physics = SharedMemoryPage(SPageFilePhysics, PHYSICS_SHM)
            self.graphics = SharedMemoryPage(SPageFileGraphic, GRAPHICS_SHM)
            self.static = SharedMemoryPage(SPageFileStatic, STATIC_SHM)
            return True
        except (FileNotFoundError, OSError):
            return False

    def session_info(self) -> dict:
        """Track / car / online flag from the static page (fixed for the session)."""
        s = self.static.read()
        return {"track": s.track, "car": s.carModel, "is_online": s.isOnline, "max_rpm": s.maxRpm}

    def read_frame(self) -> Optional[TelemetryFrame]:
        """Read and parse a complete telemetry frame"""
        if not self.physics or not self.graphics:
            return None

        try:
            p = self.physics.read()
            g = self.graphics.read()

            gas, brake, steer = p.gas, p.brake, p.steerAngle
            if abs(steer) < DEADZONE_STEER:
                steer = 0.0
            if abs(gas) < DEADZONE_THROTTLE:
                gas = 0.0
            if abs(brake) < DEADZONE_BRAKE:
                brake = 0.0

            elapsed = time.time() - self.session.start_time
            speed_kmh = float(np.clip(p.speedKmh, 0, 400))
            speed_ms = speed_kmh / 3.6
            distance = self.distance_tracker.update(speed_ms, elapsed)
            wheel_slip, wheel_slip_valid = self.wheel_slip_filter.update(list(p.wheelSlip), speed_kmh)
            coords = player_coordinates(g) or (0.0, 0.0, 0.0)

            frame = TelemetryFrame(
                timestamp=elapsed,
                lap_time_str=g.currentTime,
                completed_laps=g.completedLaps,
                current_lap=g.completedLaps + 1,
                speed_kmh=speed_kmh,
                speed_ms=speed_ms,
                gear=int(np.clip(p.gear - 1, 0, 8)),   # ACC: 0=R, 1=N, 2=1st
                rpm=int(np.clip(p.rpm, 0, 12000)),
                vx=p.velocity[0], vy=p.velocity[1], vz=p.velocity[2],
                accel_longitudinal=p.accG[0],
                accel_lateral=p.accG[2],
                gas=gas, brake=brake, steer=steer,
                distance_m=distance,
                lap_fraction=g.normalizedCarPosition,
                wheel_slip_fl=wheel_slip[0], wheel_slip_fr=wheel_slip[1],
                wheel_slip_rl=wheel_slip[2], wheel_slip_rr=wheel_slip[3],
                wheel_slip_valid=wheel_slip_valid,
                suspension_travel_fl=p.suspensionTravel[0], suspension_travel_fr=p.suspensionTravel[1],
                suspension_travel_rl=p.suspensionTravel[2], suspension_travel_rr=p.suspensionTravel[3],
                fuel_kg=p.fuel,
                session_type=ACC_SESSION_TYPE.get(g.session, str(g.session)),
                status=g.status,
                position=g.position,
                lap_time_ms=g.iCurrentTime,
                last_lap_ms=g.iLastTime,
                best_lap_ms=g.iBestTime,
                is_valid_lap=g.isValidLap,
                in_pit_lane=g.isInPitLane,
                flag=g.flag,
                green_flag=g.globalGreen,
                gap_ahead_ms=g.gapAhead,
                gap_behind_ms=g.gapBehind,
                session_time_left_s=g.sessionTimeLeft / 1000.0,   # ACC reports ms
                sector=g.currentSectorIndex,
                pos_x=coords[0], pos_y=coords[1], pos_z=coords[2],
                heading=p.heading,
                yaw_rate=p.localAngularVel[1],
                tyre_temp_fl=p.tyreCoreTemperature[0], tyre_temp_fr=p.tyreCoreTemperature[1],
                tyre_temp_rl=p.tyreCoreTemperature[2], tyre_temp_rr=p.tyreCoreTemperature[3],
                tyre_press_fl=p.wheelPressure[0], tyre_press_fr=p.wheelPressure[1],
                tyre_press_rl=p.wheelPressure[2], tyre_press_rr=p.wheelPressure[3],
                brake_temp_fl=p.brakeTemp[0], brake_temp_fr=p.brakeTemp[1],
                brake_temp_rl=p.brakeTemp[2], brake_temp_rr=p.brakeTemp[3],
                slip_angle_fl=p.slipAngle[0], slip_angle_fr=p.slipAngle[1],
                slip_angle_rl=p.slipAngle[2], slip_angle_rr=p.slipAngle[3],
                tc_active=p.tcinAction,
                abs_active=p.absInAction,
                tyres_out=p.numberOfTyresOut,
            )

            self.session.update(frame)
            return frame

        except Exception as e:
            print(f"Error reading telemetry: {e}")
            return None

    def close(self):
        """Close shared memory connections"""
        for page in (self.physics, self.graphics, self.static):
            if page:
                page.close()

# ============================================================================
# CSV LOGGER
# ============================================================================

def create_log_file(track: str = "") -> Path:
    """Create timestamped log file, tagged with the track name when known"""
    logs_dir = Path("telemetry_logs")
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{track}" if track else ""
    return logs_dir / f"acc_session_{timestamp}{suffix}.csv"


def get_csv_headers():
    """CSV column headers, taken from the TelemetryFrame field order"""
    return [f.name for f in fields(TelemetryFrame)]


def frame_to_csv_row(frame: TelemetryFrame):
    """Convert TelemetryFrame to CSV row (floats rounded to 4 dp, bools as 0/1)"""
    row = []
    for value in asdict(frame).values():
        if isinstance(value, (bool, np.bool_)):
            row.append(int(value))
        elif isinstance(value, (float, np.floating)):
            row.append(f"{value:.4f}")
        else:
            row.append(value)
    return row

# ============================================================================
# MAIN LOOP
# ============================================================================

def format_ms(ms: int) -> str:
    return f"{ms // 60000}:{(ms % 60000) / 1000:06.3f}" if 0 < ms < 2_000_000_000 else "-"


def print_session_summary(session: SessionTracker):
    """Print final session statistics"""
    summary = session.get_summary()
    print("\n" + "="*70)
    print("📊 SESSION SUMMARY")
    print("="*70)
    print(f"Duration:         {summary['duration_min']:6.1f} minutes")
    print(f"Samples:          {summary['samples']:6,}")
    print(f"Laps:             {summary['laps']:6}")
    print(f"Best Lap:         {format_ms(summary['best_lap_ms']):>6}")
    print(f"Final Position:   {summary['position'] or '-':>6}")
    print(f"Top Speed:        {summary['max_speed_kmh']:6.1f} km/h  ({summary['max_speed_kmh']/1.609:.1f} mph)")
    print(f"Max RPM:          {summary['max_rpm']:6,}")
    print(f"Max Throttle:     {summary['max_gas']:6.1%}")
    print(f"Max Brake:        {summary['max_brake']:6.1%}")
    print(f"Max |Steer|:      {summary['max_steer']:6.3f}")
    print(f"Max Accel (G):    {summary['max_accel_g_long']:.2f} long, {summary['max_accel_g_lat']:.2f} lat")
    print("="*70)


def main():
    print("="*70)
    print("ASSETTO CORSA COMPETIZIONE TELEMETRY LOGGER")
    print("High-frequency data capture (ACC Shared Memory v1.8.12)")
    print("="*70)

    reader = ACTelemetryReader()
    csv_path = None

    try:
        print("🔌 Connecting to shared memory...")
        if not reader.connect():
            print("\n❌ ACC shared memory not found. Is Assetto Corsa Competizione running?")
            return

        info = reader.session_info()
        print(f"✓ Connected!  Track: {info['track'] or '?'} | Car: {info['car'] or '?'} | "
              f"{'Online' if info['is_online'] else 'Offline'}")

        csv_path = create_log_file(info["track"])
        print(f"📁 Logging to: {csv_path}")
        print(f"⚙️  Sampling at {SAMPLING_HZ} Hz")
        print(f"📊 Capturing: {len(get_csv_headers())} channels")
        print("\n🟢 Recording... (Ctrl+C to stop)\n")

        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(get_csv_headers())

            last_print_time = time.time()
            period = 1.0 / SAMPLING_HZ
            next_tick = time.perf_counter()

            while True:
                frame = reader.read_frame()

                if frame:
                    writer.writerow(frame_to_csv_row(frame))

                    # Periodic status update (every 5 seconds)
                    if time.time() - last_print_time >= 5.0:
                        print(f"{frame.timestamp/60:5.1f}m | Lap {frame.current_lap} | P{frame.position} | "
                              f"{frame.speed_kmh:5.0f} km/h | {frame.rpm:5,} RPM | "
                              f"Best {format_ms(reader.session.best_lap_ms)} | "
                              f"Samples: {reader.session.sample_count:,}")
                        last_print_time = time.time()

                    # Flush every 500 samples
                    if reader.session.sample_count % 500 == 0:
                        csv_file.flush()

                # Fixed-rate schedule so read/write time doesn't lower the sample rate
                next_tick += period
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.perf_counter()

    except KeyboardInterrupt:
        print(f"\n\n⏹️  Recording stopped")
        print(f"💾 Saved: {csv_path}")
        print_session_summary(reader.session)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        reader.close()


if __name__ == "__main__":
    main()
