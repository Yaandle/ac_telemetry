#!/usr/bin/env python3
"""
AI Driver - Real-Time Control Using Trained MLP Models
Assetto Corsa + vgamepad (Xbox 360)
UPDATED: Uses feature_schema.py for consistency and temporal support
Now uses RPM instead of gear
"""

import sys
import argparse
import time
import mmap
import struct
import joblib
import numpy as np
import vgamepad as vg
from pathlib import Path
from typing import Tuple
from collections import deque

# Import feature schema
try:
    from feature_schema import (
        POLICY_FEATURES, SEQUENCE_LENGTH,
        build_state_from_memory, validate_state, validate_normalization
    )
except ImportError:
    print("❌ Missing feature_schema.py. Ensure it's in the same directory.")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_DIR = Path("ml_results")
NORM_DIR = Path("ml_data")

PHYSICS_SHM = "Local\\acpmf_physics"
GRAPHICS_SHM = "Local\\acpmf_graphics"
STATIC_SHM = "Local\\acpmf_static"

READ_BLOCK_SIZE = 512
GRAPHICS_BLOCK_SIZE = 256

UPDATE_HZ = 100
DT = 1.0 / UPDATE_HZ

# --- Temporal mode (set to False for frame-wise mode) ---
USE_TEMPORAL = False  # Toggle this to enable/disable temporal window

# --- Smoothing ---
SMOOTHING_ALPHA = 0.30
STEER_SMOOTH_ALPHA = 0.6

# --- Deadzones ---
DEADZONE_GAS = 0.01
DEADZONE_BRAKE = 0.01
DEADZONE_STEER = 0.001

# --- Control gains ---
GAS_GAIN = 1.2
BRAKE_GAIN = 1.2
STEER_GAIN = 1.0

# =============================================================================
# SHARED MEMORY OFFSETS
# =============================================================================

SPEED_OFFSET = 28
RPM_OFFSET = 20      
STEER_OFFSET = 24
GAS_OFFSET = 4
BRAKE_OFFSET = 8
GEAR_OFFSET = 16

VELOCITY_OFFSET = 32   
ACCEL_OFFSET = 44      

WHEEL_SLIP_OFFSET = 56
SUSPENSION_TRAVEL_OFFSET = 184

# Graphics offsets for lap tracking
GRAPHICS_CURRENT_TIME = 12
GRAPHICS_COMPLETED_LAPS = 132

# =============================================================================
# DISTANCE TRACKER
# =============================================================================

class DistanceTracker:
    """Track distance traveled via trapezoidal integration"""
    def __init__(self):
        self.total_distance = 0.0
        self.prev_speed_ms = 0.0
        self.prev_time = 0.0
        self.track_length = 5000.0
        self.lap_distance = 0.0
        
    def update(self, speed_ms: float, elapsed: float) -> float:
        if self.prev_time > 0:
            dt = elapsed - self.prev_time
            avg_speed = (speed_ms + self.prev_speed_ms) / 2
            distance_delta = avg_speed * dt
            self.total_distance += distance_delta
            self.lap_distance += distance_delta
        
        self.prev_speed_ms = speed_ms
        self.prev_time = elapsed
        return self.total_distance
    
    def get_lap_fraction(self) -> float:
        """Return progress through lap [0-1]"""
        if self.track_length > 0:
            return min(self.lap_distance / self.track_length, 0.999)
        return 0.0
    
    def mark_lap_complete(self):
        """Update track length estimate on lap completion"""
        if self.lap_distance > 100:
            if self.track_length == 5000.0:
                self.track_length = self.lap_distance
                print(f"   Track length established: {self.track_length:.0f}m")
            else:
                self.track_length = 0.9 * self.track_length + 0.1 * self.lap_distance
        
        self.lap_distance = 0.0


# -----------------------------------------------------------------------------
# Deterministic controller (open-loop by lap fraction)
# -----------------------------------------------------------------------------
class DeterministicController:
    """Simple open-loop controller that maps lap_fraction -> mean actions.

    Loads `ml_data/X_states.npy` and `ml_data/y_actions.npy` by default and
    constructs a binned mean lookup table for (gas, brake, steer) indexed by
    lap fraction in [0,1).
    """
    def __init__(self, data_dir: Path = Path("ml_data"), bins: int = 100):
        self.data_dir = Path(data_dir)
        self.bins = int(bins)
        self.actions_table = None
        self.bin_edges = None

    def load(self) -> bool:
        x_path = self.data_dir / "X_states.npy"
        y_path = self.data_dir / "y_actions.npy"
        if not x_path.exists() or not y_path.exists():
            print(f"⚠️ Deterministic data not found: {x_path}, {y_path}")
            return False

        try:
            X = np.load(x_path)
            Y = np.load(y_path)
            # Resolve feature indices dynamically from schema
            lap_idx = POLICY_FEATURES.index('lap_fraction')
            speed_idx = POLICY_FEATURES.index('speed_ms')
            gear_idx = POLICY_FEATURES.index('gear')

            lap_frac = np.clip(X[:, lap_idx].astype(float), 0.0, 0.999)
            speeds = X[:, speed_idx].astype(float)

            edges = np.linspace(0.0, 1.0, self.bins + 1)
            indices = np.digitize(lap_frac, edges) - 1

            # Build per-bin arrays of (speed, action, gear)
            gears = X[:, gear_idx].astype(int)
            bins_data = [[] for _ in range(self.bins)]
            for i, idx in enumerate(indices):
                if 0 <= idx < self.bins:
                    bins_data[idx].append((speeds[i], Y[i], gears[i]))

            # Convert to numpy arrays, sort by speed for efficient lookup
            self.actions_table = []
            for b in range(self.bins):
                arr = bins_data[b]
                if len(arr) == 0:
                    self.actions_table.append((np.array([]), np.array([]), np.array([], dtype=int)))
                else:
                    arr_sorted = sorted(arr, key=lambda x: x[0])
                    sp = np.array([a[0] for a in arr_sorted], dtype=float)
                    ac = np.array([a[1] for a in arr_sorted], dtype=float)
                    ge = np.array([a[2] for a in arr_sorted], dtype=int)
                    self.actions_table.append((sp, ac, ge))

            self.bin_edges = edges
            print(f"✅ Deterministic controller loaded ({len(X)} samples, {self.bins} bins)")
            return True
        except Exception as e:
            print(f"❌ Failed to load deterministic controller data: {e}")
            return False

    def get_action(self, lap_fraction: float, speed_ms: float = 0.0):
        if self.actions_table is None:
            return 0.0, 0.0, 0.0
        lf = float(np.clip(lap_fraction, 0.0, 0.999))
        idx = min(int(lf * self.bins), self.bins - 1)
        speeds, actions, gears = self.actions_table[idx]
        if speeds.size == 0:
            # fallback: search nearest non-empty bin
            for offset in range(1, self.bins):
                for j in (idx - offset, idx + offset):
                    if 0 <= j < self.bins and self.actions_table[j][0].size > 0:
                        speeds, actions, gears = self.actions_table[j]
                        break
                if speeds.size > 0:
                    break
            if speeds.size == 0:
                return 0.0, 0.0, 0.0

        # find nearest speed sample
        speed_ms = float(speed_ms)
        i = int(np.argmin(np.abs(speeds - speed_ms)))
        a = actions[i]
        g = int(gears[i]) if gears.size > 0 else 0
        return float(a[0]), float(a[1]), float(a[2]), g


# =============================================================================
# AI DRIVER
# =============================================================================

class AIDriver:
    """Real-time AI driver using trained MLP models"""
    def __init__(self, shift_method: str = "bumper", self_test: bool = False):
        print("🤖 Initialising AI Driver...")
        print(f"   Mode: {'TEMPORAL' if USE_TEMPORAL else 'FRAME-WISE'}")
        if USE_TEMPORAL:
            print(f"   Temporal window: {SEQUENCE_LENGTH} frames")

        # Shift method: 'bumper' (press shoulder buttons) or 'trigger' (analog triggers)
        self.shift_method = shift_method
        self.self_test = self_test

        # --- Load models ---
        self.model_gas = joblib.load(MODEL_DIR / "mlp_256_128_64_gas.pkl")
        self.model_brake = joblib.load(MODEL_DIR / "mlp_256_128_64_brake.pkl")
        self.model_steer = joblib.load(MODEL_DIR / "mlp_256_128_64_steer.pkl")

        # --- Load normalisation ---
        self.norm_min = np.load(NORM_DIR / "normalization_min.npy")
        self.norm_max = np.load(NORM_DIR / "normalization_max.npy")
        self.norm_range = np.maximum(self.norm_max - self.norm_min, 1e-6)

        self.current_gear = 1          # internal gear state
        self.RPM_UPSHIFT = 6500
        self.RPM_DOWNSHIFT = 3000
        self.MAX_GEAR = 6
        self.MIN_GEAR = 1

        # CRITICAL VALIDATION
        print(f"\n🔍 Validating configuration...")
        print(f"   Expected features: {len(POLICY_FEATURES)}")
        print(f"   Loaded norm_min: {len(self.norm_min)}")
        print(f"   Loaded norm_max: {len(self.norm_max)}")
        
        try:
            validate_normalization(self.norm_min, self.norm_max)
            print(f"   ✅ Normalization validated")
        except AssertionError as e:
            print(f"\n❌ VALIDATION ERROR:\n{e}")
            sys.exit(1)

        # --- Temporal buffer ---
        if USE_TEMPORAL:
            self.state_buffer = deque(maxlen=SEQUENCE_LENGTH)
            print(f"   ✅ Temporal buffer initialized ({SEQUENCE_LENGTH} frames)")
        else:
            self.state_buffer = None

        # --- Gamepad ---
        self.gamepad = vg.VX360Gamepad()

        # Optional self-test of inputs
        if self.self_test:
            print("Running virtual gamepad self-test...")
            try:
                # Bumpers
                self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
                self.gamepad.update()
                time.sleep(0.08)
                self.gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
                self.gamepad.update()
                self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
                self.gamepad.update()
                time.sleep(0.08)
                self.gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
                self.gamepad.update()
                # Triggers
                self.gamepad.right_trigger_float(1.0)
                self.gamepad.left_trigger_float(1.0)
                self.gamepad.update()
                time.sleep(0.12)
                self.gamepad.right_trigger_float(0.0)
                self.gamepad.left_trigger_float(0.0)
                self.gamepad.update()
                print("Self-test completed")
            except Exception as e:
                print(f"Self-test failed: {e}")

        # --- Shared memory ---
        self.physics_shm = mmap.mmap(-1, READ_BLOCK_SIZE, PHYSICS_SHM)
        self.graphics_shm = mmap.mmap(-1, GRAPHICS_BLOCK_SIZE, GRAPHICS_SHM)

        # --- Smoothed controls ---
        self.s_gas = 0.0
        self.s_brake = 0.0
        self.s_steer = 0.0
        self.prev_sent_steer = 0.0
        self.steer_smooth_alpha = STEER_SMOOTH_ALPHA
        self.steer_max_rate = 1.0  # max change in steer per second (clamps abrupt turns)

        # per-driver gains (overridable via CLI)
        self.steer_gain = STEER_GAIN

        # --- Lap tracking ---
        self.distance_tracker = DistanceTracker()
        self.last_completed_laps = 0
        self.start_time = time.time()

        # --- RPM instead of gear ---
        self.current_rpm = 0.0
            # --- Static shared memory (attempt to read maxRpm) ---
        self.static_shm = None
        self.try_read_static_maxrpm()

            # Steering PD controller state
        self.steer_p = 1.0
        self.steer_d = 0.08
        self.prev_steer_err = 0.0
        self.prev_time = time.time()

        # --- State ---
        self.frames = 0
        
        # --- Diagnostics ---
        self.prediction_history = []
        # Optional deterministic controller
        self.deterministic_controller = None

        print("✅ AI Driver ready\n")

    # =========================================================================
    # PLAYBACK (offline) - replay recorded actions from dataset
    # =========================================================================
    def run_playback_actions(self, actions_path: str, rate_hz: int = UPDATE_HZ):
        """Replay actions saved in a numpy array (N,3) as (gas,brake,steer)."""
        path = Path(actions_path)
        if not path.exists():
            print(f"❌ Playback file not found: {path}")
            return

        print(f"▶️  Playback mode: sending actions from {path} at {rate_hz} Hz")
        actions = np.load(path)
        interval = 1.0 / rate_hz

        try:
            for i, (gas, brake, steer) in enumerate(actions):
                # smoothing + apply controls
                gas_s, brake_s, steer_s = self.smooth(float(gas), float(brake), float(steer))
                # No live speed available in this mode; pass a non-zero safe speed to avoid min-speed clamp
                self.apply_controls(gas_s, brake_s, steer_s, max(2.0, 0.0))
                if (i + 1) % (rate_hz * 2) == 0:
                    print(f"  Played {i+1:,} frames")
                time.sleep(interval)
        except KeyboardInterrupt:
            pass
        finally:
            print("✅ Playback finished")
            self.shutdown()

    def run_playback_states(self, states_path: str, actions_path: str, rate_hz: int = UPDATE_HZ):
        """Replay states and actions together. States should be ml_data/X_states.npy.
        This will also reproduce gear changes by issuing simulated shifts."""
        p_states = Path(states_path)
        p_actions = Path(actions_path)
        if not p_states.exists() or not p_actions.exists():
            print(f"❌ Playback files not found: {p_states}, {p_actions}")
            return

        print(f"▶️  State+Action playback: {p_states} + {p_actions} at {rate_hz} Hz")
        X = np.load(p_states)
        Y = np.load(p_actions)
        n = min(len(X), len(Y))
        interval = 1.0 / rate_hz

        # Resolve indices from feature schema to avoid hard-coded indices
        gear_idx = POLICY_FEATURES.index('gear')
        speed_idx = POLICY_FEATURES.index('speed_ms')

        try:
            for i in range(n):
                state = X[i]
                gas, brake, steer = Y[i]

                # Extract recorded gear and speed using schema indices
                try:
                    target_gear = int(round(state[gear_idx]))
                except Exception:
                    target_gear = self.current_gear

                # If recorded gear differs, issue shifts to match
                while self.current_gear < target_gear:
                    self.shift_up()
                    time.sleep(0.05)
                while self.current_gear > target_gear:
                    self.shift_down()
                    time.sleep(0.05)

                speed_ms = float(state[speed_idx])
                speed_kmh = speed_ms * 3.6

                gas_s, brake_s, steer_s = self.smooth(float(gas), float(brake), float(steer))
                self.apply_controls(gas_s, brake_s, steer_s, speed_kmh)

                if (i + 1) % (rate_hz * 2) == 0:
                    print(f"  Played {i+1:,} frames | Gear {self.current_gear} | Speed {speed_kmh:.1f} km/h")

                time.sleep(interval)
        except KeyboardInterrupt:
            pass
        finally:
            print("✅ State+Action playback finished")
            self.shutdown()

    def enable_deterministic(self, data_dir: Path = Path("ml_data"), bins: int = 100) -> bool:
        """Enable deterministic open-loop controller using precomputed actions.

        Loads ml_data/X_states.npy and ml_data/y_actions.npy from `data_dir`.
        Returns True on success.
        """
        controller = DeterministicController(data_dir=data_dir, bins=bins)
        ok = controller.load()
        if ok:
            self.deterministic_controller = controller
        else:
            self.deterministic_controller = None
        return ok



# =========================================================================
    # TELEMETRY
    # =========================================================================

    def read_telemetry(self) -> Tuple[np.ndarray, float]:
        """Read physics shared memory and return ML state vector"""
        self.physics_shm.seek(0)
        physics_data = self.physics_shm.read(READ_BLOCK_SIZE)

        self.graphics_shm.seek(0)
        graphics_data = self.graphics_shm.read(GRAPHICS_BLOCK_SIZE)

        # --- Basic metrics ---
        speed_kmh = self._read_f(physics_data, SPEED_OFFSET)
        speed_ms = speed_kmh / 3.6
        steer = self._read_f(physics_data, STEER_OFFSET)
        # Read gear and rpm from shared memory
        gear = self._read_i(physics_data, GEAR_OFFSET)
        rpm = self._read_i(physics_data, RPM_OFFSET)
        self.current_rpm = rpm

        # --- Velocity & acceleration ---
        vx = self._read_f(physics_data, VELOCITY_OFFSET)
        vy = self._read_f(physics_data, VELOCITY_OFFSET + 4)
        vz = self._read_f(physics_data, VELOCITY_OFFSET + 8)
        ax = self._read_f(physics_data, ACCEL_OFFSET)
        ay = self._read_f(physics_data, ACCEL_OFFSET + 4)
        az = self._read_f(physics_data, ACCEL_OFFSET + 8)

        accel_longitudinal = ax
        accel_lateral = az

        # --- Lap tracking ---
        elapsed = time.time() - self.start_time
        completed_laps = self._read_i(graphics_data, GRAPHICS_COMPLETED_LAPS)
        distance = self.distance_tracker.update(speed_ms, elapsed)
        lap_fraction = self.distance_tracker.get_lap_fraction()

        if completed_laps > self.last_completed_laps:
            self.distance_tracker.mark_lap_complete()
            self.last_completed_laps = completed_laps
            print(f"🏁 Lap {completed_laps} completed!")

        # --- Wheel & suspension ---
        def arr(offset):
            return [self._read_f(physics_data, offset + i * 4) for i in range(4)]

        wheel_slip = arr(WHEEL_SLIP_OFFSET)
        suspension = arr(SUSPENSION_TRAVEL_OFFSET)

        # --- Build state vector ---
        # Include gear and RPM (feature schema expects gear)
        state = build_state_from_memory(
            speed_ms=speed_ms,
            accel_longitudinal=accel_longitudinal,
            accel_lateral=accel_lateral,
            gear=gear,
            rpm=self.current_rpm,         # Only RPM
            lap_fraction=lap_fraction,
            vx=vx,
            vy=vy,
            vz=vz,
            wheel_slip_fl=wheel_slip[0],
            wheel_slip_fr=wheel_slip[1],
            wheel_slip_rl=wheel_slip[2],
            wheel_slip_rr=wheel_slip[3],
            suspension_travel_fl=suspension[0],
            suspension_travel_fr=suspension[1],
            suspension_travel_rl=suspension[2],
            suspension_travel_rr=suspension[3],
        )

        # --- Clip inputs to training range (OOD-safe) ---
        state_clipped = np.clip(state, self.norm_min, self.norm_max)

        # --- Validate state ---
        try:
            validate_state(state_clipped, "runtime")
        except AssertionError as e:
            print(f"\n❌ RUNTIME ERROR:\n{e}")
            self.shutdown()
            sys.exit(1)

        return state_clipped, speed_kmh

    # =========================================================================
    # GEAR SHIFT LOGIC
    # =========================================================================

    def update_gear(self):
        """Single point gear update per frame"""
        if self.current_rpm >= self.RPM_UPSHIFT and self.current_gear < self.MAX_GEAR:
            self.shift_up()
        elif self.current_rpm <= self.RPM_DOWNSHIFT and self.current_gear > self.MIN_GEAR:
            self.shift_down()

    def try_read_static_maxrpm(self):
        """Attempt to read maxRpm from static shared memory; fallback to norm_max."""
        try:
            size = 2048
            shm = mmap.mmap(-1, size, STATIC_SHM)
            data = shm.read(size)
            shm.close()
            candidates = []
            # scan for 4-byte ints in little-endian that look like maxRpm (4000-10000)
            for off in range(0, size - 4, 4):
                try:
                    val = struct.unpack_from('<i', data, off)[0]
                    if 4000 <= val <= 10000:
                        candidates.append(val)
                except Exception:
                    continue
            if candidates:
                maxrpm = max(candidates)
                self.RPM_UPSHIFT = int(0.9 * maxrpm)
                self.RPM_DOWNSHIFT = int(0.4 * maxrpm)
                print(f"   🔧 Static maxRpm detected: {maxrpm} -> UPSHIFT {self.RPM_UPSHIFT}, DOWNSHIFT {self.RPM_DOWNSHIFT}")
                return
        except Exception:
            pass

        # Fallback: use normalization max for rpm (index 5)
        try:
            if len(self.norm_max) > 5:
                maxrpm = int(self.norm_max[5])
                self.RPM_UPSHIFT = int(0.9 * maxrpm)
                self.RPM_DOWNSHIFT = int(0.4 * maxrpm)
                print(f"   🔧 Using normalization maxRpm: {maxrpm} -> UPSHIFT {self.RPM_UPSHIFT}, DOWNSHIFT {self.RPM_DOWNSHIFT}")
                return
        except Exception:
            pass

        print(f"   ⚠️ Could not read static maxRpm; using defaults UPSHIFT {self.RPM_UPSHIFT}, DOWNSHIFT {self.RPM_DOWNSHIFT}")

    # --- Helper readers for shared memory ---
    def _read_f(self, data: bytes, offset: int) -> float:
        """Read 4-byte float from shared memory block (little-endian)."""
        try:
            return struct.unpack_from('<f', data, offset)[0]
        except Exception:
            return 0.0

    def _read_i(self, data: bytes, offset: int) -> int:
        """Read 4-byte int from shared memory block (little-endian)."""
        try:
            return struct.unpack_from('<i', data, offset)[0]
        except Exception:
            return 0

    # --- Smoothing helper ---
    def smooth(self, gas: float, brake: float, steer: float):
        """Exponential smoothing for actuator commands."""
        a = SMOOTHING_ALPHA
        self.s_gas = a * gas + (1 - a) * self.s_gas
        self.s_brake = a * brake + (1 - a) * self.s_brake
        # Use separate alpha for steering to increase responsiveness
        a_s = getattr(self, 'steer_smooth_alpha', SMOOTHING_ALPHA)
        self.s_steer = a_s * steer + (1 - a_s) * self.s_steer
        return self.s_gas, self.s_brake, self.s_steer

    # --- Simple gear commands (internal state only) ---
    def shift_up(self):
        if self.current_gear < self.MAX_GEAR:
            self.current_gear += 1
            if self.shift_method == "bumper":
                try:
                    self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
                    self.gamepad.update()
                    time.sleep(0.12)
                    self.gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
                    self.gamepad.update()
                    print("   ▶️ Sent virtual RIGHT_SHOULDER press")
                except Exception:
                    print("   ⚠️ Failed to send virtual RIGHT_SHOULDER press")
            else:
                try:
                    self.gamepad.right_trigger_float(1.0)
                    self.gamepad.update()
                    time.sleep(0.12)
                    self.gamepad.right_trigger_float(0.0)
                    self.gamepad.update()
                    print("   ▶️ Sent virtual RIGHT_TRIGGER press")
                except Exception:
                    print("   ⚠️ Failed to send virtual RIGHT_TRIGGER press")
            print(f"⬆️  Shift up -> {self.current_gear}")

    def shift_down(self):
        if self.current_gear > self.MIN_GEAR:
            self.current_gear -= 1
            if self.shift_method == "bumper":
                try:
                    self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
                    self.gamepad.update()
                    time.sleep(0.12)
                    self.gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
                    self.gamepad.update()
                    print("   ▶️ Sent virtual LEFT_SHOULDER press")
                except Exception:
                    print("   ⚠️ Failed to send virtual LEFT_SHOULDER press")
            else:
                try:
                    self.gamepad.left_trigger_float(1.0)
                    self.gamepad.update()
                    time.sleep(0.12)
                    self.gamepad.left_trigger_float(0.0)
                    self.gamepad.update()
                    print("   ▶️ Sent virtual LEFT_TRIGGER press")
                except Exception:
                    print("   ⚠️ Failed to send virtual LEFT_TRIGGER press")
            print(f"⬇️  Shift down -> {self.current_gear}")

    # =========================================================================
    # ML INFERENCE
    # =========================================================================

    def predict(self, state: np.ndarray):
        if USE_TEMPORAL:
            # Warm-start: repeat last state if buffer not full
            if self.state_buffer is None:
                self.state_buffer = deque(maxlen=SEQUENCE_LENGTH)
            if len(self.state_buffer) == 0:
                for _ in range(SEQUENCE_LENGTH):
                    self.state_buffer.append(state)
            else:
                self.state_buffer.append(state)

            X_seq = np.stack(self.state_buffer)
            X_norm = (X_seq - self.norm_min) / self.norm_range
            X_flat = X_norm.reshape(1, -1)
            gas = float(self.model_gas.predict(X_flat)[0])
            brake = float(self.model_brake.predict(X_flat)[0])
            steer = float(self.model_steer.predict(X_flat)[0])
        else:
            x = ((state - self.norm_min) / self.norm_range).reshape(1, -1)
            gas = float(self.model_gas.predict(x)[0])
            brake = float(self.model_brake.predict(x)[0])
            steer = float(self.model_steer.predict(x)[0])

        # --- Clip outputs to valid domain ---
        return (
            np.clip(gas, 0.0, 1.0),
            np.clip(brake, 0.0, 1.0),
            np.clip(steer, -1.0, 1.0),
        )

    # =========================================================================
    # CONTROL LOGIC
    # =========================================================================

    def apply_controls(self, gas, brake, steer, speed_kmh):
        """Apply MLP outputs with smoothing, deadzone, and physics-aware clipping"""
        # Save raw outputs for logging
        gas_raw, brake_raw, steer_raw = gas, brake, steer

        # --- Apply gains ---
        gas *= GAS_GAIN
        brake *= BRAKE_GAIN
        steer *= self.steer_gain

        # --- Apply deadzones ---
        gas = 0.0 if gas < DEADZONE_GAS else gas
        brake = 0.0 if brake < DEADZONE_BRAKE else brake
        steer = 0.0 if abs(steer) < DEADZONE_STEER else steer

        # --- Apply speed-based constraints ---
        # If very low speed, ensure we provide sufficient throttle to move out of pits
        if speed_kmh < 2.0:
            gas = max(gas, 0.35)
            brake = 0.0

        # Avoid applying gas and brake simultaneously: prefer throttle when both requested
        if gas > 0.02 and brake > 0.02:
            # If throttle stronger than brake, cancel brake, otherwise cancel throttle
            if gas >= brake:
                brake = 0.0
            else:
                gas = 0.0

        # --- Steering dynamics constraint ---
        max_lat_acc = 9.0  # m/s^2 (example: limit lateral g)
        max_steer = max_lat_acc / max(0.1, speed_kmh / 3.6)
        steer = np.clip(steer, -max_steer, max_steer)

        # --- Speed-based steering attenuation (reduce steer at high speed)
        if speed_kmh > 80.0:
            attenuation = max(0.25, 1.0 - (speed_kmh - 80.0) / 200.0)
            steer *= attenuation

        # --- Clip to actuator limits ---
        gas = np.clip(gas, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)
        steer = np.clip(steer, -1.0, 1.0)

        # --- Steering rate limiter (clamp change per second) ---
        now = time.time()
        dt = max(1e-3, now - getattr(self, 'prev_time', now))
        max_delta = self.steer_max_rate * dt
        delta = steer - self.prev_sent_steer
        if abs(delta) > max_delta:
            steer = self.prev_sent_steer + np.sign(delta) * max_delta
        self.prev_sent_steer = steer

        # --- Send to gamepad ---
        self.gamepad.right_trigger_float(gas)
        self.gamepad.left_trigger_float(brake)
        self.gamepad.left_joystick_float(steer, 0.0)
        self.gamepad.update()

        return gas, brake, steer, gas_raw, brake_raw, steer_raw


    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    def run(self):
        print("🏁 AI DRIVER ACTIVE\n")
        last_log = time.time()
        last_frame_time = time.time()
        try:
            while True:
                t0 = time.time()
                last_frame_time = t0
                state, speed = self.read_telemetry()
                self.update_gear()
                # If deterministic controller enabled, use it (open-loop by lap_fraction)
                if self.deterministic_controller is not None:
                    try:
                        lap_frac = float(state[6])
                        speed_ms = float(state[0])
                    except Exception:
                        lap_frac = 0.0
                        speed_ms = 0.0
                    gas, brake, desired_steer, target_gear = self.deterministic_controller.get_action(lap_frac, speed_ms)

                    # gear: issue virtual shifts to match recorded gear
                    try:
                        if target_gear > 0:
                            while self.current_gear < target_gear:
                                self.shift_up()
                                time.sleep(0.06)
                            while self.current_gear > target_gear:
                                self.shift_down()
                                time.sleep(0.06)
                    except Exception:
                        pass

                    # Steering PD toward recorded steering (using smoothed steer state)
                    now = time.time()
                    dt = max(1e-3, now - self.prev_time)
                    err = float(desired_steer) - float(self.s_steer)
                    deriv = (err - self.prev_steer_err) / dt
                    steer_corr = self.steer_p * err + self.steer_d * deriv
                    steer = float(self.s_steer) + steer_corr
                    self.prev_steer_err = err
                    self.prev_time = now
                else:
                    gas, brake, steer = self.predict(state)
                # Exponential smoothing
                gas, brake, steer = self.smooth(gas, brake, steer)
                gas_final, brake_final, steer_final, gas_raw, brake_raw, steer_raw = \
                    self.apply_controls(gas, brake, steer, speed)
                self.frames += 1
                self.prediction_history.append([gas_raw, brake_raw, steer_raw])
                if time.time() - last_log > 2.0:
                    elapsed = time.time() - self.start_time
                    fps = self.frames / elapsed
                    buffer_status = f"Buf {len(self.state_buffer)}/{SEQUENCE_LENGTH} | " if USE_TEMPORAL else ""
                    if self.prediction_history:
                        pred_arr = np.array(self.prediction_history)
                        gas_avg = pred_arr[:, 0].mean()
                        brake_avg = pred_arr[:, 1].mean()
                        steer_avg = pred_arr[:, 2].mean()
                        print(
                            f"[{elapsed:6.1f}s] FPS {fps:5.1f} | {buffer_status}"
                            f"{speed:5.0f} km/h | RPM {self.current_rpm:.0f} | "
                            f"G {gas_final:.2f} (raw {gas_avg:.2f}) "
                            f"B {brake_final:.2f} (raw {brake_avg:.2f}) "
                            f"S {steer_final:+.2f} (raw {steer_avg:+.2f}) | "
                            f"Lap {state[6]:.2f}"
                        )
                        self.prediction_history.clear()
                    last_log = time.time()
                sleep = DT - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)
        except KeyboardInterrupt:
            self.shutdown()

    # =========================================================================
    # SHUTDOWN
    # =========================================================================

    def shutdown(self):
        self.gamepad.reset()
        self.gamepad.update()
        self.physics_shm.close()
        self.graphics_shm.close()
        elapsed = time.time() - self.start_time
        print(
            f"\n⏹️ Stopped | {self.frames:,} frames | "
            f"{self.frames / elapsed:.1f} FPS"
        )

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run AI Driver')
    parser.add_argument('--shift-method', type=str, choices=['bumper', 'trigger'], default='bumper',
                        help='Method to simulate gear shifts when using virtual gamepad')
    parser.add_argument('--selftest', action='store_true', help='Run virtual gamepad self-test on startup')
    parser.add_argument('--playback', type=str, choices=['none','actions'], default='none',
                        help='Playback mode: "actions" to replay ml_data/y_actions.npy')
    parser.add_argument('--playback-mode', type=str, choices=['none','actions','states'], default='none',
                        help='Playback mode: "actions" or "states" (states replays X_states + y_actions)')
    parser.add_argument('--track-length', type=float, default=None,
                        help='Override track length in meters (e.g., 6213 for Mount Panorama)')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic open-loop controller from ml_data')
    parser.add_argument('--det-data', type=str, default='ml_data', help='Path to deterministic data directory (default: ml_data)')
    parser.add_argument('--det-bins', type=int, default=100, help='Number of bins for lap_fraction mapping')
    parser.add_argument('--steer-gain', type=float, default=1.0, help='Multiplier for steering output')
    parser.add_argument('--steer-alpha', type=float, default=STEER_SMOOTH_ALPHA, help='Steering smoothing alpha (0-1, higher = more responsive)')
    parser.add_argument('--steer-max-rate', type=float, default=1.0, help='Maximum steering change per second')
    parser.add_argument('--steer-p', type=float, default=None, help='Steering PD P term override')
    parser.add_argument('--steer-d', type=float, default=None, help='Steering PD D term override')
    args = parser.parse_args()

    print("=" * 70)
    print("🤖 ASSETTO CORSA AI DRIVER")
    print("=" * 70)
    driver = AIDriver(shift_method=args.shift_method, self_test=args.selftest)
    driver.steer_gain = float(args.steer_gain)
    driver.steer_smooth_alpha = float(args.steer_alpha)
    driver.steer_max_rate = float(args.steer_max_rate)
    if args.steer_p is not None:
        driver.steer_p = float(args.steer_p)
    if args.steer_d is not None:
        driver.steer_d = float(args.steer_d)
    if args.steer_gain != 1.0:
        print(f"Using steer gain override: {driver.steer_gain}")
    if args.track_length is not None:
        driver.distance_tracker.track_length = float(args.track_length)
        print(f"Using track length override: {driver.distance_tracker.track_length} m")

    if args.deterministic:
        ok = driver.enable_deterministic(Path(args.det_data), bins=args.det_bins)
        if not ok:
            print("❌ Failed to enable deterministic controller; continuing without it.")

    if args.playback_mode == 'actions' or args.playback == 'actions':
        driver.run_playback_actions('ml_data/y_actions.npy', rate_hz=UPDATE_HZ)
    elif args.playback_mode == 'states':
        driver.run_playback_states('ml_data/X_states.npy', 'ml_data/y_actions.npy', rate_hz=UPDATE_HZ)
    else:
        driver.run()
