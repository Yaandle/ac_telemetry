#!/usr/bin/env python3
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
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from feature_schema import POLICY_FEATURES, build_state_from_memory, validate_state
except ImportError:
    print("❌ Missing feature_schema.py")
    sys.exit(1)

MODEL_DIR = Path("ml_results")
NORM_DIR = Path("ml_data")

PHYSICS_SHM = "Local\\acpmf_physics"
GRAPHICS_SHM = "Local\\acpmf_graphics"

READ_BLOCK_SIZE = 512
GRAPHICS_BLOCK_SIZE = 256
UPDATE_HZ = 100
DT = 1.0 / UPDATE_HZ

SMOOTHING_ALPHA = 0.30
DEADZONE_GAS = 0.01
DEADZONE_BRAKE = 0.01
DEADZONE_STEER = 0.001

SPEED_OFFSET = 28
RPM_OFFSET = 20
GEAR_OFFSET = 16
VELOCITY_OFFSET = 32
ACCEL_OFFSET = 44
WHEEL_SLIP_OFFSET = 56
SUSPENSION_TRAVEL_OFFSET = 184
GRAPHICS_COMPLETED_LAPS = 132
SUSPENSION_TRAVEL_OFFSET = 184
DAMAGE_OFFSET = 200

class DistanceTracker:
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
        if self.track_length > 0:
            return min(self.lap_distance / self.track_length, 0.999)
        return 0.0

    def mark_lap_complete(self):
        if self.lap_distance > 100:
            if self.track_length == 5000.0:
                self.track_length = self.lap_distance
            else:
                self.track_length = 0.9 * self.track_length + 0.1 * self.lap_distance
        self.lap_distance = 0.0

class MetricsCollector:
    def __init__(self):
        self.start_time = time.time()
        self.frames = 0
        self.max_lap_fraction = 0.0
        self.crash_count = 0
        self.lap_times = []
        self.steering_changes = []
        self.prev_steer = 0.0
        self.lap_completed_count = 0

    def update_frame(self, lap_fraction: float, speed_kmh: float, steer_angle: float, crashed: bool, lap_completed: bool):
        self.frames += 1
        self.max_lap_fraction = max(self.max_lap_fraction, lap_fraction)
        if crashed:
            self.crash_count += 1
        if lap_completed:
            self.lap_completed_count += 1
            lap_time = time.time() - self.start_time
            self.lap_times.append(lap_time)
        steer_change = abs(steer_angle - self.prev_steer)
        self.steering_changes.append(steer_change)
        self.prev_steer = steer_angle

    def finalize(self):
        elapsed = time.time() - self.start_time
        completion_pct = self.max_lap_fraction * 100.0
        crash_penalty = self.crash_count * 0.1
        if len(self.steering_changes) > 0:
            avg_jerk = np.mean(self.steering_changes)
            smoothness_score = max(0, 1.0 - avg_jerk * 10)
        else:
            smoothness_score = 0.0
        tcs = (completion_pct / 100.0) - crash_penalty + (smoothness_score * 0.1)
        tcs = max(0.0, min(1.0, tcs))
        avg_lap_time = np.mean(self.lap_times) if self.lap_times else 0.0
        return {
            'completion_pct': completion_pct,
            'tcs_score': tcs,
            'crash_count': self.crash_count,
            'avg_lap_time': avg_lap_time,
            'steering_smoothness': smoothness_score,
            'laps_completed': self.lap_completed_count
        }

class PolicyLoader:
    @staticmethod
    def load(policy_type: str, model_dir: Path, **kwargs):
        if policy_type == 'linear':
            return PolicyLoader._load_linear(model_dir)
        elif policy_type == 'mlp':
            model_suffix = kwargs.get('model_suffix', 'mlp_256_128_64')
            return PolicyLoader._load_mlp(model_dir, model_suffix)
        elif policy_type == 'ga':
            ga_path = kwargs.get('ga_path', 'ml_results/ga_model.pkl')
            return PolicyLoader._load_ga(ga_path)
        elif policy_type == 'dql':
            dql_path = kwargs.get('dql_path', 'ml_results/dql_model.pt')
            return PolicyLoader._load_dql(dql_path)
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")

    @staticmethod
    def _load_linear(model_dir: Path):
        gas = joblib.load(model_dir / "linear_regression_gas.pkl")
        brake = joblib.load(model_dir / "linear_regression_brake.pkl")
        steer = joblib.load(model_dir / "linear_regression_steer.pkl")
        def predict(state_norm):
            x = state_norm.reshape(1, -1)
            return float(gas.predict(x)[0]), float(brake.predict(x)[0]), float(steer.predict(x)[0])
        return predict, "Linear Regression"

    @staticmethod
    def _load_mlp(model_dir: Path, model_suffix: str):
        gas = joblib.load(model_dir / f"{model_suffix}_gas.pkl")
        brake = joblib.load(model_dir / f"{model_suffix}_brake.pkl")
        steer = joblib.load(model_dir / f"{model_suffix}_steer.pkl")
        def predict(state_norm):
            x = state_norm.reshape(1, -1)
            return float(gas.predict(x)[0]), float(brake.predict(x)[0]), float(steer.predict(x)[0])
        return predict, f"MLP ({model_suffix})"

    @staticmethod
    def _load_ga(ga_path: str):
        """Load GA model from weights"""
        # Try loading the weights directly (more portable)
        weights_path = ga_path.replace('ga_model.pkl', 'ga_weights.npy')
        
        if Path(weights_path).exists():
            print(f"  Loading GA weights from: {weights_path}")
            weights = np.load(weights_path)
            state_dim = 17
            action_dim = 3
            
            def decode_individual(individual):
                """Decode flattened individual into weights and bias"""
                split_point = action_dim * state_dim
                w = individual[:split_point].reshape(action_dim, state_dim)
                b = individual[split_point:]
                return w, b
            
            def predict(state_norm):
                # Reshape if needed
                if state_norm.ndim == 1:
                    state_norm = state_norm.reshape(1, -1)
                
                # Decode weights
                w, b = decode_individual(weights)
                
                # Predict: action = state @ W.T + b
                predictions = (state_norm @ w.T) + b
                
                # Clip to valid ranges
                gas = float(np.clip(predictions[0, 0], 0, 1))
                brake = float(np.clip(predictions[0, 1], 0, 1))
                steer = float(np.clip(predictions[0, 2], -1, 1))
                
                return gas, brake, steer
            
            return predict, "Genetic Algorithm"
        else:
            raise FileNotFoundError(f"GA weights not found at {weights_path}")

    @staticmethod
    def _load_dql(dql_path: str):
        try:
            import torch
            dql_data = torch.load(dql_path)
            def predict(state_norm):
                return 0.5, 0.0, 0.0
            return predict, "Deep Q-Learning"
        except ImportError:
            print("❌ PyTorch not available for DQL")
            raise

class AIDriver:
    class DQLNetwork(nn.Module):
        def __init__(self, state_dim: int, action_dim: int, hidden=[128, 128]):
            super().__init__()
            layers = []
            last_dim = state_dim
            for h in hidden:
                layers.append(nn.Linear(last_dim, h))
                layers.append(nn.ReLU())
                last_dim = h
            layers.append(nn.Linear(last_dim, action_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)
    
    def __init__(self, policy_type: str, policy_kwargs: dict = None):
        
        policy_kwargs = policy_kwargs or {}
        self.policy_predict, self.policy_name = PolicyLoader.load(policy_type, MODEL_DIR, **policy_kwargs)
        self.online = '--online' in sys.argv
        self.replay_buffer = deque(maxlen=5000)
        self.episode = []
        self.norm_min = np.load(NORM_DIR / "normalization_min.npy")
        self.norm_max = np.load(NORM_DIR / "normalization_max.npy")
        self.norm_range = np.maximum(self.norm_max - self.norm_min, 1e-6)
        self.gamepad = vg.VX360Gamepad()
        self.physics_shm = mmap.mmap(-1, READ_BLOCK_SIZE, PHYSICS_SHM)
        self.graphics_shm = mmap.mmap(-1, GRAPHICS_BLOCK_SIZE, GRAPHICS_SHM)
        self.s_gas = 0.0
        self.s_brake = 0.0
        self.s_steer = 0.0
        self.distance_tracker = DistanceTracker()
        self.last_completed_laps = 0
        self.start_time = time.time()
        self.frames = 0
        self.test_mode = False
        self.test_laps_target = 0
        self.metrics_collector = None
        self.policy_enabled = False
        
        # GA-specific initialization
        if self.policy_name.lower().startswith('genetic') or self.policy_name.lower().startswith('ga'):
            self.ga_path = policy_kwargs.get('ga_path', 'ml_results/ga_model.pkl')
            self.ga_weights_path = self.ga_path.replace('ga_model.pkl', 'ga_weights.npy')
            self.state_dim = 17
            self.action_dim = 3
            
            # Load existing genome if available
            if Path(self.ga_weights_path).exists():
                self.current_genome = np.load(self.ga_weights_path)
                print(f"✅ Loaded existing GA genome from {self.ga_weights_path}")
            else:
                # Initialize random genome
                genome_size = self.action_dim * self.state_dim + self.action_dim
                self.current_genome = np.random.randn(genome_size) * 0.1
                print(f"🔄 Initialized new random GA genome")

    def _read_f(self, data: bytes, offset: int) -> float:
        try:
            return struct.unpack_from('<f', data, offset)[0]
        except:
            return 0.0

    def _read_i(self, data: bytes, offset: int) -> int:
        try:
            return struct.unpack_from('<i', data, offset)[0]
        except:
            return 0
        
    def store_transition(self, state, action, reward, next_state, done):
        if self.policy_name.lower().startswith('dql'):
            self.replay_buffer.append((state, action, reward, next_state, done))
            batch_size = 32
            if len(self.replay_buffer) >= batch_size:
                batch = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
                sampled = [self.replay_buffer[i] for i in batch]
                self.train_dql_batch(sampled)
        elif self.policy_name.lower().startswith('genetic') or self.policy_name.lower().startswith('ga'):
            self.episode.append((state, action, reward, next_state, done))
            max_episode_length = 500
            if len(self.episode) >= max_episode_length:
                self.evolve_ga([self.episode])
                self.episode = []

    def evolve_ga(self, episodes):
        """Simple GA evolution: mutate current genome based on episode performance"""
        if not episodes or len(episodes[0]) == 0:
            return
        
        # Calculate episode reward
        total_reward = sum(r for _, _, r, _, _ in episodes[0])
        
        # Simple mutation strategy: if performance is poor, mutate more aggressively
        mutation_rate = 0.1 if total_reward > 0 else 0.3
        noise = np.random.randn(*self.current_genome.shape) * mutation_rate
        
        # Create mutated genome
        new_genome = self.current_genome + noise
        
        # For now, always accept the mutation (in a full GA, you'd compare fitness)
        # In production, you'd maintain a population and select the best
        self.current_genome = new_genome
        
        # Save the updated genome
        MODEL_DIR.mkdir(exist_ok=True)
        np.save(self.ga_weights_path, self.current_genome)
        print(f"💾 Saved evolved GA genome (reward: {total_reward:.2f}) to {self.ga_weights_path}")

    def train_dql_batch(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32, device=self.dql_device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.dql_device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.dql_device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.dql_device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.dql_device).unsqueeze(1)

        q_values = self.dql_model(states)

        with torch.no_grad():
            q_next = self.dql_model(next_states)
            q_target = rewards + self.dql_gamma * torch.max(q_next, dim=1, keepdim=True)[0] * (1 - dones)

        loss = self.dql_loss_fn(q_values, q_target)

        self.dql_optimizer.zero_grad()
        loss.backward()
        self.dql_optimizer.step()

    def read_damage(self, physics_data: bytes) -> list:
        return [self._read_f(physics_data, DAMAGE_OFFSET + i * 4) for i in range(5)]

    def detect_crash(self, speed_kmh: float, gas: float, brake: float, steer: float, 
                     wheel_slip: list, suspension: list, damage: list) -> bool:
        if any(d > 0.01 for d in damage):
            return True
        
        if speed_kmh < 5.0 and gas > 0.4:
            return True
        
        slip_std = np.std(wheel_slip)
        if slip_std > 0.6:
            return True
        
        susp_std = np.std(suspension)
        if susp_std > 0.05:
            return True
        
        return False
    
    def read_telemetry(self) -> Tuple[np.ndarray, float]:
        self.physics_shm.seek(0)
        physics_data = self.physics_shm.read(READ_BLOCK_SIZE)
        self.graphics_shm.seek(0)
        graphics_data = self.graphics_shm.read(GRAPHICS_BLOCK_SIZE)
        speed_kmh = self._read_f(physics_data, SPEED_OFFSET)
        speed_ms = speed_kmh / 3.6
        rpm = self._read_i(physics_data, RPM_OFFSET)
        gear = self._read_i(physics_data, GEAR_OFFSET)
        vx = self._read_f(physics_data, VELOCITY_OFFSET)
        vy = self._read_f(physics_data, VELOCITY_OFFSET + 4)
        vz = self._read_f(physics_data, VELOCITY_OFFSET + 8)
        ax = self._read_f(physics_data, ACCEL_OFFSET)
        az = self._read_f(physics_data, ACCEL_OFFSET + 8)
        elapsed = time.time() - self.start_time
        completed_laps = self._read_i(graphics_data, GRAPHICS_COMPLETED_LAPS)
        distance = self.distance_tracker.update(speed_ms, elapsed)
        lap_fraction = self.distance_tracker.get_lap_fraction()
        if completed_laps > self.last_completed_laps:
            self.distance_tracker.mark_lap_complete()
            self.last_completed_laps = completed_laps
        wheel_slip = [self._read_f(physics_data, WHEEL_SLIP_OFFSET + i*4) for i in range(4)]
        suspension = [self._read_f(physics_data, SUSPENSION_TRAVEL_OFFSET + i*4) for i in range(4)]
        state = build_state_from_memory(
            speed_ms=speed_ms, accel_longitudinal=ax, accel_lateral=az,
            gear=gear, rpm=rpm, lap_fraction=lap_fraction,
            vx=vx, vy=vy, vz=vz,
            wheel_slip_fl=wheel_slip[0], wheel_slip_fr=wheel_slip[1],
            wheel_slip_rl=wheel_slip[2], wheel_slip_rr=wheel_slip[3],
            suspension_travel_fl=suspension[0], suspension_travel_fr=suspension[1],
            suspension_travel_rl=suspension[2], suspension_travel_rr=suspension[3]
        )
        state = np.clip(state, self.norm_min, self.norm_max)
        return state, speed_kmh

    def predict(self, state: np.ndarray) -> Tuple[float, float, float]:
        state_norm = (state - self.norm_min) / self.norm_range
        gas, brake, steer = self.policy_predict(state_norm)
        return np.clip(gas, 0.0, 1.0), np.clip(brake, 0.0, 1.0), np.clip(steer, -1.0, 1.0)

    def smooth(self, gas: float, brake: float, steer: float):
        a = SMOOTHING_ALPHA
        self.s_gas = a * gas + (1 - a) * self.s_gas
        self.s_brake = a * brake + (1 - a) * self.s_brake
        self.s_steer = a * steer + (1 - a) * self.s_steer
        return self.s_gas, self.s_brake, self.s_steer

    def apply_controls(self, gas: float, brake: float, steer: float, speed_kmh: float):
        gas = 0.0 if gas < DEADZONE_GAS else gas
        brake = 0.0 if brake < DEADZONE_BRAKE else brake
        steer = 0.0 if abs(steer) < DEADZONE_STEER else steer
        if speed_kmh < 2.0:
            gas = max(gas, 0.35)
            brake = 0.0
        if gas > 0.02 and brake > 0.02:
            if gas >= brake:
                brake = 0.0
            else:
                gas = 0.0
        if brake > 0.1:
            gas = 0.0
        if speed_kmh > 80.0:
            attenuation = max(0.25, 1.0 - (speed_kmh - 80.0) / 200.0)
            steer *= attenuation
        if speed_kmh > 200:
            steer = np.clip(steer, -0.4, 0.4)
        if speed_kmh > 250:
            steer = np.clip(steer, -0.25, 0.25)
        gas = np.clip(gas, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)
        steer = np.clip(steer, -1.0, 1.0)
        self.gamepad.right_trigger_float(gas)
        self.gamepad.left_trigger_float(brake)
        self.gamepad.left_joystick_float(steer, 0.0)
        self.gamepad.update()
        return gas, brake, steer

    
    def hot_lap_start(self, initial_state: np.ndarray):
        """Replay pit exit sequence from CSV recording"""
        print("🏁 Pit start - replaying from CSV")
        
        pad = self.gamepad
        
        # Car ignition
        pad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        pad.update()
        time.sleep(0.05)
        pad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        pad.update()

        
        # Launch with clutch
        pad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        pad.right_trigger_float(1)
        pad.update()
        time.sleep(0.3)
        
        # Shift to 2nd gear
        pad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
        pad.update()
        time.sleep(0.075)
        pad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
        pad.update()
        
        # Load CSV and replay
        try:
            import pandas as pd
            csv_path = "telemetry_logs/acc_session_20260121_040734_gas_steer.csv"
            df = pd.read_csv(csv_path)
            
            if 'gas' not in df.columns or 'steer' not in df.columns:
                print("❌ CSV missing required columns")
                return
            
            print(f"▶️  Replaying {len(df)} frames...")
            
            # Replay frame-by-frame at 100Hz
            for idx, row in df.iterrows():
                frame_start = time.time()
                
                gas = float(row['gas'])
                steer = float(row['steer'])
                
                # Apply controls (keep A button held from earlier)
                pad.right_trigger_float(gas)
                pad.left_joystick_float(steer, 0.0)
                pad.update()
                
                # Maintain 100Hz timing
                elapsed = time.time() - frame_start
                sleep_time = DT - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # Release all controls
            pad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
            pad.right_trigger_float(0.0)
            pad.left_joystick_float(0.0, 0.0)
            pad.update()
            
            print("✅ CSV replay complete")
            
        except FileNotFoundError:
            print(f"❌ CSV file not found: {csv_path}")
        except Exception as e:
            print(f"❌ Error replaying CSV: {e}")

    def run(self):
        print("🚦 Executing hot-lap start")
        state, _ = self.read_telemetry()
        self.hot_lap_start(state)

        stabilize_time = 2.0
        start_time = time.time()
        while True:
            state, speed_kmh = self.read_telemetry()

            vy_idx = POLICY_FEATURES.index('vy')
            lateral_velocity = abs(state[vy_idx])

            if speed_kmh < 30.0 or lateral_velocity > 0.1:
                gas = 0.35
                brake = 0.0
                steer = 0.0
            else:
                break

            self.apply_controls(gas, brake, steer, speed_kmh)
            time.sleep(DT)

            if (time.time() - start_time) > stabilize_time:
                break

        print("🟢 Policy enabled (hot-lap start)")
        self.policy_enabled = True

        if self.test_mode:
            self.metrics_collector = MetricsCollector()

        last_log = time.time()

        try:
            while True:
                t0 = time.time()
                state, speed_kmh = self.read_telemetry()

                if not self.policy_enabled and speed_kmh > 25.0:
                    print("🟢 Policy enabled mid-lap")
                    self.policy_enabled = True
                elif not self.policy_enabled:
                    self.gamepad.update()
                    time.sleep(DT)
                    continue

                gas, brake, steer = self.predict(state)
                gas, brake, steer = self.smooth(gas, brake, steer)
                gas_final, brake_final, steer_final = self.apply_controls(
                    gas, brake, steer, speed_kmh
                )

                self.frames += 1

                self.physics_shm.seek(0)
                physics_data = self.physics_shm.read(READ_BLOCK_SIZE)
                wheel_slip = [self._read_f(physics_data, WHEEL_SLIP_OFFSET + i*4) for i in range(4)]
                suspension = [self._read_f(physics_data, SUSPENSION_TRAVEL_OFFSET + i*4) for i in range(4)]
                damage = self.read_damage(physics_data)

                crashed = False
                if (time.time() - start_time) > 5.0:
                    crashed = self.detect_crash(
                        speed_kmh=speed_kmh,
                        gas=gas_final,
                        brake=brake_final,
                        steer=steer_final,
                        wheel_slip=wheel_slip,
                        suspension=suspension,
                        damage=damage
                    )

                if self.metrics_collector:
                    lap_frac_idx = POLICY_FEATURES.index('lap_fraction')
                    lap_fraction = float(state[lap_frac_idx])
                    lap_completed = self.last_completed_laps > 0 and lap_fraction < 0.1
                    self.metrics_collector.update_frame(
                        lap_fraction=lap_fraction,
                        speed_kmh=speed_kmh,
                        steer_angle=steer_final,
                        crashed=crashed,
                        lap_completed=lap_completed
                    )

                    if self.last_completed_laps >= self.test_laps_target:
                        break

                if time.time() - last_log > 2.0:
                    elapsed = time.time() - self.start_time
                    fps = self.frames / elapsed
                    damage_str = f"Dmg[{damage[0]:.2f},{damage[1]:.2f},{damage[2]:.2f},{damage[3]:.2f},{damage[4]:.2f}]"
                    print(
                        f"[{elapsed:6.1f}s] FPS {fps:5.1f} | "
                        f"{speed_kmh:5.0f} km/h | "
                        f"G {gas_final:.2f} B {brake_final:.2f} S {steer_final:+.2f} | "
                        f"{damage_str} | "
                        f"{'💥 CRASH' if crashed else '✅ Safe'}"
                    )
                    last_log = time.time()

                sleep_time = DT - (time.time() - t0)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        self.gamepad.reset()
        self.gamepad.update()
        self.physics_shm.close()
        self.graphics_shm.close()
        
        # Save GA genome on shutdown if using GA
        if self.policy_name.lower().startswith('genetic') or self.policy_name.lower().startswith('ga'):
            MODEL_DIR.mkdir(exist_ok=True)
            np.save(self.ga_weights_path, self.current_genome)
            print(f"💾 Final GA genome saved to {self.ga_weights_path}")
        
        elapsed = time.time() - self.start_time
        print(f"\nStopped | {self.frames:,} frames | {self.frames/elapsed:.1f} FPS")
        if self.metrics_collector:
            metrics = self.metrics_collector.finalize()
            print(f"Completion: {metrics['completion_pct']:.1f}% | TCS: {metrics['tcs_score']:.3f} | Crashes: {metrics['crash_count']} | Avg Lap: {metrics['avg_lap_time']:.1f}s | Smoothness: {metrics['steering_smoothness']:.3f} | Laps Done: {metrics['laps_completed']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AI Driver - Unified Policy Execution')
    parser.add_argument('--policy', type=str, choices=['linear', 'mlp', 'ga', 'dql'], default='mlp')
    parser.add_argument('--model-suffix', type=str, default='mlp_256_128_64')
    parser.add_argument('--ga-path', type=str, default='ml_results/ga_model.pkl')
    parser.add_argument('--dql-path', type=str, default='ml_results/dql_model.pt')
    parser.add_argument('--online', action='store_true', help='Enable online DQL training')

    parser.add_argument('--test-laps', type=int, default=0)
    args = parser.parse_args()
    policy_kwargs = {'model_suffix': args.model_suffix, 'ga_path': args.ga_path, 'dql_path': args.dql_path}
    driver = AIDriver(policy_type=args.policy, policy_kwargs=policy_kwargs)
    if args.test_laps > 0:
        driver.test_mode = True
        driver.test_laps_target = args.test_laps
    driver.run()
