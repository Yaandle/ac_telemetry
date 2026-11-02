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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root: feature_schema.py lives there
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
        self.last_lap_time = None

    def update_frame(self, lap_fraction: float, speed_kmh: float, steer_angle: float, crashed: bool, lap_completed: bool):
        self.frames += 1
        self.max_lap_fraction = max(self.max_lap_fraction, lap_fraction)
        if crashed:
            self.crash_count += 1
        if lap_completed:
            self.lap_completed_count += 1
            current_time = time.time()
            if self.last_lap_time is not None:
                lap_time = current_time - self.last_lap_time
                self.lap_times.append(lap_time)
                print(f"⏱️  Lap {self.lap_completed_count} complete: {lap_time:.2f}s")
            self.last_lap_time = current_time
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
                if state_norm.ndim == 1:
                    state_norm = state_norm.reshape(1, -1)
                w, b = decode_individual(weights)
                predictions = (state_norm @ w.T) + b
                gas = float(np.clip(predictions[0, 0], 0, 1))
                brake = float(np.clip(predictions[0, 1], 0, 1))
                steer = float(np.clip(predictions[0, 2], -1, 1))
                return gas, brake, steer
            
            return predict, "Genetic Algorithm"
        else:
            raise FileNotFoundError(f"GA weights not found at {weights_path}")

    @staticmethod
    def _load_dql(dql_path: str):
        import torch
        model = torch.load(dql_path, map_location='cpu')
        model.eval()

        def predict(state_norm):
            state_t = torch.tensor(state_norm, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = model(state_t)[0]
            return (
                float(torch.sigmoid(action[0])),
                float(torch.sigmoid(action[1])),
                float(torch.tanh(action[2]))
            )

        return predict, "Deep Q-Learning"


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
        
        # Load normalization parameters
        self.norm_min = np.load(NORM_DIR / "normalization_min.npy")
        self.norm_max = np.load(NORM_DIR / "normalization_max.npy")
        self.norm_range = np.maximum(self.norm_max - self.norm_min, 1e-6)
        
        # Gamepad and shared memory
        self.gamepad = vg.VX360Gamepad()
        self.physics_shm = mmap.mmap(-1, READ_BLOCK_SIZE, PHYSICS_SHM)
        self.graphics_shm = mmap.mmap(-1, GRAPHICS_BLOCK_SIZE, GRAPHICS_SHM)
        
        # Control smoothing
        self.s_gas = 0.0
        self.s_brake = 0.0
        self.s_steer = 0.0
        
        # Tracking
        self.distance_tracker = DistanceTracker()
        self.last_completed_laps = 0
        self.start_time = time.time()
        self.frames = 0
        self.last_crash_frame = -1000  # For crash cooldown
        
        # Test mode
        self.test_mode = False
        self.test_laps_target = 0
        self.metrics_collector = MetricsCollector()
        
        # Policy state
        self.policy_enabled = True
        self.prev_state = None
        self.prev_lap_fraction = 0.0
        self.prev_steer = 0.0

        # GA-specific initialization
        if self.policy_name.lower().startswith('genetic') or 'ga' in self.policy_name.lower():
            self.ga_path = policy_kwargs.get('ga_path', 'ml_results/ga_model.pkl')
            self.ga_weights_path = self.ga_path.replace('ga_model.pkl', 'ga_weights.npy')
            self.state_dim = 17
            self.action_dim = 3
            
            # Load existing genome if available
            if Path(self.ga_weights_path).exists():
                self.current_genome = np.load(self.ga_weights_path)
                print(f"✅ Loaded existing GA genome from {self.ga_weights_path}")
            else:
                # Initialize genome with STRONG throttle bias for movement
                genome_size = self.action_dim * self.state_dim + self.action_dim
                
                # Weights: small random values
                weights = np.random.randn(self.action_dim * self.state_dim) * 0.05
                
                # Biases: CRITICAL - ensure positive throttle by default
                biases = np.array([
                    0.6,   # Gas bias: start with 60% throttle
                    0.0,   # Brake bias: no braking initially  
                    0.0    # Steer bias: centered steering
                ])
                
                self.current_genome = np.concatenate([weights, biases])
                print(f"🔄 Initialized new GA genome with throttle bias (gas=0.6)")
                
                # Save this baseline immediately so it's not lost
                MODEL_DIR.mkdir(exist_ok=True)
                np.save(self.ga_weights_path, self.current_genome)
                print(f"💾 Saved baseline genome to {self.ga_weights_path}")
            
            # GA evolution tracking
            self.best_reward = float('-inf')
            self.best_genome = self.current_genome.copy()

        # DQL-specific initialization
        if self.policy_name.lower().startswith('deep') or 'dql' in self.policy_name.lower():
            self.dql_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.dql_model = self.DQLNetwork(state_dim=17, action_dim=3).to(self.dql_device)
            self.dql_target_model = self.DQLNetwork(state_dim=17, action_dim=3).to(self.dql_device)
            self.dql_target_model.load_state_dict(self.dql_model.state_dict())
            self.dql_optimizer = optim.Adam(self.dql_model.parameters(), lr=1e-4)
            self.dql_loss_fn = nn.MSELoss()
            self.dql_gamma = 0.99
            self.dql_target_update_freq = 100
            self.dql_update_counter = 0

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
        
    def _decode_ga_genome(self):
        """Helper to decode GA genome into weights and bias"""
        split_point = self.action_dim * self.state_dim
        w = self.current_genome[:split_point].reshape(self.action_dim, self.state_dim)
        b = self.current_genome[split_point:]
        return w, b
        
    def store_transition(self, state, action, reward, next_state, done):
        if self.policy_name.lower().startswith('dql') or 'dql' in self.policy_name.lower():
            self.replay_buffer.append((state, action, reward, next_state, done))
            batch_size = 32
            if len(self.replay_buffer) >= batch_size:
                batch = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
                sampled = [self.replay_buffer[i] for i in batch]
                self.train_dql_batch(sampled)
        elif self.policy_name.lower().startswith('genetic') or 'ga' in self.policy_name.lower():
            self.episode.append((state, action, reward, next_state, done))
            max_episode_length = 500
            if len(self.episode) >= max_episode_length or done:
                self.evolve_ga([self.episode])
                self.episode = []

    def evolve_ga(self, episodes):
        """Hill-climbing GA evolution with adaptive mutation"""
        if not episodes or len(episodes[0]) == 0:
            return
        
        # Calculate episode reward
        total_reward = sum(r for _, _, r, _, _ in episodes[0])
        
        # Compare with best
        if total_reward > self.best_reward:
            self.best_reward = total_reward
            self.best_genome = self.current_genome.copy()
            print(f"🏆 New best GA genome! Reward: {total_reward:.2f}")
            
            # Save the best genome
            MODEL_DIR.mkdir(exist_ok=True)
            np.save(self.ga_weights_path, self.best_genome)
        else:
            # Revert to best and mutate
            print(f"⚠️  Reward {total_reward:.2f} < Best {self.best_reward:.2f}, reverting")
            self.current_genome = self.best_genome.copy()
        
        # Adaptive mutation - reduce mutation as performance improves
        if total_reward > 50:
            mutation_rate = 0.05  # Fine-tuning
        elif total_reward > 0:
            mutation_rate = 0.1   # Exploration
        else:
            mutation_rate = 0.2   # Heavy exploration when stuck
        
        # Apply mutation for next episode
        noise = np.random.randn(*self.current_genome.shape) * mutation_rate
        self.current_genome = self.current_genome + noise
        
        # Ensure gas bias stays positive (critical for movement)
        split_point = self.action_dim * self.state_dim
        gas_bias_idx = split_point  # First bias is gas
        if self.current_genome[gas_bias_idx] < 0.2:
            self.current_genome[gas_bias_idx] = 0.2  # Minimum 20% throttle bias

    def train_dql_batch(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.dql_device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32, device=self.dql_device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.dql_device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.dql_device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.dql_device).unsqueeze(1)

        # Q-learning with target network
        with torch.no_grad():
            next_actions = self.dql_target_model(next_states)
            next_q = torch.sigmoid(next_actions[:, :2]).mean(dim=1, keepdim=True)
            target_q = rewards + (1 - dones) * self.dql_gamma * next_q

        pred_actions = self.dql_model(states)
        pred_q = torch.sigmoid(pred_actions[:, :2]).mean(dim=1, keepdim=True)
        
        loss = self.dql_loss_fn(pred_q, target_q)

        self.dql_optimizer.zero_grad()
        loss.backward()
        self.dql_optimizer.step()
        
        # Update target network periodically
        self.dql_update_counter += 1
        if self.dql_update_counter % self.dql_target_update_freq == 0:
            self.dql_target_model.load_state_dict(self.dql_model.state_dict())

    def read_damage(self, physics_data: bytes) -> list:
        return [self._read_f(physics_data, DAMAGE_OFFSET + i * 4) for i in range(5)]

    def detect_crash(self, speed_kmh: float, gas: float, brake: float, steer: float, 
                     wheel_slip: list, suspension: list, damage: list) -> bool:
        # Crash cooldown - only detect once every 3 seconds
        if (self.frames - self.last_crash_frame) < 300:
            return False
        
        # Actual damage
        if any(d > 0.01 for d in damage):
            self.last_crash_frame = self.frames
            return True
        
        # Don't trigger crash detection in first 5 seconds (pit start grace period)
        if self.frames < 500:  # 5 seconds at 100Hz
            return False
        
        # Stuck detection - only after grace period and if applying throttle
        if speed_kmh < 5.0 and gas > 0.5 and self.frames > 1000:  # After 10 seconds
            self.last_crash_frame = self.frames
            return True
        
        # Extreme wheel slip
        slip_std = np.std(wheel_slip)
        if slip_std > 0.8:  # Increased threshold
            self.last_crash_frame = self.frames
            return True
        
        # Extreme suspension compression (hard impact)
        susp_std = np.std(suspension)
        if susp_std > 0.08:  # Increased threshold
            self.last_crash_frame = self.frames
            return True
        
        return False
    
    def read_telemetry(self) -> Tuple[np.ndarray, float]:
        """Read telemetry and return NORMALIZED state + speed"""
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
        
        # Clip to valid range
        state = np.clip(state, self.norm_min, self.norm_max)
        
        # NORMALIZE the state
        state_norm = (state - self.norm_min) / self.norm_range
        
        return state_norm, speed_kmh

    def predict(self, state_norm):
        """
        Predict gas, brake, steer from current policy.
        Works for GA, Linear, MLP, DQL.
        """
        # GA policy
        if self.policy_name.lower().startswith("genetic") or 'ga' in self.policy_name.lower():
            state_flat = state_norm.flatten()
            w, b = self._decode_ga_genome()
            pred = (state_flat @ w.T) + b  # (17,) @ (17,3) -> (3,)
            gas = float(np.clip(pred[0], 0, 1))
            brake = float(np.clip(pred[1], 0, 1))
            steer = float(np.clip(pred[2], -1, 1))
            
            # Debug output for first 10 frames to verify throttle output
            if self.frames < 10:
                print(f"   Frame {self.frames}: Raw pred=[{pred[0]:.3f}, {pred[1]:.3f}, {pred[2]:.3f}] -> gas={gas:.3f}")
            
            return gas, brake, steer

        # Linear / MLP policy
        elif callable(self.policy_predict):
            return self.policy_predict(state_norm)

        # DQL fallback
        else:
            return 0.5, 0.0, 0.0


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
        
        # CRITICAL FIX: Ensure minimum throttle at low speeds
        if speed_kmh < 5.0:  # Changed from 2.0 to 5.0 for better startup
            gas = max(gas, 0.5)  # Changed from 0.35 to 0.5 for stronger initial acceleration
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
        


    def run(self):
        """
        Main control loop with all enhancements applied
        """
        print(f"🚗 Starting AI Driver: {self.policy_name}")
        print(f"📊 Test mode: {self.test_mode} | Target laps: {self.test_laps_target}")
        print(f"🔄 Online learning: {self.online}")
        
        # Start the car and execute pit exit (NO DELAY - start immediately)
        try:
            state_norm, _ = self.read_telemetry()
            self.hot_lap_start(state_norm)
        except Exception as e:
            print(f"⚠️  Could not replay pit exit: {e}")
            print("   Continuing with live control...\n")
        
        try:
            while True:
                loop_start = time.time()
                
                # --------------------------------------------------
                # 1. READ TELEMETRY (now returns normalized state)
                # --------------------------------------------------
                try:
                    state_norm, speed_kmh = self.read_telemetry()
                    
                    self.physics_shm.seek(0)
                    physics_data = self.physics_shm.read(READ_BLOCK_SIZE)
                    wheel_slip = [self._read_f(physics_data, WHEEL_SLIP_OFFSET + i*4) for i in range(4)]
                    suspension = [self._read_f(physics_data, SUSPENSION_TRAVEL_OFFSET + i*4) for i in range(4)]
                    damage = self.read_damage(physics_data)
                except Exception as e:
                    print(f"❌ Telemetry read error: {e}")
                    time.sleep(DT)
                    continue

                # --------------------------------------------------
                # 2. POLICY INFERENCE
                # --------------------------------------------------
                if self.policy_enabled:
                    gas_raw, brake_raw, steer_raw = self.predict(state_norm)
                    gas_smooth, brake_smooth, steer_smooth = self.smooth(gas_raw, brake_raw, steer_raw)
                else:
                    gas_smooth, brake_smooth, steer_smooth = 0.0, 0.0, 0.0

                # --------------------------------------------------
                # 3. SEND CONTROLS TO SIM
                # --------------------------------------------------
                gas_final, brake_final, steer_final = self.apply_controls(
                    gas_smooth, brake_smooth, steer_smooth, speed_kmh
                )

                # --------------------------------------------------
                # 4. CRASH DETECTION
                # --------------------------------------------------
                crashed = self.detect_crash(
                    speed_kmh=speed_kmh,
                    gas=gas_final,
                    brake=brake_final,
                    steer=steer_final,
                    wheel_slip=wheel_slip,
                    suspension=suspension,
                    damage=damage
                )

                # --------------------------------------------------
                # 5. REWARD COMPUTATION
                # --------------------------------------------------
                lap_idx = POLICY_FEATURES.index('lap_fraction')
                lap_fraction = float(state_norm[lap_idx])
                
                lap_delta = lap_fraction - self.prev_lap_fraction
                if lap_delta < -0.5:
                    lap_delta = 0.0  # lap reset protection

                # Improved reward function
                reward = lap_delta * 100.0  # Progress is primary
                reward += (speed_kmh / 200.0) * 2.0  # Speed bonus (normalized)
                reward -= abs(steer_final - self.prev_steer) * 0.5  # Smoothness
                
                # Check for lap completion
                prev_completed_laps = self.last_completed_laps
                self.graphics_shm.seek(0)
                graphics_data = self.graphics_shm.read(GRAPHICS_BLOCK_SIZE)
                current_completed_laps = self._read_i(graphics_data, GRAPHICS_COMPLETED_LAPS)
                lap_completed = (current_completed_laps > prev_completed_laps)

                done = crashed
                if crashed:
                    reward -= 10.0
                    print(f"💥 Crash detected at {lap_fraction*100:.1f}%")

                # --------------------------------------------------
                # 6. METRICS COLLECTION
                # --------------------------------------------------
                if self.metrics_collector:
                    self.metrics_collector.update_frame(
                        lap_fraction, speed_kmh, steer_final, crashed, lap_completed
                    )

                # --------------------------------------------------
                # 7. ONLINE LEARNING (OPTIONAL)
                # --------------------------------------------------
                if self.online and self.prev_state is not None:
                    self.store_transition(
                        self.prev_state,
                        np.array([gas_final, brake_final, steer_final], dtype=np.float32),
                        reward,
                        state_norm,
                        done
                    )

                # --------------------------------------------------
                # 8. UPDATE PREVIOUS STATE
                # --------------------------------------------------
                self.prev_state = state_norm.copy()
                self.prev_lap_fraction = lap_fraction
                self.prev_steer = steer_final

                # --------------------------------------------------
                # 9. FRAME COUNTER AND TEST MODE EXIT
                # --------------------------------------------------
                self.frames += 1
                
                if self.test_mode and self.last_completed_laps >= self.test_laps_target:
                    print(f"\n✅ Test complete: {self.test_laps_target} laps finished")
                    break
                
                # Status update every 500 frames
                if self.frames % 500 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frames / elapsed
                    print(f"📈 Frame {self.frames:,} | {fps:.1f} FPS | Lap {lap_fraction*100:.1f}% | Speed {speed_kmh:.0f} km/h")

                # --------------------------------------------------
                # 10. TIMING CONTROL - MAINTAIN UPDATE_HZ
                # --------------------------------------------------
                elapsed = time.time() - loop_start
                sleep_time = DT - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif elapsed > DT * 1.5:
                    print(f"⚠️  Loop slow: {elapsed*1000:.1f}ms (target {DT*1000:.1f}ms)")
        
        except KeyboardInterrupt:
            print("\n⏸️  Interrupted by user")
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown()

    def shutdown(self):
        print("\n🛑 Shutting down...")
        
        self.gamepad.reset()
        self.gamepad.update()
        self.physics_shm.close()
        self.graphics_shm.close()
        
        # Save GA genome on shutdown if using GA
        if self.policy_name.lower().startswith('genetic') or 'ga' in self.policy_name.lower():
            MODEL_DIR.mkdir(exist_ok=True)
            np.save(self.ga_weights_path, self.best_genome)  # Save BEST genome
            print(f"💾 Best GA genome saved to {self.ga_weights_path} (reward: {self.best_reward:.2f})")
        
        elapsed = time.time() - self.start_time
        print(f"\n📊 Session Stats")
        print(f"   Frames: {self.frames:,}")
        print(f"   Duration: {elapsed:.1f}s")
        print(f"   FPS: {self.frames/elapsed:.1f}")
        
        if self.metrics_collector:
            metrics = self.metrics_collector.finalize()
            print(f"\n🏁 Performance Metrics")
            print(f"   Completion: {metrics['completion_pct']:.1f}%")
            print(f"   TCS Score: {metrics['tcs_score']:.3f}")
            print(f"   Crashes: {metrics['crash_count']}")
            print(f"   Laps Completed: {metrics['laps_completed']}")
            if metrics['avg_lap_time'] > 0:
                print(f"   Avg Lap Time: {metrics['avg_lap_time']:.2f}s")
            print(f"   Steering Smoothness: {metrics['steering_smoothness']:.3f}")
            
            # Save session log
            import json
            from datetime import datetime
            log_dir = Path("training_logs")
            log_dir.mkdir(exist_ok=True)
            
            session_data = {
                'timestamp': datetime.now().isoformat(),
                'policy': self.policy_name,
                'frames': self.frames,
                'duration': elapsed,
                'fps': self.frames/elapsed,
                'metrics': metrics,
                'best_reward': getattr(self, 'best_reward', None)
            }
            
            log_file = log_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(log_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            print(f"\n💾 Session logged to {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AI Driver - Enhanced Unified Policy Execution')
    parser.add_argument('--policy', type=str, choices=['linear', 'mlp', 'ga', 'dql'], default='mlp',
                        help='Policy type to use')
    parser.add_argument('--model-suffix', type=str, default='mlp_256_128_64',
                        help='Model suffix for MLP models')
    parser.add_argument('--ga-path', type=str, default='ml_results/ga_model.pkl',
                        help='Path to GA model')
    parser.add_argument('--dql-path', type=str, default='ml_results/dql_model.pt',
                        help='Path to DQL model')
    parser.add_argument('--online', action='store_true',
                        help='Enable online learning (DQL/GA)')
    parser.add_argument('--test-laps', type=int, default=0,
                        help='Number of laps to complete in test mode (0 = run indefinitely)')

    args = parser.parse_args()
    
    policy_kwargs = {
        'model_suffix': args.model_suffix,
        'ga_path': args.ga_path,
        'dql_path': args.dql_path
    }
    
    print("=" * 60)
    print("🏎️  AI DRIVER - BATHURST EDITION")
    print("=" * 60)
    
    driver = AIDriver(policy_type=args.policy, policy_kwargs=policy_kwargs)
    
    if args.test_laps > 0:
        driver.test_mode = True
        driver.test_laps_target = args.test_laps
        print(f"🎯 Test mode: Will complete {args.test_laps} laps then exit")
    
    driver.run()