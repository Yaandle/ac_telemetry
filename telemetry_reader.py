#!/usr/bin/env python3
"""
Assetto Corsa Competizione Telemetry Reader - Robotics-Ready Data Ingestion
High-frequency telemetry capture with derived features and real-time logging
Based on ACC Shared Memory v1.8.12 Documentation
"""

import mmap
import struct
import time
import csv
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import deque
from typing import Optional

# ============================================================================
# SHARED MEMORY CONFIGURATION - ACC v1.8.12
# ============================================================================

PHYSICS_SHM = "Local\\acpmf_physics"
GRAPHICS_SHM = "Local\\acpmf_graphics"


# struct SPageFilePhysics {
#   int packetId;                // 0      (4 bytes)
#   float gas;                   // 4      (4 bytes)
#   float brake;                 // 8      (4 bytes)
#   float fuel;                  // 12     (4 bytes)
#   int gear;                    // 16     (4 bytes)
#   int rpm;                     // 20     (4 bytes)
#   float steerAngle;            // 24     (4 bytes)
#   float speedKmh;              // 28     (4 bytes)
#   float velocity[3];           // 32-43  (12 bytes) 
#   float accG[3];               // 44-55  (12 bytes) 
#   float wheelSlip[4];          // 56-71  (16 bytes)

#   float wheelPressure[4];      // 88-103 (16 bytes)
#   float wheelAngularSpeed[4];  // 104-119 (16 bytes)
#   float tyreWear[4];           // 120-135 (16 bytes)
#   float tyreDirtyLevel[4];     // 136-151 (16 bytes)
#   float tyreCoreTemperature[4];// 152-167 (16 bytes)
#   float camberRAD[4];          // 168-183 (16 bytes)
#   float suspensionTravel[4];   // 184-199 (16 bytes)
# }


PACKET_ID_OFFSET = 0
GAS_OFFSET = 4
BRAKE_OFFSET = 8
FUEL_OFFSET = 12
GEAR_OFFSET = 16
RPM_OFFSET = 20
STEER_OFFSET = 24
SPEED_OFFSET = 28
VELOCITY_OFFSET = 32        # 3x Float - velocity[3]
ACCEL_OFFSET = 44           # 3x Float - accG[3]
WHEEL_SLIP_OFFSET = 56      # 4x Float - wheelSlip[4]
SLIP_ANGLE_OFFSET = 72
SUSPENSION_TRAVEL_OFFSET = 184  # 4x Float - suspensionTravel[4]

# Graphics Memory Offsets
GRAPHICS_CURRENT_TIME = 12
GRAPHICS_COMPLETED_LAPS = 132

# Configuration
SAMPLING_HZ = 100
DEADZONE_STEER = 0.001
DEADZONE_THROTTLE = 0.001
DEADZONE_BRAKE = 0.001
SMOOTHING_WINDOW = 5        # Samples for moving average
READ_BLOCK_SIZE = 512       # Larger block for extended data
GRAPHICS_BLOCK_SIZE = 256

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TelemetryFrame:
    """Single telemetry sample with derived features"""
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

    
    # Velocity & Acceleration (m/s, G-forces)
    vx: float
    vy: float
    vz: float
    ax: float  
    ay: float  
    az: float  
    
    # Derived kinematics
    accel_longitudinal: float  
    accel_lateral: float       
    
    # Control Inputs (actions)
    gas: float
    brake: float
    steer: float
    abs_steer: float
    
    # Smoothed controls
    gas_smooth: float
    brake_smooth: float
    steer_smooth: float
    
    # Position & Distance
    distance_m: float
    lap_fraction: float
    
    # Wheel Data (FL, FR, RL, RR)
    wheel_slip_fl: float
    wheel_slip_fr: float
    wheel_slip_rl: float
    wheel_slip_rr: float
    
    slip_angle_fl: float   # radians
    slip_angle_fr: float
    slip_angle_rl: float
    slip_angle_rr: float

    suspension_travel_fl: float  # meters
    suspension_travel_fr: float  # meters
    suspension_travel_rl: float  # meters
    suspension_travel_rr: float  # meters
    
    # Additional
    fuel_kg: float

class SmoothingFilter:
    """Moving average filter for control inputs"""
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
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

class DistanceTracker:
    """Track distance traveled via trapezoidal integration"""
    def __init__(self):
        self.total_distance = 0.0
        self.prev_speed_ms = 0.0
        self.prev_time = 0.0
        self.track_length = 0.0
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
        if self.track_length > 0 and self.lap_distance > 0:
            fraction = self.lap_distance / self.track_length
            return min(fraction, 0.999)  # Cap at 0.999 to avoid overflow
        return 0.0
    
    def mark_lap_complete(self):
        """Update track length estimate on lap completion"""
        if self.lap_distance > 100:  # Sanity check (at least 100m)
            if self.track_length == 0:
                self.track_length = self.lap_distance
            else:
                # Exponential moving average
                self.track_length = 0.9 * self.track_length + 0.1 * self.lap_distance
        
        # Reset for next lap
        self.lap_distance = 0.0



class SessionTracker:
    """Track session statistics and lap boundaries"""
    def __init__(self):
        self.start_time = time.time()
        self.lap_count = 0
        self.sample_count = 0
        
        # Statistics
        self.max_speed = 0.0
        self.max_gas = 0.0
        self.max_brake = 0.0
        self.max_steer = 0.0
        self.max_accel_long = 0.0
        self.max_accel_lat = 0.0
        self.max_rpm = 0

        # Lap tracking
        self.last_lap_number = 0
        
    def update(self, frame: TelemetryFrame):
        self.sample_count += 1
        
        # Update maxima
        self.max_speed = max(self.max_speed, frame.speed_kmh)
        self.max_gas = max(self.max_gas, frame.gas)
        self.max_brake = max(self.max_brake, frame.brake)
        self.max_steer = max(self.max_steer, frame.abs_steer)
        self.max_accel_long = max(self.max_accel_long, abs(frame.accel_longitudinal))
        self.max_accel_lat = max(self.max_accel_lat, abs(frame.accel_lateral))
        self.max_rpm = max(self.max_rpm, frame.rpm)

        # Detect lap change
        if frame.current_lap > self.last_lap_number:
            self.last_lap_number = frame.current_lap
            self.lap_count += 1
            return True
        return False
    
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
            "max_accel_g_long": self.max_accel_long / 9.81,  # Convert m/s² to G
            "max_accel_g_lat": self.max_accel_lat / 9.81,
        }

# ============================================================================
# SHARED MEMORY READER
# ============================================================================

class ACTelemetryReader:
    """Read and parse Assetto Corsa Competizione shared memory"""
    
    def __init__(self):
        self.physics_shm: Optional[mmap.mmap] = None
        self.graphics_shm: Optional[mmap.mmap] = None
        self.smoother = SmoothingFilter(window_size=SMOOTHING_WINDOW)
        self.distance_tracker = DistanceTracker()
        self.session = SessionTracker()
        
    def connect(self):
        """Open shared memory connections"""
        try:
            self.physics_shm = mmap.mmap(-1, READ_BLOCK_SIZE, PHYSICS_SHM)
            self.graphics_shm = mmap.mmap(-1, GRAPHICS_BLOCK_SIZE, GRAPHICS_SHM)
            return True
        except FileNotFoundError:
            return False
    
    def read_frame(self) -> Optional[TelemetryFrame]:
        """Read and parse a complete telemetry frame"""
        if not self.physics_shm or not self.graphics_shm:
            return None
        
        try:
            # Read physics block
            self.physics_shm.seek(0)
            physics_data = self.physics_shm.read(READ_BLOCK_SIZE)
            
            # Read graphics block
            self.graphics_shm.seek(0)
            graphics_data = self.graphics_shm.read(GRAPHICS_BLOCK_SIZE)
            
            # Parse basic controls
            gas = struct.unpack('<f', physics_data[GAS_OFFSET:GAS_OFFSET+4])[0]
            brake = struct.unpack('<f', physics_data[BRAKE_OFFSET:BRAKE_OFFSET+4])[0]
            steer = struct.unpack('<f', physics_data[STEER_OFFSET:STEER_OFFSET+4])[0]
            speed_kmh = struct.unpack('<f', physics_data[SPEED_OFFSET:SPEED_OFFSET+4])[0]
            fuel = struct.unpack('<f', physics_data[FUEL_OFFSET:FUEL_OFFSET+4])[0]
            gear = struct.unpack('<i', physics_data[GEAR_OFFSET:GEAR_OFFSET+4])[0]
            rpm = struct.unpack('<i', physics_data[RPM_OFFSET:RPM_OFFSET+4])[0]

            # Apply deadzones
            if abs(steer) < DEADZONE_STEER:
                steer = 0.0
            if abs(gas) < DEADZONE_THROTTLE:
                gas = 0.0
            if abs(brake) < DEADZONE_BRAKE:
                brake = 0.0
            
            # Parse velocity
            vx = struct.unpack('<f', physics_data[VELOCITY_OFFSET:VELOCITY_OFFSET+4])[0]
            vy = struct.unpack('<f', physics_data[VELOCITY_OFFSET+4:VELOCITY_OFFSET+8])[0]
            vz = struct.unpack('<f', physics_data[VELOCITY_OFFSET+8:VELOCITY_OFFSET+12])[0]
            # Parse acceleration
            ax = struct.unpack('<f', physics_data[ACCEL_OFFSET:ACCEL_OFFSET+4])[0]
            ay = struct.unpack('<f', physics_data[ACCEL_OFFSET+4:ACCEL_OFFSET+8])[0]
            az = struct.unpack('<f', physics_data[ACCEL_OFFSET+8:ACCEL_OFFSET+12])[0]
            
            # Derive longitudinal/lateral (these are in m/s²)
            accel_longitudinal = ax
            accel_lateral = az

            def read_float_array(offset, count=4):
                return [struct.unpack('<f', physics_data[offset + i*4:offset + i*4 + 4])[0] 
                       for i in range(count)]
            
            # Parse wheel slip (4 wheels: FL, FR, RL, RR)
            wheel_slip = read_float_array(WHEEL_SLIP_OFFSET)
            # Parse slip angles (4 wheels: FL, FR, RL, RR) - in radians
            slip_angle = read_float_array(SLIP_ANGLE_OFFSET)

            # Parse suspension travel (4 wheels: FL, FR, RL, RR) - in meters
            suspension_travel = [
                struct.unpack('<f', physics_data[SUSPENSION_TRAVEL_OFFSET + i*4 : SUSPENSION_TRAVEL_OFFSET + i*4 + 4])[0] 
                for i in range(4)
            ]

            # Parse graphics data
            lap_time_bytes = graphics_data[GRAPHICS_CURRENT_TIME:GRAPHICS_CURRENT_TIME+30]
            lap_time_str = lap_time_bytes.decode('utf-16le').strip('\x00')
            completed_laps = struct.unpack('<i', graphics_data[GRAPHICS_COMPLETED_LAPS:GRAPHICS_COMPLETED_LAPS+4])[0]
            
            # Timing
            elapsed = time.time() - self.session.start_time
            current_lap = completed_laps + 1
            
            # Derived features
            speed_ms = speed_kmh / 3.6
            gear_display = max(gear - 1, 0)

            # Clamp extreme values to prevent telemetry glitches
            speed_kmh = np.clip(speed_kmh, 0, 400)
            speed_ms = np.clip(speed_ms, 0, 111.1)  # 400 km/h in m/s
            rpm = np.clip(rpm, 0, 12000)
            gear_display = np.clip(gear_display, 0, 8)

            
            # Smooth controls
            gas_smooth, brake_smooth, steer_smooth = self.smoother.update(gas, brake, steer)
            
            # Distance and lap fraction
            distance = self.distance_tracker.update(speed_ms, elapsed)
            lap_fraction = self.distance_tracker.get_lap_fraction()
            
            # Build frame with ALL required fields
            frame = TelemetryFrame(
                timestamp=elapsed,
                lap_time_str=lap_time_str,
                completed_laps=completed_laps,
                current_lap=current_lap,
                speed_kmh=speed_kmh,
                speed_ms=speed_ms,
                gear=gear_display,
                rpm=rpm,
                vx=vx, vy=vy, vz=vz,
                ax=ax, ay=ay, az=az,
                accel_longitudinal=accel_longitudinal,
                accel_lateral=accel_lateral,
                gas=gas,
                brake=brake,
                steer=steer,
                abs_steer=abs(steer),
                gas_smooth=gas_smooth,
                brake_smooth=brake_smooth,
                steer_smooth=steer_smooth,
                distance_m=distance,
                lap_fraction=lap_fraction,
                wheel_slip_fl=wheel_slip[0],
                wheel_slip_fr=wheel_slip[1],
                wheel_slip_rl=wheel_slip[2],
                wheel_slip_rr=wheel_slip[3],
                slip_angle_fl=slip_angle[0],
                slip_angle_fr=slip_angle[1],
                slip_angle_rl=slip_angle[2],
                slip_angle_rr=slip_angle[3],
                suspension_travel_fl=suspension_travel[0],
                suspension_travel_fr=suspension_travel[1],
                suspension_travel_rl=suspension_travel[2],
                suspension_travel_rr=suspension_travel[3],
                fuel_kg=fuel
            )

            # Update session tracking
            lap_changed = self.session.update(frame)
            if lap_changed:
                self.distance_tracker.mark_lap_complete()

            # Clamp extreme values to prevent telemetry glitches
            speed_kmh = np.clip(speed_kmh, 0, 400)
            rpm = np.clip(rpm, 0, 12000)
            gear_display = np.clip(gear_display, 0, 7)  

            return frame
            
        except Exception as e:
            print(f"Error reading telemetry: {e}")
            return None
    
    def close(self):
        """Close shared memory connections"""
        if self.physics_shm:
            self.physics_shm.close()
        if self.graphics_shm:
            self.graphics_shm.close()

# ============================================================================
# CSV LOGGER
# ============================================================================

def create_log_file() -> Path:
    """Create timestamped log file"""
    logs_dir = Path("telemetry_logs")
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return logs_dir / f"acc_session_{timestamp}.csv"

def get_csv_headers():
    """Return CSV column headers - ALL 38 columns"""
    return [
        "timestamp", "lap_time_str", "completed_laps", "current_lap",
        "speed_kmh", "speed_ms", "gear", "rpm",
        "vx", "vy", "vz", "ax", "ay", "az",
        "accel_longitudinal", "accel_lateral",
        "gas", "brake", "steer", "abs_steer",
        "gas_smooth", "brake_smooth", "steer_smooth",
        "distance_m", "lap_fraction",
        "wheel_slip_fl", "wheel_slip_fr", "wheel_slip_rl", "wheel_slip_rr",
        "slip_angle_fl", "slip_angle_fr", "slip_angle_rl", "slip_angle_rr",
        "suspension_travel_fl", "suspension_travel_fr", "suspension_travel_rl", "suspension_travel_rr",
        "fuel_kg"
    ]


def frame_to_csv_row(frame: TelemetryFrame):
    """Convert TelemetryFrame to CSV row"""
    return [
        f"{frame.timestamp:.3f}",
        frame.lap_time_str,
        frame.completed_laps,
        frame.current_lap,
        f"{frame.speed_kmh:.2f}",
        f"{frame.speed_ms:.3f}",
        frame.gear,
        frame.rpm,
        f"{frame.vx:.3f}", f"{frame.vy:.3f}", f"{frame.vz:.3f}",
        f"{frame.ax:.3f}", f"{frame.ay:.3f}", f"{frame.az:.3f}", 
        f"{frame.accel_longitudinal:.3f}",
        f"{frame.accel_lateral:.3f}",
        f"{frame.gas:.4f}",
        f"{frame.brake:.4f}",
        f"{frame.steer:.4f}",
        f"{frame.abs_steer:.4f}",
        f"{frame.gas_smooth:.4f}",
        f"{frame.brake_smooth:.4f}",
        f"{frame.steer_smooth:.4f}",
        f"{frame.distance_m:.2f}",
        f"{frame.lap_fraction:.4f}",
        f"{frame.wheel_slip_fl:.4f}", f"{frame.wheel_slip_fr:.4f}",
        f"{frame.wheel_slip_rl:.4f}", f"{frame.wheel_slip_rr:.4f}",
        f"{frame.slip_angle_fl:.4f}", f"{frame.slip_angle_fr:.4f}",
        f"{frame.slip_angle_rl:.4f}", f"{frame.slip_angle_rr:.4f}",
        f"{frame.suspension_travel_fl:.4f}", f"{frame.suspension_travel_fr:.4f}",  
        f"{frame.suspension_travel_rl:.4f}", f"{frame.suspension_travel_rr:.4f}",
        f"{frame.fuel_kg:.2f}"
    ]

# ============================================================================
# MAIN LOOP
# ============================================================================

def print_session_summary(session: SessionTracker):
    """Print final session statistics"""
    summary = session.get_summary()
    print("\n" + "="*70)
    print("📊 SESSION SUMMARY")
    print("="*70)
    print(f"Duration:         {summary['duration_min']:6.1f} minutes")
    print(f"Samples:          {summary['samples']:6,}")
    print(f"Laps:             {summary['laps']:6}")
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
    
    csv_path = create_log_file()
    print(f"📁 Logging to: {csv_path}")
    
    reader = ACTelemetryReader()
    
    try:
        print("🔌 Connecting to shared memory...")
        if not reader.connect():
            print("\n❌ ACC shared memory not found. Is Assetto Corsa Competizione running?")
            return
        
        print("✓ Connected!")
        print(f"⚙️  Sampling at {SAMPLING_HZ} Hz")
        print(f"🔧 Smoothing window: {SMOOTHING_WINDOW} samples")
        print(f"📊 Capturing: 38 channels")
        print("\n🟢 Recording... (Ctrl+C to stop)\n")
        
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(get_csv_headers())
            
            last_print_time = time.time()
            
            while True:
                frame = reader.read_frame()
                
                if frame:
                    writer.writerow(frame_to_csv_row(frame))
                    
                    # Periodic status update (every 5 seconds)
                    if time.time() - last_print_time >= 5.0:
                        print(f"{frame.timestamp/60:5.1f}m | Lap {frame.current_lap} | "
                              f"{frame.speed_kmh:5.0f} km/h | {frame.rpm:5,} RPM | "
                              f"Top: {reader.session.max_speed:5.0f} | "
                              f"Samples: {reader.session.sample_count:,}")
                        last_print_time = time.time()
                    
                    # Flush every 500 samples
                    if reader.session.sample_count % 500 == 0:
                        csv_file.flush()
                
                time.sleep(1.0 / SAMPLING_HZ)
    
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
