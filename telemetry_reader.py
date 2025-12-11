import mmap
import struct
import time
import csv
from datetime import datetime
from pathlib import Path

# ----------------------------
# Shared Memory Configuration
# ----------------------------
PHYSICS_SHM = "Local\\acpmf_physics"
GRAPHICS_SHM = "Local\\acpmf_graphics"

# Memory offsets (physics memory)
SPEED_OFFSET = 28      # Float - Speed in km/h
GAS_OFFSET = 4         # Float - Throttle (0.0-1.0)
BRAKE_OFFSET = 8       # Float - Brake (0.0-1.0)
FUEL_OFFSET = 12       # Float - Fuel (kg)
GEAR_OFFSET = 16       # Int   - Current gear
STEER_OFFSET = 24      # Float - Steering angle

# Memory offsets (graphics memory)
GRAPHICS_CURRENT_TIME = 12
GRAPHICS_COMPLETED_LAPS = 132   # int

# Runtime tuning
DEADZONE = 0.05
READ_BLOCK_SIZE = 32
GRAPHICS_BLOCK_SIZE = 256

# ----------------------------
# Data Structures
# ----------------------------
class CarTelemetry:
    """Per-car telemetry tracker"""
    def __init__(self):
        self.speeds = []
        self.steers = []
        self.max_speed = 0
        self.max_gas = 0
        self.max_brake = 0
        self.max_steer = 0
        self.total_distance = 0
        self.prev_speed = 0
        self.prev_time = 0

    def update(self, speed, gas, brake, elapsed, steer):
        self.speeds.append(speed)
        self.steers.append(steer)
        self.max_speed = max(self.max_speed, speed)
        self.max_gas = max(self.max_gas, gas)
        self.max_brake = max(self.max_brake, brake)
        self.max_steer = max(self.max_steer, abs(steer))

        # Distance via trapezoidal integration
        if self.prev_time > 0:
            dt = elapsed - self.prev_time
            avg_speed_ms = ((speed + self.prev_speed) / 2) / 3.6
            self.total_distance += avg_speed_ms * dt

        self.prev_speed = speed
        self.prev_time = elapsed

    def average_speed(self):
        moving = [s for s in self.speeds if s > 5]
        return sum(moving) / len(moving) if moving else 0

    def average_steer(self):
        if not self.steers:
            return 0
        vals = [abs(s) for s in self.steers]
        return sum(vals) / len(vals)

    def summary(self):
        return {
            "Top Speed km/h": self.max_speed,
            "Average Speed km/h": self.average_speed(),
            "Distance km": self.total_distance / 1000,
            "Max Throttle": self.max_gas,
            "Max Brake": self.max_brake,
            "Average |Steer|": self.average_steer(),
            "Max |Steer|": self.max_steer
        }

class Session:
    """Overall session tracking"""
    def __init__(self):
        self.start_time = time.time()
        self.lap_count = 0
        self.car = CarTelemetry()

# ----------------------------
# Utilities
# ----------------------------
def create_log_file():
    logs_dir = Path("telemetry_logs")
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return logs_dir / f"ac_session_{timestamp}.csv"

def print_stats(car: CarTelemetry):
    summary = car.summary()
    print("\n📊 SESSION STATISTICS")
    print("=" * 70)
    print(f"Top Speed:        {summary['Top Speed km/h']:6.1f} km/h  ({summary['Top Speed km/h']/1.609:.1f} mph)")
    print(f"Average Speed:    {summary['Average Speed km/h']:6.1f} km/h")
    print(f"Distance:         {summary['Distance km']:6.2f} km ({summary['Distance km']*1000/1609:.2f} miles)")
    print(f"Max Throttle:     {summary['Max Throttle']:6.1%}")
    print(f"Max Brake:        {summary['Max Brake']:6.1%}")
    print(f"Average |Steer|:   {summary['Average |Steer|']:6.3f} (abs, 0-1)")
    print(f"Max |Steer|:       {summary['Max |Steer|']:6.3f} (abs, 0-1)")
    print("=" * 70)

# ----------------------------
# Main Loop
# ----------------------------
def main():
    print("=" * 70)
    print("ASSETTO CORSA TELEMETRY LOGGER")
    print("=" * 70)

    csv_path = create_log_file()
    print(f"Logging to: {csv_path}\n")

    session = Session()
    sample_count = 0
    debug_mode = False  # Set True for first 20 sample debug output

    try:
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "lap_time_sec", "speed_kmh", "gas", "brake", "steer",
                "gear", "fuel", "lap_time_str", "completed_laps", "current_lap", "distance_m"
            ])

            print("Connecting to shared memory...")
            with mmap.mmap(-1, 2048, PHYSICS_SHM) as shm, \
                 mmap.mmap(-1, GRAPHICS_BLOCK_SIZE, GRAPHICS_SHM) as graphics_shm:
                print("✓ Connected to AC\n")
                print("Live telemetry (Ctrl+C to stop):")
                print("-" * 70)

                while True:
                    elapsed = time.time() - session.start_time

                    # Read physics block
                    shm.seek(0)
                    blk = shm.read(READ_BLOCK_SIZE)

                    gas = struct.unpack('<f', blk[GAS_OFFSET:GAS_OFFSET+4])[0]
                    brake = struct.unpack('<f', blk[BRAKE_OFFSET:BRAKE_OFFSET+4])[0]
                    fuel = struct.unpack('<f', blk[FUEL_OFFSET:FUEL_OFFSET+4])[0]
                    gear = struct.unpack('<i', blk[GEAR_OFFSET:GEAR_OFFSET+4])[0]
                    steer = struct.unpack('<f', blk[STEER_OFFSET:STEER_OFFSET+4])[0]
                    speed_kmh = struct.unpack('<f', blk[SPEED_OFFSET:SPEED_OFFSET+4])[0]

                    # Apply deadzone
                    # if abs(steer) < DEADZONE:
                       # steer = 0.0

                    gear_display = max(gear - 1, 0)

                    # Read graphics block
                    graphics_shm.seek(0)
                    graphics_block = graphics_shm.read(GRAPHICS_BLOCK_SIZE)

                    lap_time_bytes = graphics_block[GRAPHICS_CURRENT_TIME:GRAPHICS_CURRENT_TIME+30]
                    try:
                        lap_time_str = lap_time_bytes.decode('utf-16le').strip('\x00')
                    except Exception:
                        lap_time_str = ""

                    completed_laps = struct.unpack('<i', graphics_block[GRAPHICS_COMPLETED_LAPS:GRAPHICS_COMPLETED_LAPS+4])[0]
                    current_lap = completed_laps + 1

                    # Update telemetry
                    session.car.update(speed_kmh, gas, brake, elapsed, steer)

                    # Log CSV
                    writer.writerow([
                        f"{elapsed:.3f}",
                        f"{speed_kmh:.2f}",
                        f"{gas:.3f}",
                        f"{brake:.3f}",
                        f"{steer:.3f}",
                        gear_display,
                        f"{fuel:.2f}",
                        lap_time_str,
                        completed_laps,
                        current_lap,
                        f"{session.car.total_distance:.2f}"
                    ])

                    sample_count += 1
                    if sample_count % 100 == 0:  # ~1 Hz
                        csv_file.flush()
                        print(f"{elapsed:7.1f}s | Lap: {current_lap:2d} | Speed: {speed_kmh:6.1f} km/h | Gear: {gear_display:2d} | Top: {session.car.max_speed:6.1f}")

                    time.sleep(0.01)  # 100 Hz

    except FileNotFoundError:
        print("\n❌ AC shared memory not found. Is Assetto Corsa running?")

    except KeyboardInterrupt:
        duration = time.time() - session.start_time
        print("\n\n" + "=" * 70)
        print("Session Complete")
        print(f"Duration: {duration:.1f}s | Samples: {sample_count} | Rate: {sample_count/duration:.1f} Hz")
        print(f"Saved: {csv_path}")
        print_stats(session.car)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
