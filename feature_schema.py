#!/usr/bin/env python3
"""
Feature Schema - Single Source of Truth for ML Feature Order
This file defines the exact feature order used across all components.
"""

import numpy as np

# Sequence settings for temporal models
# Match dataset: sequence length used when creating temporal sequences
SEQUENCE_LENGTH = 15
STRIDE = 1

# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

# Policy input features (17 features total - NO actions to prevent leakage)
# These are the features the ML model uses for prediction (kept for compatibility)
POLICY_FEATURES = [
    "timestamp",
    "completed_laps",
    "current_lap",
    "speed_kmh",
    "speed_ms",
    "gear",
    "rpm",
    "vx",
    "vy",
    "vz",
    "ax",
    "ay",
    "az",
    "accel_longitudinal",
    "accel_lateral",
    "distance_m",
    "lap_fraction",
]

# Full CSV telemetry fields produced by `reader.py` (source of truth for logging)
FULL_TELEMETRY_FIELDS = [
    "timestamp", "lap_time_str", "completed_laps", "current_lap",
    "speed_kmh", "speed_ms", "gear", "rpm",
    "vx", "vy", "vz", "ax", "ay", "az",
    "accel_longitudinal", "accel_lateral",
    "gas", "brake", "steer", "abs_steer",
    "gas_smooth", "brake_smooth", "steer_smooth",
    "distance_m", "lap_fraction",
    "wheel_slip_fl", "wheel_slip_fr", "wheel_slip_rl", "wheel_slip_rr",
    "suspension_travel_fl", "suspension_travel_fr", "suspension_travel_rl", "suspension_travel_rr",
    "fuel_kg",
]

# AVAILABLE_FIELDS: full set of fields available/working in ACC shared memory
# (mirrors green-highlighted items from the ACCSharedMemoryDocumentation)
AVAILABLE_FIELDS = {
    "SPageFilePhysics": [
        "packetId", "gas", "brake", "clutch", "fuel", "gear", "rpm", "steerAngle",
        "speedKmh", "velocity_vx", "velocity_vy", "velocity_vz",
        "accG_ax", "accG_ay", "accG_az",
        "wheelSlip_fl", "wheelSlip_fr", "wheelSlip_rl", "wheelSlip_rr",
        "wheelPressure_fl", "wheelPressure_fr", "wheelPressure_rl", "wheelPressure_rr",
        "wheelAngularSpeed_fl", "wheelAngularSpeed_fr", "wheelAngularSpeed_rl", "wheelAngularSpeed_rr",
        "tyreWear_fl", "tyreWear_fr", "tyreWear_rl", "tyreWear_rr",
        "tyreDirtyLevel_fl", "tyreDirtyLevel_fr", "tyreDirtyLevel_rl", "tyreDirtyLevel_rr",
        "tyreCoreTemp_fl", "tyreCoreTemp_fr", "tyreCoreTemp_rl", "tyreCoreTemp_rr",
        "camberRAD_fl", "camberRAD_fr", "camberRAD_rl", "camberRAD_rr",
        "suspensionTravel_fl", "suspensionTravel_fr", "suspensionTravel_rl", "suspensionTravel_rr",
        "heading", "pitch", "roll", "cgHeight",
        "carDamage_front", "carDamage_rear", "carDamage_left", "carDamage_right", "carDamage_center",
        "numberOfTyresOut", "pitLimiterOn", "abs", "rideHeight_front", "rideHeight_rear",
        "turboBoost", "airDensity", "airTemp", "roadTemp",
        "localAngularVel_x", "localAngularVel_y", "localAngularVel_z",
        "finalFF", "brakeTemp_fl", "brakeTemp_fr", "brakeTemp_rl", "brakeTemp_rr",
        "isAIControlled", "tyreContactPoint_fl_x", "tyreContactPoint_fl_y", "tyreContactPoint_fl_z",
        "tyreContactPoint_fr_x", "tyreContactPoint_fr_y", "tyreContactPoint_fr_z",
        "tyreContactPoint_rl_x", "tyreContactPoint_rl_y", "tyreContactPoint_rl_z",
        "tyreContactPoint_rr_x", "tyreContactPoint_rr_y", "tyreContactPoint_rr_z",
        "tyreContactNormal_fl_x", "tyreContactNormal_fr_x", "tyreContactNormal_rl_x", "tyreContactNormal_rr_x",
        "tyreContactHeading_fl", "tyreContactHeading_fr", "tyreContactHeading_rl", "tyreContactHeading_rr",
        "brakeBias", "localVelocity_x", "localVelocity_y", "localVelocity_z",
        "currentMaxRpm", "slipRatio_fl", "slipRatio_fr", "slipRatio_rl", "slipRatio_rr",
        "slipAngle_fl", "slipAngle_fr", "slipAngle_rl", "slipAngle_rr",
        "tcInAction", "absInAction", "suspensionDamage_fl", "suspensionDamage_fr", "suspensionDamage_rl", "suspensionDamage_rr",
        "waterTemp", "brakePressure_fl", "brakePressure_fr", "brakePressure_rl", "brakePressure_rr",
        "frontBrakeCompound", "rearBrakeCompound", "padLife_fl", "padLife_fr", "padLife_rl", "padLife_rr",
        "discLife_fl", "discLife_fr", "discLife_rl", "discLife_rr",
        "ignitionOn", "starterEngineOn", "isEngineRunning",
        "kerbVibration", "slipVibrations", "gVibrations", "absVibrations",
        # KERS/DRS not used in ACC but present in doc
        "kersCharge", "kersInput", "performanceMeter", "drsAvailable",
    ],
    "SPageFileGraphic": [
        "packetId", "status", "session", "currentTime", "lastTime", "bestTime", "split",
        "completedLaps", "position", "iCurrentTime", "iLastTime", "iBestTime",
        "sessionTimeLeft", "distanceTraveled", "isInPit", "isInPitLane",
        "currentSectorIndex", "lastSectorTime", "numberOfLaps", "tyreCompound",
        "replayTimeMultiplier", "normalizedCarPosition", "activeCars",
        "carCoordinates", "carID", "playerCarID", "penaltyTime", "penalty",
        "idealLineOn", "surfaceGrip", "mandatoryPitDone", "windSpeed", "windDirection",
        "isSetupMenuVisible", "mainDisplayIndex", "secondaryDisplayIndex", "TC", "TCCUT", "EngineMap", "ABS",
        "fuelXLap", "rainLights", "flashingLights", "lightsStage", "exhaustTemperature", "wiperLV",
        "driverStintTotalTimeLeft", "driverStintTimeLeft", "rainTyres", "sessionIndex", "usedFuel",
        "deltaLapTime", "iDeltaLapTime", "estimatedLapTime", "iEstimatedLapTime", "isDeltaPositive", "iSplit", "isValidLap",
        "fuelEstimatedLaps", "trackStatus", "missingMandatoryPits", "Clock",
        "directionLightsLeft", "directionLightsRight", "GlobalYellow", "GlobalWhite", "GlobalGreen", "GlobalChequered", "GlobalRed",
        "mfdTyreSet", "mfdFuelToAdd", "mfdTyrePressureLF", "mfdTyrePressureRF", "mfdTyrePressureLR", "mfdTyrePressureRR",
        "trackGripStatus", "rainIntensity", "rainIntensityIn10min", "rainIntensityIn30min", "currentTyreSet", "strategyTyreSet",
        "gapAhead", "gapBehind"
    ],
    "SPageFileStatic": [
        "smVersion", "acVersion", "numberOfSessions", "numCars", "track", "sectorCount",
        "maxRpm", "maxFuel", "dryTyresName", "wetTyresName", "isOnline",
        "suspensionMaxTravel_fl", "suspensionMaxTravel_fr", "suspensionMaxTravel_rl", "suspensionMaxTravel_rr",
        "tyreRadius_fl", "tyreRadius_fr", "tyreRadius_rl", "tyreRadius_rr",
    ]
}

# Action labels (outputs, never inputs)
ACTION_LABELS = [
    "gas",
    "brake", 
    "steer"
]

# ============================================================================
# STATE CONSTRUCTION FUNCTIONS
# ============================================================================

def build_state_from_row(row: dict) -> np.ndarray:
    """
    Build state vector from CSV row (pandas Series or dict)
    Used in: telemetry_analysis.py
    
    Args:
        row: Dictionary-like object with feature names as keys
        
    Returns:
        State vector as numpy array, shape (17,)
    """
    # Build state in exact order of POLICY_FEATURES
    state = np.array([
        float(row['timestamp']),
        float(row['completed_laps']),
        float(row['current_lap']),
        float(row['speed_kmh']),
        float(row['speed_ms']),
        float(row['gear']),
        float(row['rpm']),
        float(row['vx']),
        float(row['vy']),
        float(row['vz']),
        float(row['ax']),
        float(row['ay']),
        float(row['az']),
        float(row['accel_longitudinal']),
        float(row['accel_lateral']),
        float(row['distance_m']),
        float(row['lap_fraction']),
    ], dtype=np.float32)
    
    return state


def build_state_from_memory(
    speed_ms: float,
    accel_longitudinal: float,
    accel_lateral: float,
    gear: float,
    rpm: float,
    lap_fraction: float,
    vx: float,
    vy: float,
    vz: float,
    wheel_slip_fl: float,
    wheel_slip_fr: float,
    wheel_slip_rl: float,
    wheel_slip_rr: float,
    # (slip angles removed)
    suspension_travel_fl: float,
    suspension_travel_fr: float,
    suspension_travel_rl: float,
    suspension_travel_rr: float,
) -> np.ndarray:
    """
    Build state vector from individual values (shared memory)
    Used in: driver.py
    
    Args:
        Individual feature values in exact order
        
    Returns:
        State vector as numpy array, shape (17,)
    """
    # This helper is retained for compatibility with driver/memory-based builds
    # but maps to the reduced 17-feature vector. For values not available
    # in memory callers should supply sensible defaults.
    state = np.array([
        0.0,                 # timestamp (unknown here)
        0.0,                 # completed_laps
        0.0,                 # current_lap
        speed_ms * 3.6,      # speed_kmh (approx)
        speed_ms,
        gear,
        rpm,
        vx,
        vy,
        vz,
        accel_longitudinal,  # ax
        accel_lateral,       # ay
        0.0,                 # az (not provided)
        accel_longitudinal,
        accel_lateral,
        0.0,                 # distance_m (unknown)
        lap_fraction,
    ], dtype=np.float32)
    
    return state


def validate_state(state: np.ndarray, context: str = ""):
    """
    Validate state vector has correct shape
    
    Args:
        state: State vector to validate
        context: Optional context string for error message
        
    Raises:
        AssertionError: If state shape doesn't match expected feature count
    """
    expected = len(POLICY_FEATURES)
    actual = state.shape[-1] if state.ndim > 0 else 0
    
    if actual != expected:
        error_msg = f"Feature count mismatch{' in ' + context if context else ''}!\n"
        error_msg += f"  Expected: {expected} features\n"
        error_msg += f"  Got: {actual} features\n"
        error_msg += f"  Expected features: {POLICY_FEATURES}"
        raise AssertionError(error_msg)


def validate_normalization(norm_min: np.ndarray, norm_max: np.ndarray):
    """
    Validate normalization arrays match feature count
    
    Args:
        norm_min: Minimum values for normalization
        norm_max: Maximum values for normalization
        
    Raises:
        AssertionError: If normalization arrays don't match feature count
    """
    expected = len(POLICY_FEATURES)
    
    if len(norm_min) != expected:
        raise AssertionError(
            f"Normalization min array mismatch!\n"
            f"  Expected: {expected} features\n"
            f"  Got: {len(norm_min)} features"
        )
    
    if len(norm_max) != expected:
        raise AssertionError(
            f"Normalization max array mismatch!\n"
            f"  Expected: {expected} features\n"
            f"  Got: {len(norm_max)} features"
        )


# ============================================================================
# FULL TELEMETRY FIELD METADATA
# ============================================================================
# Detailed metadata for all fields produced by the CSV logger. This is
# documentation-only and used by downstream tools for validation and UI.
FIELD_METADATA = {
    # Timing
    "timestamp":       {"type": "float", "description": "Elapsed seconds since session start", "unit": "s", "source": "SPageFileGraphic/Sess"},
    "lap_time_str":    {"type": "string", "description": "Formatted current lap time", "unit": "mm:ss:ms", "source": "SPageFileGraphic"},
    "completed_laps":  {"type": "int", "description": "Completed laps from graphics memory", "unit": "count", "source": "SPageFileGraphic"},
    "current_lap":     {"type": "int", "description": "Current lap index (completed_laps+1)", "unit": "count", "source": "derived"},

    # Core vehicle state
    "speed_kmh":       {"type": "float", "description": "Vehicle speed", "unit": "km/h", "source": "SPageFilePhysics.speedKmh"},
    "speed_ms":        {"type": "float", "description": "Vehicle speed", "unit": "m/s", "source": "derived (speed_kmh / 3.6)"},
    "gear":            {"type": "int", "description": "Displayed gear (0..N)", "unit": "ordinal", "source": "SPageFilePhysics.gear (mapped)"},
    "rpm":             {"type": "int", "description": "Engine RPM", "unit": "rpm", "source": "SPageFilePhysics.rpm"},

    # Kinematics
    "vx":              {"type": "float", "description": "Velocity X (local)", "unit": "m/s", "source": "SPageFilePhysics.velocity[0]"},
    "vy":              {"type": "float", "description": "Velocity Y (local)", "unit": "m/s", "source": "SPageFilePhysics.velocity[1]"},
    "vz":              {"type": "float", "description": "Velocity Z (local)", "unit": "m/s", "source": "SPageFilePhysics.velocity[2]"},
    "ax":              {"type": "float", "description": "Acceleration X (raw)", "unit": "m/s²", "source": "SPageFilePhysics.accG[0]"},
    "ay":              {"type": "float", "description": "Acceleration Y (raw)", "unit": "m/s²", "source": "SPageFilePhysics.accG[1]"},
    "az":              {"type": "float", "description": "Acceleration Z (raw)", "unit": "m/s²", "source": "SPageFilePhysics.accG[2]"},
    "accel_longitudinal": {"type": "float", "description": "Longitudinal acceleration (derived from ax)", "unit": "m/s²", "source": "derived (ax)"},
    "accel_lateral":   {"type": "float", "description": "Lateral acceleration (derived from az)", "unit": "m/s²", "source": "derived (az)"},

    # Controls (raw + smoothed)
    "gas":             {"type": "float", "description": "Throttle input", "unit": "[0-1]", "source": "SPageFilePhysics.gas"},
    "brake":           {"type": "float", "description": "Brake input", "unit": "[0-1]", "source": "SPageFilePhysics.brake"},
    "steer":           {"type": "float", "description": "Steering wheel angle (raw) - typically in radians. ML models should normalize to [-1,1] for training/inference.", "unit": "rad", "source": "SPageFilePhysics.steerAngle", "recommended_normalization": "[-1,1]"},
    "abs_steer":       {"type": "float", "description": "Absolute steering magnitude", "unit": "rad", "source": "derived (abs(steer))"},
    "gas_smooth":      {"type": "float", "description": "Smoothed throttle (moving average)", "unit": "[0-1]", "source": "derived"},
    "brake_smooth":    {"type": "float", "description": "Smoothed brake (moving average)", "unit": "[0-1]", "source": "derived"},
    "steer_smooth":    {"type": "float", "description": "Smoothed steer (moving average)", "unit": "rad", "source": "derived"},

    # Position & progress
    "distance_m":      {"type": "float", "description": "Accumulated distance (integrated from speed)", "unit": "m", "source": "derived (trapezoidal integration)"},
    "lap_fraction":    {"type": "float", "description": "Progress through current lap [0-1]", "unit": "fraction", "source": "derived (DistanceTracker)"},

    # Wheel / suspension
    "wheel_slip_fl":   {"type": "float", "description": "Wheel slip front-left", "unit": "ratio", "source": "SPageFilePhysics.wheelSlip[0]"},
    "wheel_slip_fr":   {"type": "float", "description": "Wheel slip front-right", "unit": "ratio", "source": "SPageFilePhysics.wheelSlip[1]"},
    "wheel_slip_rl":   {"type": "float", "description": "Wheel slip rear-left", "unit": "ratio", "source": "SPageFilePhysics.wheelSlip[2]"},
    "wheel_slip_rr":   {"type": "float", "description": "Wheel slip rear-right", "unit": "ratio", "source": "SPageFilePhysics.wheelSlip[3]"},
    "slip_angle_fl":   {"type": "float", "description": "Slip angle front-left (may be zero or absent depending on game/version)", "unit": "rad", "source": "SPageFilePhysics (SLIP_ANGLE_OFFSET - 4 floats)", "optional": True},
    "slip_angle_fr":   {"type": "float", "description": "Slip angle front-right (may be zero or absent depending on game/version)", "unit": "rad", "source": "SPageFilePhysics (SLIP_ANGLE_OFFSET - 4 floats)", "optional": True},
    "slip_angle_rl":   {"type": "float", "description": "Slip angle rear-left (may be zero or absent depending on game/version)", "unit": "rad", "source": "SPageFilePhysics (SLIP_ANGLE_OFFSET - 4 floats)", "optional": True},
    "slip_angle_rr":   {"type": "float", "description": "Slip angle rear-right (may be zero or absent depending on game/version)", "unit": "rad", "source": "SPageFilePhysics (SLIP_ANGLE_OFFSET - 4 floats)", "optional": True},
    "suspension_travel_fl": {"type": "float", "description": "Suspension travel front-left", "unit": "m", "source": "SPageFilePhysics.suspensionTravel[0]"},
    "suspension_travel_fr": {"type": "float", "description": "Suspension travel front-right", "unit": "m", "source": "SPageFilePhysics.suspensionTravel[1]"},
    "suspension_travel_rl": {"type": "float", "description": "Suspension travel rear-left", "unit": "m", "source": "SPageFilePhysics.suspensionTravel[2]"},
    "suspension_travel_rr": {"type": "float", "description": "Suspension travel rear-right", "unit": "m", "source": "SPageFilePhysics.suspensionTravel[3]"},

    # Misc
    "fuel_kg":         {"type": "float", "description": "Fuel remaining", "unit": "kg", "source": "SPageFilePhysics.fuel"},
}

def print_full_schema():
    """Print detailed telemetry schema for human review."""
    print("=" * 70)
    print("FULL TELEMETRY SCHEMA")
    print("=" * 70)
    print(f"Total logged fields: {len(FULL_TELEMETRY_FIELDS)}\n")
    for name in FULL_TELEMETRY_FIELDS:
        meta = FIELD_METADATA.get(name, {})
        dtype = meta.get('type', 'unknown')
        unit = meta.get('unit', '')
        src = meta.get('source', '')
        desc = meta.get('description', '')
        print(f"- {name:25s} | {dtype:7s} | {unit:8s} | {src:30s} | {desc}")
    print("\n" + "=" * 70)


# ============================================================================
# FEATURE DESCRIPTIONS (for reference)
# ============================================================================

FEATURE_DESCRIPTIONS = {
    "speed_ms": "Vehicle speed in meters/second",
    "accel_longitudinal": "Longitudinal acceleration (m/s²)",
    "accel_lateral": "Lateral acceleration (m/s²)",
    "gear": "Current gear (integer)",
    "rpm": "Engine revolutions per minute",
    "lap_fraction": "Progress through lap [0-1]",
    "vx": "Velocity X component (m/s)",
    "vy": "Velocity Y component (m/s)",
    "vz": "Velocity Z component (m/s)",
    "wheel_slip_fl": "Wheel slip front-left",
    "wheel_slip_fr": "Wheel slip front-right",
    "wheel_slip_rl": "Wheel slip rear-left",
    "wheel_slip_rr": "Wheel slip rear-right",
    "suspension_travel_fl": "Suspension travel front-left (m)",
    "suspension_travel_fr": "Suspension travel front-right (m)",
    "suspension_travel_rl": "Suspension travel rear-left (m)",
    "suspension_travel_rr": "Suspension travel rear-right (m)",
}

ACTION_DESCRIPTIONS = {
    "gas": "Throttle input [0-1]",
    "brake": "Brake input [0-1]",
    "steer": "Steering input [-1, 1]",
}


def print_schema_info():
    """Print schema information for debugging"""
    print("=" * 70)
    print("FEATURE SCHEMA")
    print("=" * 70)
    print(f"\nTotal features: {len(POLICY_FEATURES)}")
    print(f"\nPOLICY INPUT FEATURES ({len(POLICY_FEATURES)}):")
    for i, name in enumerate(POLICY_FEATURES):
        desc = FEATURE_DESCRIPTIONS.get(name, "No description")
        print(f"  [{i:2d}] {name:25s} - {desc}")
    
    print(f"\nACTION LABELS ({len(ACTION_LABELS)}):")
    for i, name in enumerate(ACTION_LABELS):
        desc = ACTION_DESCRIPTIONS.get(name, "No description")
        print(f"  [{i}] {name:10s} - {desc}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_schema_info()
