#!/usr/bin/env python3
"""
Feature Schema - Single Source of Truth for ML Feature Order
This file defines the exact feature order used across all components.
"""

import numpy as np

# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

# Complete feature order (for logging/analysis only - NOT for policy input)
FEATURE_ORDER = [
    # --- Kinematics ---
    "speed_ms",
    "accel_longitudinal",
    "accel_lateral",
    "abs_steer",
    "gear",
    "rpm",
    "lap_fraction",
    
    # --- Velocity ---
    "vx",
    "vy", 
    "vz",
    
    # --- Wheel Slip (4 wheels) ---
    "wheel_slip_fl",
    "wheel_slip_fr",
    "wheel_slip_rl",
    "wheel_slip_rr",
    
    # --- Suspension Travel (4 wheels) ---
    "suspension_travel_fl",
    "suspension_travel_fr",
    "suspension_travel_rl",
    "suspension_travel_rr",
]

# Policy input features (EXCLUDES actions to prevent leakage)
# These are the features the ML model uses for prediction
POLICY_FEATURES = FEATURE_ORDER  # All 17 features, no actions

# Action labels (outputs, never inputs)
ACTION_LABELS = [
    "gas",
    "brake", 
    "steer"
]

# Temporal configuration
SEQUENCE_LENGTH = 15  # Number of frames in temporal window
STRIDE = 1  # Frame stride for sequence generation

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
    state = np.array([
        row['speed_ms'],
        row['accel_longitudinal'],
        row['accel_lateral'],
        row['abs_steer'],
        row['gear'],
        row['rpm'],
        row['lap_fraction'],
        row['vx'],
        row['vy'],
        row['vz'],
        row['wheel_slip_fl'],
        row['wheel_slip_fr'],
        row['wheel_slip_rl'],
        row['wheel_slip_rr'],
        row['suspension_travel_fl'],
        row['suspension_travel_fr'],
        row['suspension_travel_rl'],
        row['suspension_travel_rr'],
    ], dtype=np.float32)
    
    return state


def build_state_from_memory(
    speed_ms: float,
    accel_longitudinal: float,
    accel_lateral: float,
    abs_steer: float,
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
    state = np.array([
        speed_ms,
        accel_longitudinal,
        accel_lateral,
        abs_steer,
        gear,
        rpm,  
        lap_fraction,
        vx,
        vy,
        vz,
        wheel_slip_fl,
        wheel_slip_fr,
        wheel_slip_rl,
        wheel_slip_rr,
        suspension_travel_fl,
        suspension_travel_fr,
        suspension_travel_rl,
        suspension_travel_rr,
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
# FEATURE DESCRIPTIONS
# ============================================================================

FEATURE_DESCRIPTIONS = {
    "speed_ms": "Vehicle speed in meters/second",
    "accel_longitudinal": "Longitudinal acceleration (m/s²)",
    "accel_lateral": "Lateral acceleration (m/s²)",
    "abs_steer": "Absolute steering angle",
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
    print(f"Sequence length: {SEQUENCE_LENGTH}")
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
