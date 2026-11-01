#!/usr/bin/env python3
"""
ACC shared memory layouts as ctypes structures.
Field order copied from ACCSharedMemoryDocumentationV1.8.12.pdf, so every
field is read by name instead of by a hand-counted byte offset.
"""

import ctypes
import mmap
from ctypes import c_float, c_int, c_wchar

PHYSICS_SHM = "Local\\acpmf_physics"
GRAPHICS_SHM = "Local\\acpmf_graphics"
STATIC_SHM = "Local\\acpmf_static"

F4 = c_float * 4
F3 = c_float * 3

# Enums (from the PDF)
ACC_STATUS = {0: "off", 1: "replay", 2: "live", 3: "pause"}
ACC_SESSION_TYPE = {-1: "unknown", 0: "practice", 1: "qualify", 2: "race", 3: "hotlap",
                    4: "time_attack", 5: "drift", 6: "drag", 7: "hotstint", 8: "superpole"}
ACC_FLAG_TYPE = {0: "none", 1: "blue", 2: "yellow", 3: "black", 4: "white",
                 5: "chequered", 6: "penalty", 7: "green", 8: "orange"}


class SPageFilePhysics(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("packetId", c_int), ("gas", c_float), ("brake", c_float), ("fuel", c_float),
        ("gear", c_int), ("rpm", c_int), ("steerAngle", c_float), ("speedKmh", c_float),
        ("velocity", F3), ("accG", F3),
        ("wheelSlip", F4), ("wheelLoad", F4), ("wheelPressure", F4), ("wheelAngularSpeed", F4),
        ("tyreWear", F4), ("tyreDirtyLevel", F4), ("tyreCoreTemperature", F4), ("camberRAD", F4),
        ("suspensionTravel", F4),
        ("drs", c_float), ("tc", c_float), ("heading", c_float), ("pitch", c_float),
        ("roll", c_float), ("cgHeight", c_float), ("carDamage", c_float * 5),
        ("numberOfTyresOut", c_int), ("pitLimiterOn", c_int), ("abs", c_float),
        ("kersCharge", c_float), ("kersInput", c_float), ("autoShifterOn", c_int),
        ("rideHeight", c_float * 2), ("turboBoost", c_float), ("ballast", c_float),
        ("airDensity", c_float), ("airTemp", c_float), ("roadTemp", c_float),
        ("localAngularVel", F3), ("finalFF", c_float), ("performanceMeter", c_float),
        ("engineBrake", c_int), ("ersRecoveryLevel", c_int), ("ersPowerLevel", c_int),
        ("ersHeatCharging", c_int), ("ersIsCharging", c_int), ("kersCurrentKJ", c_float),
        ("drsAvailable", c_int), ("drsEnabled", c_int), ("brakeTemp", F4), ("clutch", c_float),
        ("tyreTempI", F4), ("tyreTempM", F4), ("tyreTempO", F4), ("isAIControlled", c_int),
        ("tyreContactPoint", F3 * 4), ("tyreContactNormal", F3 * 4), ("tyreContactHeading", F3 * 4),
        ("brakeBias", c_float), ("localVelocity", F3), ("P2PActivation", c_int), ("P2PStatus", c_int),
        ("currentMaxRpm", c_float), ("mz", F4), ("fx", F4), ("fy", F4),
        ("slipRatio", F4), ("slipAngle", F4), ("tcinAction", c_int), ("absInAction", c_int),
        ("suspensionDamage", F4), ("tyreTemp", F4), ("waterTemp", c_float), ("brakePressure", F4),
        ("frontBrakeCompound", c_int), ("rearBrakeCompound", c_int), ("padLife", F4), ("discLife", F4),
        ("ignitionOn", c_int), ("starterEngineOn", c_int), ("isEngineRunning", c_int),
        ("kerbVibration", c_float), ("slipVibrations", c_float), ("gVibrations", c_float),
        ("absVibrations", c_float),
    ]


class SPageFileGraphic(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("packetId", c_int), ("status", c_int), ("session", c_int),
        ("currentTime", c_wchar * 15), ("lastTime", c_wchar * 15), ("bestTime", c_wchar * 15),
        ("split", c_wchar * 15),
        ("completedLaps", c_int), ("position", c_int), ("iCurrentTime", c_int), ("iLastTime", c_int),
        ("iBestTime", c_int), ("sessionTimeLeft", c_float), ("distanceTraveled", c_float),
        ("isInPit", c_int), ("currentSectorIndex", c_int), ("lastSectorTime", c_int),
        ("numberOfLaps", c_int), ("tyreCompound", c_wchar * 33), ("replayTimeMultiplier", c_float),
        ("normalizedCarPosition", c_float), ("activeCars", c_int),
        ("carCoordinates", F3 * 60), ("carID", c_int * 60), ("playerCarID", c_int),
        ("penaltyTime", c_float), ("flag", c_int), ("penalty", c_int), ("idealLineOn", c_int),
        ("isInPitLane", c_int), ("surfaceGrip", c_float), ("mandatoryPitDone", c_int),
        ("windSpeed", c_float), ("windDirection", c_float), ("isSetupMenuVisible", c_int),
        ("mainDisplayIndex", c_int), ("secondaryDisplayIndex", c_int), ("TC", c_int),
        ("TCCut", c_int), ("EngineMap", c_int), ("ABS", c_int), ("fuelXLap", c_float),
        ("rainLights", c_int), ("flashingLights", c_int), ("lightsStage", c_int),
        ("exhaustTemperature", c_float), ("wiperLV", c_int), ("driverStintTotalTimeLeft", c_int),
        ("driverStintTimeLeft", c_int), ("rainTyres", c_int), ("sessionIndex", c_int),
        ("usedFuel", c_float), ("deltaLapTime", c_wchar * 15), ("iDeltaLapTime", c_int),
        ("estimatedLapTime", c_wchar * 15), ("iEstimatedLapTime", c_int), ("isDeltaPositive", c_int),
        ("iSplit", c_int), ("isValidLap", c_int), ("fuelEstimatedLaps", c_float),
        ("trackStatus", c_wchar * 33), ("missingMandatoryPits", c_int), ("clock", c_float),
        ("directionLightsLeft", c_int), ("directionLightsRight", c_int),
        ("globalYellow", c_int), ("globalYellow1", c_int), ("globalYellow2", c_int),
        ("globalYellow3", c_int), ("globalWhite", c_int), ("globalGreen", c_int),
        ("globalChequered", c_int), ("globalRed", c_int), ("mfdTyreSet", c_int),
        ("mfdFuelToAdd", c_float), ("mfdTyrePressureLF", c_float), ("mfdTyrePressureRF", c_float),
        ("mfdTyrePressureLR", c_float), ("mfdTyrePressureRR", c_float), ("trackGripStatus", c_int),
        ("rainIntensity", c_int), ("rainIntensityIn10min", c_int), ("rainIntensityIn30min", c_int),
        ("currentTyreSet", c_int), ("strategyTyreSet", c_int), ("gapAhead", c_int), ("gapBehind", c_int),
    ]


class SPageFileStatic(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("smVersion", c_wchar * 15), ("acVersion", c_wchar * 15), ("numberOfSessions", c_int),
        ("numCars", c_int), ("carModel", c_wchar * 33), ("track", c_wchar * 33),
        ("playerName", c_wchar * 33), ("playerSurname", c_wchar * 33), ("playerNick", c_wchar * 33),
        ("sectorCount", c_int), ("maxTorque", c_float), ("maxPower", c_float), ("maxRpm", c_int),
        ("maxFuel", c_float), ("suspensionMaxTravel", F4), ("tyreRadius", F4),
        ("maxTurboBoost", c_float), ("deprecated_1", c_float), ("deprecated_2", c_float),
        ("penaltiesEnabled", c_int), ("aidFuelRate", c_float), ("aidTireRate", c_float),
        ("aidMechanicalDamage", c_float), ("allowTyreBlankets", c_float), ("aidStability", c_float),
        ("aidAutoClutch", c_int), ("aidAutoBlip", c_int), ("hasDRS", c_int), ("hasERS", c_int),
        ("hasKERS", c_int), ("kersMaxJ", c_float), ("engineBrakeSettingsCount", c_int),
        ("ersPowerControllerCount", c_int), ("trackSplineLength", c_float),
        ("trackConfiguration", c_wchar * 33), ("ersMaxJ", c_float), ("isTimedRace", c_int),
        ("hasExtraLap", c_int), ("carSkin", c_wchar * 33), ("reversedGridPositions", c_int),
        ("PitWindowStart", c_int), ("PitWindowEnd", c_int), ("isOnline", c_int),
        ("dryTyresName", c_wchar * 33), ("wetTyresName", c_wchar * 33),
    ]


class SharedMemoryPage:
    """One named shared memory page, read into a fresh struct on each call."""

    def __init__(self, struct_type, tag: str):
        self.struct_type = struct_type
        self.size = ctypes.sizeof(struct_type)
        self.shm = mmap.mmap(-1, self.size, tag)  # raises FileNotFoundError-like OSError if missing

    def read(self):
        self.shm.seek(0)
        return self.struct_type.from_buffer_copy(self.shm.read(self.size))

    def close(self):
        self.shm.close()


def player_coordinates(graphics: SPageFileGraphic):
    """World (x, y, z) of the player's car, or None if it isn't in the car list."""
    for i in range(min(graphics.activeCars, 60)):
        if graphics.carID[i] == graphics.playerCarID:
            return tuple(graphics.carCoordinates[i])
    return None


if __name__ == "__main__":
    for s in (SPageFilePhysics, SPageFileGraphic, SPageFileStatic):
        print(f"{s.__name__:18s} {ctypes.sizeof(s)} bytes")
