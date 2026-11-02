#!/usr/bin/env python3
"""
pilot_sim.py: a stand-in for ACC so pilot.py can be tested without the game.

The car is fitted to your telemetry (track_line.car_envelope): acceleration, braking
and cornering grip per speed band, and how much steer a given curvature took. It runs
on the same world coordinates as the reference line and exposes the same read/write
interface as pilot.AccBackend.

What it deliberately gets wrong, so a controller that passes here still has to be robust:
  - steering lag (ACC's input filtering, the car's yaw response): first-order, 0.10 s
  - observation delay: the pilot sees state 2 frames old
  - steer_mismatch: the car needs this many times more steer than the fit says
  - grip: past the fitted lateral limit the car understeers wide instead of turning

Not modelled: elevation (Bathurst climbs ~170 m), kerbs, weight transfer, oversteer.
"""

import math
from collections import deque

import numpy as np

from track_line import TrackLine

DT = 0.01


class SimBackend:
    def __init__(self, track: TrackLine, start_s=10.0, start_speed=0.0, steer_mismatch=1.0,
                 grip=1.0, steer_lag=0.10, delay_frames=2, road_half_m=7.5, wall_m=12.0, steer_scale=None):
        self.t = track
        self.car = track.meta["car"]
        # Lateral grip: enforce non-decreasing with speed (straights never reached the limit)
        self.lat = np.maximum.accumulate(np.array(self.car["lat"])) * grip
        self.mismatch = steer_mismatch
        self.steer_scale = steer_scale                  # (v, gain): ACC's gamepad scales the joystick down with speed
        self.steer_lag = steer_lag
        self.road_half, self.wall = road_half_m, wall_m
        self.x = track.at(start_s, track.x)
        self.z = track.at(start_s, track.z)
        self.h = track.heading_at(start_s)
        self.v = start_speed
        self.steer = 0.0
        self.cmd = (0.0, 0.0, 0.0)
        self.crashed = False
        self.sliding = False
        self.probe = TrackLine.__new__(TrackLine)     # own locator so it doesn't share the pilot's index
        self.probe.__dict__.update({k: v for k, v in track.__dict__.items()})
        self.probe.idx = None
        self.hist = deque(maxlen=delay_frames + 1)

    def _state(self):
        return {"x": self.x, "z": self.z, "heading": self.h, "speed": self.v, "steer": self.steer,
                "lap_fraction": None, "crashed": self.crashed, "sliding": self.sliding}

    def read(self):
        self._physics()
        self.hist.append(self._state())
        return self.hist[0]

    def write(self, gas, brake, steer, st=None):
        self.cmd = (gas, brake, steer)

    def _physics(self):
        gas, brake, joy = self.cmd
        v = self.v
        if self.steer_scale is not None:
            joy *= float(np.interp(v, *self.steer_scale))
        self.steer += (joy - self.steer) * (1 - math.exp(-DT / self.steer_lag))

        gain = np.interp(v, self.car["v"], self.car["steer_gain"]) * self.mismatch
        kappa = self.steer / gain
        lat_max = np.interp(v, self.car["v"], self.lat)
        self.sliding = v * v * abs(kappa) > lat_max
        if self.sliding:
            kappa = math.copysign(lat_max / max(v * v, 1.0), kappa)

        _, lat = self.probe.locate(self.x, self.z)
        acc = np.interp(v, self.car["v"], self.car["accel"])
        brk = np.interp(v, self.car["v"], self.car["brake"])
        a = gas * acc - brake * brk - (1 - gas) * (0.3 + 0.0004 * v * v)
        if abs(lat) > self.road_half:
            a -= 6.0                                    # grass
        if abs(lat) > self.wall:
            self.crashed = True
        self.v = max(0.0, v + a * DT)
        self.h += kappa * self.v * DT                   # + curvature (right) raises ACC heading
        self.x += -math.sin(self.h) * self.v * DT
        self.z += math.cos(self.h) * self.v * DT
