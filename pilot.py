#!/usr/bin/env python3
"""
pilot.py: closed-loop driver for ACC, built on the reference line (track_line.py).

The first attempt (driver.py) fed models local vehicle state only: speed, slip,
suspension. No idea where the road went next, so every policy reacted after the
problem existed. This one starts from structure:

  where am I?        world (x, z) -> distance along the line, lateral offset, heading error
  where does it go?  look-ahead angles and a pure-pursuit target point on the line
  how fast?          your speed trace x pace, capped by a braking envelope fitted to the car

Policies (--policy):
  classical   pure-pursuit steering + speed-envelope pedals. The foundation.
  mltt        an ML Track Trainer network (same 9 inputs, [9,12,8,2] tanh) drives;
              the speed envelope stays on as a governor and takes over if it leaves the line.

Backends:
  default     live ACC (shared memory in, vgamepad out)
  --sim       pilot_sim.py: a car fitted to your telemetry, on the same line. Test here first.

ACC setup (once): Assists -> automatic gearbox ON (or pass --manual-gears), TC/ABS on.
Controls -> gamepad: steer gamma 1.0, speed sensitivity 0, steer filter 0, deadzones 0.
Start a Practice / Hotlap session, drive onto the track, then run this.

  python pilot.py --sim                        # simulated lap, prints result
  python pilot.py --pace 0.80                  # live, 80% of your speed trace
  python pilot.py --pace 0.80 --laps 5         # five timed laps (0 = until Ctrl+C)

Laps run start/finish to start/finish. The drive from wherever the car starts to the
first crossing is an untimed out-lap; each crossing after that completes a timed lap.
Live, ACC's own lap time and lap validity (track limits) are recorded for each lap too.
  python pilot.py --policy mltt --weights best.json --pace 0.8
"""

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from track_line import TrackLine, wrap

UPDATE_HZ = 100
DT = 1.0 / UPDATE_HZ


# ============================================================
# SPEED ENVELOPE
# ============================================================

def speed_envelope(track: TrackLine, pace: float, brake_margin: float) -> np.ndarray:
    """
    Target speed (m/s) at every line point: your median speed x pace, lowered wherever
    the car couldn't brake down to the next corner using brake_margin x its best braking.
    """
    car = track.meta["car"]
    v_t = track.v_ref * pace
    grid = track.grid
    for _ in range(2):                                   # two passes so the wrap at s/f settles
        for i in range(track.n - 1, -1, -1):
            nxt = v_t[(i + 1) % track.n]
            b = brake_margin * np.interp(nxt, car["v"], car["brake"])
            v_t[i] = min(v_t[i], math.sqrt(nxt * nxt + 2 * b * grid))
    return v_t


# ============================================================
# POLICIES
# ============================================================

class Classical:
    """Pure pursuit on the reference line + speed tracking. No learning."""

    name = "classical"

    def __init__(self, track: TrackLine, look_gain=0.55, look_min=10.0, look_max=45.0):
        self.t = track
        self.car = track.meta["car"]
        self.look_gain, self.look_min, self.look_max = look_gain, look_min, look_max

    def steer(self, st, obs):
        v = st["speed"]
        ld = min(self.look_max, max(self.look_min, 6.0 + self.look_gain * v))
        s_t = obs["s"] + ld
        tx, tz = self.t.at(s_t, self.t.x), self.t.at(s_t, self.t.z)
        alpha = wrap(math.atan2(-(tx - st["x"]), tz - st["z"]) - st["heading"])   # + = target to the right
        kappa = 2.0 * math.sin(alpha) / ld
        gain = np.interp(v, self.car["v"], self.car["steer_gain"])
        return float(np.clip(gain * kappa, -1.0, 1.0))

    def act(self, st, obs):
        return None, self.steer(st, obs)                 # pedals come from the speed controller


class MLTrackTrainer:
    """
    A network exported from ML Track Trainer. Inputs are rebuilt exactly as env.js
    observe() does: speed/90, lateral/WALL, heading error and 5 look-ahead angles over
    (pi/2), current steer. Outputs [pedal, steer] in -1..1.

    Steering goes through the same first-order lag the network trained with (pilot_evolve.py,
    0.10 s), and input 8 is that internal steer, not ACC's measured steer. Feeding back ACC's
    steer made a fast loop: with ACC answering within a frame, the networks flipped between
    full lock left and right every other frame live (2 Nov) though they drove smoothly in the sim.

    Grip limit: the steer target is capped at grip_limit x the steer that reaches the car's
    measured cornering grip at this speed (steer_gain x lat_max / v^2, from your laps), the
    steering version of the speed envelope. Live, the networks asked for 2-3.5x that into
    Murray's and Griffins Bend, the car slid and ACC counter-steered; classical stays under
    0.92x it 99% of the time. 0 turns it off.
    """

    name = "mltt"
    LAYERS = (9, 12, 8, 2)

    def __init__(self, track: TrackLine, weights_path: str, width_factor=0.0053, verge=0.08, steer_lag=0.10,
                 grip_limit=2.0):
        d = json.loads(Path(weights_path).read_text())
        w = np.array(d["w"] if isinstance(d, dict) else d, dtype=float)
        size = sum(n * self.LAYERS[k] + n for k, n in enumerate(self.LAYERS[1:]))
        if w.size != size:
            raise SystemExit(f"Expected {size} weights for {self.LAYERS}, got {w.size}")
        self.mats = []
        k = 0
        for nin, nout in zip(self.LAYERS, self.LAYERS[1:]):
            W = w[k:k + nin * nout].reshape(nout, nin); k += nin * nout
            b = w[k:k + nout]; k += nout
            self.mats.append((W, b))
        # Same wall distance the network trained against, in metres (pilot_evolve.py stores it;
        # browser exports use ML Track Trainer's road width)
        self.wall_m = d.get("wall_m") if isinstance(d, dict) and d.get("wall_m") else \
            track.lap_m * width_factor * (0.5 + verge)
        self.alpha = 1 - math.exp(-DT / steer_lag)
        self.str = 0.0                                    # internal steer: the lagged network target
        car = track.meta["car"]
        self.cv, self.gain = np.array(car["v"]), np.array(car["steer_gain"])
        self.lat_max = np.maximum.accumulate(np.array(car["lat"]))
        self.grip_limit = grip_limit
        self.capped = 0                                   # frames where the cap cut the target

    def act(self, st, obs):
        q = math.pi / 2
        x = np.array([st["speed"] / 90, np.clip(obs["lat"] / self.wall_m, -1, 1), np.clip(obs["herr"] / q, -1, 1)]
                     + [np.clip(a / q, -1, 1) for a in obs["look"]] + [self.str])
        for W, b in self.mats:
            x = np.tanh(W @ x + b)
        target = float(x[1])
        if self.grip_limit:
            v = st["speed"]
            cap = self.grip_limit * np.interp(v, self.cv, self.gain) * np.interp(v, self.cv, self.lat_max) / max(v, 5.0) ** 2
            if abs(target) > cap:
                target = math.copysign(cap, target)
                self.capped += 1
        self.str += (target - self.str) * self.alpha
        return float(x[0]), self.str

    def sync(self, steer):
        """The fallback steered this frame: carry on from there when the network takes back over."""
        self.str = float(np.clip(steer, -1, 1))


# ============================================================
# STEER COMPENSATION
# ============================================================

STEER_GAIN_PATH = Path("tracks/acc_steer_gain.json")


class SteerComp:
    """
    ACC's gamepad steering scales the joystick down as speed rises (measured on the
    Bathurst runs: ~1.0 at walking pace, 0.48 at 60 km/h, 0.16 at 190 km/h). Divide the
    requested steer by that gain so ACC's steer matches what the policy asked for.

    gain[v] = ACC steer / joystick, per 5 m/s band. Learned online from unsaturated frames
    and saved after live runs, so changing ACC's gamepad settings is picked up automatically.
    """

    V = np.arange(2.5, 80.0, 5.0)

    def __init__(self, gain=None, adapt=True, rate=0.003, floor=0.12):
        self.gain = np.ones_like(self.V) if gain is None else np.interp(self.V, *gain)
        self.adapt, self.rate, self.floor = adapt, rate, floor
        self.last_joy = 0.0

    @classmethod
    def load(cls, path=STEER_GAIN_PATH, **kw):
        try:
            d = json.loads(Path(path).read_text())
            return cls((np.array(d["v"]), np.array(d["gain"])), **kw)
        except (OSError, ValueError, KeyError):
            return cls(**kw)

    def save(self, path=STEER_GAIN_PATH):
        Path(path).parent.mkdir(exist_ok=True)
        Path(path).write_text(json.dumps({"v": self.V.tolist(), "gain": self.gain.round(3).tolist(),
                                          "saved": datetime.now().isoformat(timespec="seconds")}))

    def joystick(self, steer, v):
        return steer / max(float(np.interp(v, self.V, self.gain)), self.floor)

    def learn(self, v, measured):
        """Update the band for speed v from ACC's steer response to last frame's joystick."""
        j = self.last_joy
        if not self.adapt or not 0.05 < abs(j) < 0.95:
            return
        r = measured / j
        if 0.05 < r < 1.5:                               # ignore sign flips (slides, kerb kicks)
            i = int(np.argmin(np.abs(self.V - v)))
            self.gain[i] += self.rate * (r - self.gain[i])


# ============================================================
# DRIVER
# ============================================================

class Pilot:
    def __init__(self, track, policy, pace=0.85, brake_margin=0.8, preview_s=0.25,
                 recover_lat=6.0, recover_speed=18.0, steer_comp=None):
        self.t = track
        self.policy = policy
        self.car = track.meta["car"]
        self.v_env = speed_envelope(track, pace, brake_margin)
        self.preview_s = preview_s
        self.recover_lat = recover_lat
        self.recover_speed = recover_speed
        self.steer_i = 0.0                                # inner loop: commanded vs measured steer
        self.comp = steer_comp                            # feed-forward for ACC's speed-scaled steering
        self.fallback = policy if isinstance(policy, Classical) else Classical(track)

    def target_speed(self, s, v):
        return self.t.at(s + v * self.preview_s, self.v_env)

    def pedals(self, v, v_target):
        acc_max = np.interp(v, self.car["v"], self.car["accel"])
        brk_max = np.interp(v, self.car["v"], self.car["brake"])
        a_des = 1.5 * (v_target - v)
        if a_des > -0.5:
            return float(np.clip(0.3 + a_des / max(acc_max, 0.5), 0, 1)), 0.0
        return 0.0, float(np.clip(-a_des / brk_max, 0, 1))

    def step(self, st):
        obs = self.t.observe(st["x"], st["z"], st["heading"], st["speed"], st["steer"], st.get("lap_fraction"))
        v = st["speed"]
        v_target = self.target_speed(obs["s"], v)
        lost = abs(obs["lat"]) > self.recover_lat or abs(obs["herr"]) > 0.6
        if lost:
            v_target = min(v_target, self.recover_speed)

        pedal, steer = self.policy.act(st, obs)
        gas, brake = self.pedals(v, v_target)
        if pedal is not None and not lost:
            # Learned pedal, but never faster than the envelope allows
            gas, brake = (max(0.0, pedal), max(0.0, -pedal))
            if v > v_target + 1.0:
                gas, brake = self.pedals(v, v_target)
        if lost:
            steer = self.fallback.steer(st, obs)
            if hasattr(self.policy, "sync"):
                self.policy.sync(steer)

        # Feed-forward for ACC's steering scale, then an inner loop that trims what's left
        self.steer_i = float(np.clip(self.steer_i + 2.0 * (steer - st["steer"]) * DT, -0.25, 0.25))
        want = steer + self.steer_i                       # in ACC's steer units
        if self.comp is not None:
            self.comp.learn(v, st["steer"])
            want = self.comp.joystick(want, v)
        joy = float(np.clip(want, -1, 1))
        if self.comp is not None:
            self.comp.last_joy = joy
        return {**obs, "gas": gas, "brake": brake, "steer": joy, "steer_des": steer,
                "v_target": v_target, "lost": lost}


# ============================================================
# BACKENDS
# ============================================================

class AccBackend:
    """Live ACC: shared memory in, virtual Xbox pad out."""

    def __init__(self, manual_gears=False):
        import vgamepad as vg
        from acc_memory import (GRAPHICS_SHM, PHYSICS_SHM, STATIC_SHM, SharedMemoryPage,
                                SPageFileGraphic, SPageFilePhysics, SPageFileStatic, player_coordinates)
        self.vg = vg
        self.coords = player_coordinates
        self.phys = SharedMemoryPage(SPageFilePhysics, PHYSICS_SHM)
        self.gfx = SharedMemoryPage(SPageFileGraphic, GRAPHICS_SHM)
        static = SharedMemoryPage(SPageFileStatic, STATIC_SHM).read()
        self.max_rpm = static.maxRpm or 9000
        self.track_name = static.track
        self.pad = vg.VX360Gamepad()
        self.manual_gears = manual_gears
        self.last_shift = 0.0
        self.pending = None

    def read(self):
        p, g = self.phys.read(), self.gfx.read()
        xyz = self.coords(g)
        if xyz is None:
            return None
        return {"x": xyz[0], "z": xyz[2], "heading": p.heading, "speed": p.speedKmh / 3.6,
                "steer": p.steerAngle, "lap_fraction": g.normalizedCarPosition, "rpm": p.rpm,
                "gear": p.gear - 1, "laps": g.completedLaps, "status": g.status,
                "in_pit_lane": g.isInPitLane, "tyres_out": p.numberOfTyresOut,
                "auto_shift": p.autoShifterOn, "lap_ms": g.iCurrentTime, "last_ms": g.iLastTime,
                "valid": g.isValidLap, "ignition": p.ignitionOn, "engine_running": p.isEngineRunning}

    def write(self, gas, brake, steer, st=None):
        pad = self.pad
        pad.right_trigger_float(gas)
        pad.left_trigger_float(brake)
        pad.left_joystick_float(steer, 0.0)
        if self.manual_gears and st is not None:
            self._shift(st, brake)
        pad.update()

    def _shift(self, st, brake):
        """RPM-based shifting on the shoulder buttons; a press lasts one 80 ms tick."""
        now = time.time()
        B = self.vg.XUSB_BUTTON
        if self.pending and now - self.last_shift > 0.08:
            self.pad.release_button(self.pending)
            self.pending = None
        if self.pending or now - self.last_shift < 0.35:
            return
        r, gear = st["rpm"] / self.max_rpm, st["gear"]
        btn = None
        if gear < 1 or (r > 0.94 and gear < 6):
            btn = B.XUSB_GAMEPAD_RIGHT_SHOULDER
        elif gear > 1 and r < (0.62 if brake > 0.1 else 0.45):
            btn = B.XUSB_GAMEPAD_LEFT_SHOULDER
        if btn:
            self.pad.press_button(btn)
            self.pending, self.last_shift = btn, now

    def tap(self, btn, hold=0.1):
        self.pad.press_button(btn)
        self.pad.update()
        time.sleep(hold)
        self.pad.release_button(btn)
        self.pad.update()

    def startup(self, starter_gas=0.3, crank_s=6.0, idle_rpm=800):
        """
        Ignition (X), then hold the starter (A) with a little throttle until the engine
        catches, then upshift (RB) into 1st if in neutral. Prints what it reads at each step.
        """
        B = self.vg.XUSB_BUTTON
        st = self.read()
        if st is None:
            raise SystemExit("Car not found in shared memory. Are you on track?")
        print(f"Startup: ignition {'on' if st['ignition'] else 'off'}, engine "
              f"{'running' if st['engine_running'] else 'off'}, RPM {st['rpm']:.0f}, gear {st['gear']} (-1 R, 0 N, 1 = 1st)")
        if st["rpm"] < 500 and not st["engine_running"]:
            # X toggles the ignition, so only press it when it's off
            if st["ignition"]:
                print("🔑 Ignition already on")
            else:
                print("🔑 Ignition (X)")
                self.tap(B.XUSB_GAMEPAD_X)
                time.sleep(1.0)
                st = self.read() or st
                if not st["ignition"]:
                    print("⚠️  ACC still reports ignition off; trying the starter anyway")
            for attempt in range(2):
                print(f"🔑 Starter (hold A, {starter_gas:.0%} gas)", end="", flush=True)
                self.pad.press_button(B.XUSB_GAMEPAD_A)
                self.pad.right_trigger_float(starter_gas)
                self.pad.update()
                # Hold until it runs on its own: ACC says running and RPM is above cranking speed
                # (~500-600) for 0.3 s. Releasing at the first 500 RPM let it die.
                t0, ok_since = time.time(), None
                while time.time() - t0 < crank_s:
                    time.sleep(0.05)
                    st = self.read() or st
                    if st["engine_running"] and st["rpm"] >= idle_rpm:
                        ok_since = ok_since or time.time()
                        if time.time() - ok_since >= 0.3:
                            break
                    else:
                        ok_since = None
                self.pad.release_button(B.XUSB_GAMEPAD_A)
                self.pad.right_trigger_float(0.0)
                self.pad.update()
                time.sleep(0.5)
                st = self.read() or st
                print(f" -> released after {time.time() - t0:.1f} s, RPM {st['rpm']:.0f} 0.5 s later")
                if st["engine_running"] and st["rpm"] >= 500:
                    break
                time.sleep(0.5)
            if not (st["engine_running"] and st["rpm"] >= 500):
                raise SystemExit("Engine didn't start (RPM still under 500). Check ACC's ignition (X) "
                                 "and starter (A) bindings, or start it by hand and untick automatic startup.")
            print(f"✅ Engine running ({st['rpm']:.0f} RPM)")
            time.sleep(1.0)                               # let the idle settle before shifting
        else:
            print("✅ Engine already running")
        for _ in range(3):
            st = self.read() or st
            if st["gear"] >= 1:
                break
            print(f"⬆️  Shift up (RB) from gear {st['gear']}", end="", flush=True)
            self.tap(B.XUSB_GAMEPAD_RIGHT_SHOULDER, hold=0.15)
            time.sleep(0.5)
            st = self.read() or st
            print(f" -> gear {st['gear']}")
        print(f"✅ In gear {st['gear']}" if st["gear"] >= 1 else
              f"⚠️  Still in gear {st['gear']}; the pilot will shift once it drives (auto gearbox or --manual-gears)")
        time.sleep(0.5)
        st = self.read() or st
        if st["rpm"] < 300:
            raise SystemExit(f"Engine stalled after the shift (RPM {st['rpm']:.0f}). The pad has no clutch: "
                             "turn on ACC's auto-clutch assist, or start in gear by hand and untick automatic startup.")
        print(f"✅ Engine still running in gear ({st['rpm']:.0f} RPM)")

    def close(self):
        self.pad.reset()
        self.pad.update()
        self.phys.close()
        self.gfx.close()


# ============================================================
# RUN LOOP
# ============================================================

def fmt_lap(t):
    return f"{int(t // 60)}:{t % 60:06.3f}"


def run(pilot, backend, laps=1, log_path=None, realtime=True, max_s=400.0, quiet=False, launch=None):
    """
    Drive `laps` timed laps (0 = until Ctrl+C), start/finish to start/finish, unless the
    car is stuck or crashes. The drive to the first crossing is an untimed out-lap.
    launch=(gas, speed m/s): hold at least that throttle from rest until that speed, so
    the car pulls away cleanly before the policy's pedals take over (it steers throughout).
    Returns a summary dict with per-lap results.
    """
    st = backend.read()
    if st is None:
        raise SystemExit("Car not found in shared memory. Are you on track?")
    obs = pilot.t.observe(st["x"], st["z"], st["heading"], st["speed"], st["steer"], st.get("lap_fraction"))
    if abs(obs["lat"]) > 10:
        raise SystemExit(f"Car is {obs['lat']:.0f} m from the reference line (pit lane?). Start on track.")

    log = None
    if log_path:
        Path(log_path).parent.mkdir(exist_ok=True)
        f = open(log_path, "w", newline="")
        log = csv.writer(f)
        log.writerow(["t", "s", "lat", "herr", "speed", "v_target", "gas", "brake", "steer_des", "steer_cmd",
                      "steer_meas", "lost", "x", "z", "gear", "rpm", "tyres_out", "lap", "acc_laps", "acc_valid"])

    L = pilot.t.lap_m
    s_prev, travelled = obs["s"], 0.0
    s0 = obs["s"]                                         # a crossing is (s0 + travelled) passing k * L
    crossings = 0
    # ACC's lap counter is matched to our crossings by time, not by order: in Practice the
    # out-lap counts as a lap, in Hotlap it doesn't. acc_ok = validity since ACC's last increment.
    acc_laps, acc_ok, acc_events, cross_t = st.get("laps"), 1, [], []

    def match_acc():
        for ev in acc_events[:]:
            near = [i for i, t in enumerate(cross_t) if abs(t - ev["t"]) < 5.0]
            if near:
                i = near[0]
                if 1 <= i <= len(done):                   # crossing i ends timed lap i
                    done[i - 1]["acc_s"] = ev["ms"] / 1000.0 if ev["ms"] else None
                    done[i - 1]["acc_valid"] = ev["valid"]
                if i == 0 or i <= len(done):
                    acc_events.remove(ev)
            elif sim_t - ev["t"] > 5.0:
                acc_events.remove(ev)                     # no crossing of ours nearby: ignore it
    sim_t, stuck, lost_t, max_lat, frames = 0.0, 0.0, 0.0, 0.0, 0
    # Live, the clock is real time: Windows' sleep overshoots, so the loop runs ~3.5% under
    # 100 Hz and counting frames made laps ~5 s short of ACC's time. The sim steps exactly DT.
    t_start = time.perf_counter()
    lats, done = [], []
    cur = None                                            # lap in progress; None during the out-lap
    finish_t = None
    result = "stopped"
    try:
        while True:
            loop_t = time.time()
            st = backend.read()
            if st is None:
                continue
            out = pilot.step(st)
            if (launch is not None and st.get("rpm") is not None and st["rpm"] < 300 and sim_t > 1.0
                    and st["speed"] < 1.0):
                result = "stalled"
                print("Engine stalled at the start (RPM under 300). Restart it, or check ACC's auto-clutch assist.")
                break
            if launch is not None:
                if st["speed"] < launch[1]:
                    out["gas"], out["brake"] = max(out["gas"], launch[0]), 0.0
                else:
                    launch = None
                    if not quiet:
                        print(f"🏁 Launched at {st['speed'] * 3.6:.0f} km/h; policy has the pedals")
            backend.write(out["gas"], out["brake"], out["steer"], st)

            ds = (out["s"] - s_prev + L / 2) % L - L / 2
            if abs(ds) > 50.0:
                # Teleported: ACC put the car back (Hotlap restarts the lap after an off)
                result = "reset"
                print("ACC reset the car (in Hotlap, leaving the track restarts the lap). Stopping.")
                break
            travelled += ds
            s_prev = out["s"]
            max_lat = max(max_lat, abs(out["lat"]))
            lats.append(abs(out["lat"]))
            now = time.perf_counter() - t_start if realtime else sim_t + DT
            dt, sim_t = now - sim_t, now
            lost_t += dt if out["lost"] else 0.0
            stuck = stuck + dt if st["speed"] < 2.0 else 0.0
            frames += 1
            if cur is not None:
                cur["lats"].append(abs(out["lat"]))
                cur["lost"] += dt if out["lost"] else 0.0

            if acc_laps is not None and st.get("laps") is not None and st["laps"] > acc_laps:
                acc_laps = st["laps"]
                acc_events.append({"t": sim_t, "ms": st.get("last_ms"), "valid": bool(acc_ok)})
                acc_ok = 1
            elif st.get("valid") is not None:
                acc_ok = min(acc_ok, int(st["valid"]))

            # Our own start/finish crossing (reversing back over the line can't count twice)
            k = int((s0 + travelled) // L)
            if k > crossings:
                crossings = k
                cross_t.append(sim_t)
                if cur is not None:
                    a = np.array(cur["lats"])
                    n = len(done) + 1
                    lap = {"lap": n, "time_s": round(sim_t - cur["t0"], 3), "acc_s": None,
                           "median_lat_m": round(float(np.median(a)), 3), "max_lat_m": round(float(a.max()), 3),
                           "recover_s": round(cur["lost"], 2), "acc_valid": None}
                    done.append(lap)
                    match_acc()
                    if not quiet:
                        print(f"LAP {n}  {fmt_lap(lap['time_s'])}  median {lap['median_lat_m']:.2f} m"
                              f"  max {lap['max_lat_m']:.1f} m  recovering {lap['recover_s']:.1f} s"
                              + ("  INVALID (ACC)" if lap["acc_valid"] is False else ""))
                elif not quiet:
                    print(f"Start/finish after a {sim_t:.1f} s out-lap. Timing starts.")
                cur = {"t0": sim_t, "lats": [], "lost": 0.0}
            if acc_events:
                match_acc()

            if log:
                log.writerow([f"{sim_t:.2f}", f"{out['s']:.1f}", f"{out['lat']:.2f}", f"{out['herr']:.3f}",
                              f"{st['speed']:.2f}", f"{out['v_target']:.2f}", f"{out['gas']:.2f}",
                              f"{out['brake']:.2f}", f"{out['steer_des']:.3f}", f"{out['steer']:.3f}",
                              f"{st['steer']:.3f}", int(out["lost"]), f"{st['x']:.2f}", f"{st['z']:.2f}",
                              st.get("gear", ""), f"{st.get('rpm', 0):.0f}", st.get("tyres_out", ""),
                              len(done) + 1 if cur is not None else 0, st.get("laps", ""), st.get("valid", "")])
            if not quiet and frames % UPDATE_HZ == 0:
                lap_no = f"lap {len(done) + 1}" if cur is not None else "out-lap"
                print(f"{sim_t:6.1f}s  {lap_no:7s} {100 * ((s0 + travelled) % L) / L:5.1f}%  "
                      f"{st['speed'] * 3.6:5.0f}/{out['v_target'] * 3.6:3.0f} km/h  lat {out['lat']:+5.1f} m"
                      f"  steer {out['steer_des']:+.2f}" + ("  RECOVER" if out["lost"] else ""))

            if laps and len(done) >= laps:
                # Live, give ACC up to 3 s to register the last lap so its official time is kept
                finish_t = sim_t if finish_t is None else finish_t
                if acc_laps is None or done[-1]["acc_valid"] is not None or sim_t - finish_t > 3.0:
                    result = "laps"
                    break
            if st.get("crashed") or stuck > 8.0 or (laps and sim_t > max_s * (laps + 1)):
                result = "crash" if st.get("crashed") else "stuck" if stuck > 8.0 else "timeout"
                break
            if realtime:
                time.sleep(max(0.0, DT - (time.time() - loop_t)))
    except KeyboardInterrupt:
        result = "interrupted"
        print("\nInterrupted" + (" (ACC hadn't counted the last lap yet, so it has no ACC time)"
                                 if acc_laps is not None and done and done[-1]["acc_valid"] is None else ""))
    finally:
        backend.write(0.0, 0.0, 0.0)
        if log:
            f.close()

    times = [l["acc_s"] or l["time_s"] for l in done]
    return {"result": result, "time_s": sim_t, "progress": travelled / L, "laps_done": len(done),
            "best_lap_s": min(times) if times else None, "laps": done,
            "max_lat_m": max_lat, "median_lat_m": float(np.median(lats)) if lats else None,
            "recover_s": lost_t}


def confirm_start(auto, summary):
    """Terminal toggle before a live run: Space ticks/unticks automatic startup, Enter starts, Q quits."""
    print(summary)
    try:
        import msvcrt
    except ImportError:                                   # not Windows: plain prompt
        ans = input(f"Automatic startup (ignition, starter, 1st gear)? [{'Y/n' if auto else 'y/N'}] ").strip().lower()
        auto = auto if not ans else ans.startswith("y")
        input("Enter to start, Ctrl+C to cancel ")
        return auto
    while True:
        box = "x" if auto else " "
        print(f"\r  [{box}] Automatic startup (ignition, starter, 1st gear)   Space: toggle   Enter: start   Q: quit ",
              end="", flush=True)
        k = msvcrt.getwch()
        if k == " ":
            auto = not auto
        elif k in "\r\n":
            print()
            return auto
        elif k.lower() == "q" or k == "\x03":
            print()
            raise SystemExit("Cancelled")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", default="tracks/mount_panorama.json")
    ap.add_argument("--policy", choices=["classical", "mltt"], default="classical")
    ap.add_argument("--weights", help="ML Track Trainer weights JSON ({\"w\": [...]})")
    ap.add_argument("--pace", type=float, default=0.85, help="fraction of your median speed trace")
    ap.add_argument("--brake-margin", type=float, default=0.8, help="fraction of the car's best braking")
    ap.add_argument("--laps", type=int, default=1, help="timed laps, start/finish to start/finish (0 = until Ctrl+C)")
    ap.add_argument("--start-s", type=float, help="sim only: start this far along the line, in m "
                                                   "(default 150 m before start/finish)")
    ap.add_argument("--manual-gears", action="store_true", help="shift on RPM (if ACC auto gearbox is off)")
    ap.add_argument("--sim", action="store_true", help="drive the simulated car instead of ACC")
    ap.add_argument("--steer-mismatch", type=float, default=1.0, help="sim only: car needs this x more steer")
    ap.add_argument("--acc-steer", action="store_true",
                    help=f"sim only: scale the joystick down with speed like ACC's gamepad ({STEER_GAIN_PATH})")
    ap.add_argument("--grip-limit", type=float, default=2.0,
                    help="mltt: cap the network's steer at this x the car's measured grip at each speed (0 = off)")
    ap.add_argument("--steer-lag", type=float, default=0.10,
                    help="sim only: how fast the car's steering follows the joystick, s (live ACC is ~0.01)")
    ap.add_argument("--no-steer-comp", action="store_true", help="don't compensate for ACC's speed-scaled steering")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--launch-gas", type=float, default=0.7,
                    help="live: minimum throttle from rest until --launch-kmh, so the car doesn't stall (0 = off)")
    ap.add_argument("--launch-kmh", type=float, default=30.0, help="live: speed where the policy takes the pedals")
    ap.add_argument("--no-auto-start", action="store_true",
                    help="live: leave automatic startup unticked on the start screen")
    a = ap.parse_args()

    track = TrackLine(a.track)
    policy = MLTrackTrainer(track, a.weights, grip_limit=a.grip_limit) if a.policy == "mltt" else Classical(track)
    # Live starts from the gains learned on earlier runs; the sim starts neutral and has to learn them
    comp = None if a.no_steer_comp else SteerComp() if a.sim else SteerComp.load()
    pilot = Pilot(track, policy, pace=a.pace, brake_margin=a.brake_margin, steer_comp=comp)
    log = None if a.no_log else f"training_logs/pilot_{a.policy}_{datetime.now():%Y%m%d_%H%M%S}.csv"

    if a.sim:
        from pilot_sim import SimBackend
        acc = SteerComp.load() if a.acc_steer else None
        backend = SimBackend(track, start_s=a.start_s if a.start_s is not None else track.lap_m - 150,
                             steer_mismatch=a.steer_mismatch, steer_lag=a.steer_lag,
                             steer_scale=(acc.V, acc.gain) if acc is not None else None)
        res = run(pilot, backend, laps=a.laps, log_path=log, realtime=False, quiet=True)
    else:
        backend = AccBackend(manual_gears=a.manual_gears)
        norm = lambda s: s.lower().replace(" ", "_")
        if norm(backend.track_name) != norm(track.meta["name"]):
            backend.close()
            raise SystemExit(
                f"ACC is on '{backend.track_name}' but {a.track} is '{track.meta['name']}'.\n"
                f"Load {track.meta['name']} in ACC, or record 2+ clean laps here with reader.py and build a line:\n"
                f"  python track_line.py telemetry_logs/*{norm(backend.track_name)}.csv "
                f"--out tracks/{norm(backend.track_name)}.json")
        st = backend.read()
        if st and not st["auto_shift"] and not a.manual_gears:
            # Without this the car stays in 1st (the networks hit the limiter at ~130 km/h on 2 Nov)
            backend.manual_gears = True
            print("ACC automatic gearbox is off: the pilot will shift on RPM (same as --manual-gears).")
        try:
            auto = confirm_start(not a.no_auto_start,
                                 f"ACC track: {backend.track_name} | policy {policy.name} | pace {a.pace:.0%}"
                                 f" | {a.laps or 'unlimited'} lap(s)")
            print("Starting in 3 s...")
            time.sleep(3)
            if auto:
                backend.startup()
        except (SystemExit, KeyboardInterrupt):
            backend.close()
            raise
        launch = (a.launch_gas, a.launch_kmh / 3.6) if a.launch_gas > 0 else None
        try:
            res = run(pilot, backend, laps=a.laps, log_path=log, launch=launch)
        finally:
            backend.close()
            if comp is not None:
                comp.save()
                print(f"Steer gain (ACC/joystick) at 25/50/100/150/200 km/h: " + " ".join(
                    f"{np.interp(k / 3.6, comp.V, comp.gain):.2f}" for k in (25, 50, 100, 150, 200))
                      + f"  -> {STEER_GAIN_PATH}")

    print(f"\n{res['result'].upper()}  {res['laps_done']}/{a.laps or 'unlimited'} timed lap(s),"
          f" {res['time_s']:.1f} s total | max lateral {res['max_lat_m']:.1f} m | recovering {res['recover_s']:.1f} s")
    for l in res["laps"]:
        acc = f"  ACC {fmt_lap(l['acc_s'])}" if l["acc_s"] else ""
        valid = {None: "", True: "  valid", False: "  INVALID"}[l["acc_valid"]]
        print(f"  lap {l['lap']}: {fmt_lap(l['time_s'])}{acc}  median {l['median_lat_m']:.2f} m"
              f"  max {l['max_lat_m']:.1f} m  recovering {l['recover_s']:.1f} s{valid}")
    if res["best_lap_s"]:
        print(f"  best {fmt_lap(res['best_lap_s'])}")
    if log:
        # Sidecar: what was run and how it went, so a log can be traced without guessing (docs/ml_record.md)
        side = {"log": Path(log).name, "mode": "sim" if a.sim else "live", "policy": a.policy,
                "weights": a.weights, "track": a.track, "pace": a.pace, "brake_margin": a.brake_margin,
                "steer_comp": not a.no_steer_comp, "acc_steer": a.acc_steer if a.sim else None,
                "steer_mismatch": a.steer_mismatch if a.sim else None,
                **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}}
        Path(log).with_suffix(".json").write_text(json.dumps(side, indent=1))
        print(f"Log: {log} (+ .json summary)")


if __name__ == "__main__":
    main()
