#!/usr/bin/env python3
"""
pilot_evolve.py: ML Track Trainer's neuroevolution, on the real ACC line.

Same network ([9, 12, 8, 2] tanh), same 9 inputs, same GA (64 drivers, 4 elites,
4 fresh, top 25% breed, 15% of weights mutated, decaying sigma) and the same shape
of reward as env.js. Two differences from the browser version:

  track  the reference line from your ACC laps (track_line.py), in real metres
  car    fitted to your telemetry (pilot_sim.py's model, vectorised over the population)

Road edges: with a <track>.edges.json (track_line.py --edges) next to the line, the road
is where you actually drove, per point and per side, plus --edge-margin; the wall is
--wall-margin past that. Without it, the old flat 7.5 m road / 12 m wall, which the
from-scratch genome exploited (9% of its lap past 7.5 m). Over the grip limit the car
scrubs speed (SCRUB) instead of clamping for free (15% of that lap sat at the clamp).

--wall-margin 0 makes leaving the road end the run, like ACC's Hotlap mode, which restarts
the lap at the first off. --steer-noise gives each car a random steering gain mismatch and
bias every generation, so a genome has to correct drift instead of relying on an exact car:
on 2 Nov both networks drifted 2-3 m wide on a near-straight live, steering almost straight.

The whole population is simulated at once with numpy. Best weights are saved in the
format ML Track Trainer keeps in localStorage ({"w": [...], "fit", "gen"}), plus the
wall distance the inputs were scaled by, so pilot.py --policy mltt can drive ACC with them.

  python pilot_evolve.py --gens 150
  python pilot.py --sim --policy mltt --weights tracks/mount_panorama.evolved.json
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from track_line import LOOK_M, TrackLine

LAYERS = (9, 12, 8, 2)
SIZE = sum(n * LAYERS[k] + n for k, n in enumerate(LAYERS[1:]))
CONFIG = dict(population=64, elites=4, fresh=4, parent_fraction=0.25, crossover=0.5,
              mutation_rate=0.15, sigma0=0.3, sigma_decay=0.99, sigma_min=0.08)
REWARD = dict(progress=1.0, time=1.0, grass=15.0, crash=500.0, stall=100.0, stall_time=5.0,
              lap_bonus=5000.0, ref_time=180.0, lap_time=60.0, max_time=300.0)
DT = 1 / 30
ROAD_HALF, WALL = 7.5, 12.0               # flat fallback when there's no edges file
SCRUB = 5.0                               # m/s^2 lost per 100% over the grip limit (capped at 100%)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ============================================================
# NETWORK + GA (port of brain.js)
# ============================================================

def random_genome(rng):
    w = np.zeros(SIZE)
    k = 0
    for nin, nout in zip(LAYERS, LAYERS[1:]):
        w[k:k + nin * nout] = rng.standard_normal(nin * nout) * math.sqrt(1 / nin)
        k += nin * nout + nout
    return w


def forward(pop, x):
    """pop: (P, SIZE) genomes, x: (P, 9) inputs -> (P, 2) outputs."""
    k = 0
    for nin, nout in zip(LAYERS, LAYERS[1:]):
        W = pop[:, k:k + nin * nout].reshape(-1, nout, nin); k += nin * nout
        b = pop[:, k:k + nout]; k += nout
        x = np.tanh(np.einsum("poi,pi->po", W, x) + b)
    return x


def evolve(ranked, gen, rng):
    C = CONFIG
    sigma = max(C["sigma_min"], C["sigma0"] * C["sigma_decay"] ** gen)
    nxt = [g.copy() for g in ranked[:C["elites"]]]
    nxt += [random_genome(rng) for _ in range(C["fresh"])]
    pool = ranked[:max(1, round(len(ranked) * C["parent_fraction"]))]
    while len(nxt) < C["population"]:
        a, b = pool[rng.integers(len(pool))], pool[rng.integers(len(pool))]
        child = np.where(rng.random(SIZE) < 0.5, a, b) if rng.random() < C["crossover"] else a.copy()
        m = rng.random(SIZE) < C["mutation_rate"]
        child[m] += rng.standard_normal(m.sum()) * sigma
        nxt.append(child)
    return np.array(nxt)


# ============================================================
# VECTORISED CAR (same model as pilot_sim.SimBackend)
# ============================================================

class Population:
    def __init__(self, track: TrackLine, n, start_s=10.0, steer_lag=0.10, edges=None, scrub=SCRUB,
                 steer_noise=0.0, rng=None):
        self.t = track
        # Per-point road and wall, each side of the line (+ = right)
        if edges is None:
            edges = dict(road_r=np.full(track.n, ROAD_HALF), road_l=np.full(track.n, ROAD_HALF),
                         wall_r=np.full(track.n, WALL), wall_l=np.full(track.n, WALL))
        self.e = edges
        self.scrub = scrub
        # Per car: steering gain mismatch (x1/(1+noise) .. x(1+noise)) and a bias of ~noise/10
        rng = rng or np.random.default_rng()
        k = 1 + steer_noise
        self.mis = np.exp(rng.uniform(-math.log(k), math.log(k), n)) if steer_noise else np.ones(n)
        self.bias = rng.normal(0, steer_noise / 10, n) if steer_noise else np.zeros(n)
        self.off_t = np.zeros(n); self.clamp_t = np.zeros(n); self.alive_t = np.zeros(n)
        c = track.meta["car"]
        self.cv = np.array(c["v"])
        self.acc, self.brk = np.array(c["accel"]), np.array(c["brake"])
        self.lat_max = np.maximum.accumulate(np.array(c["lat"]))
        self.gain = np.array(c["steer_gain"])
        self.alpha = 1 - math.exp(-DT / steer_lag)
        self.look_idx = np.round(np.array(LOOK_M) / track.grid).astype(int)
        i0 = int(start_s / track.grid)
        self.x = np.full(n, track.x[i0]); self.z = np.full(n, track.z[i0])
        self.h = np.full(n, track.h[i0]); self.v = np.zeros(n); self.steer = np.zeros(n)
        self.idx = np.full(n, i0); self.s = np.zeros(n); self.best_s = np.zeros(n)
        self.stall = np.zeros(n); self.R = np.zeros(n); self.t_el = 0.0
        self.alive = np.ones(n, bool); self.lap_time = np.full(n, np.nan)
        self.lat = np.zeros(n); self.reason = np.array([""] * n, dtype=object)

    def locate(self):
        T, W = self.t, 12
        cand = (self.idx[:, None] + np.arange(-W, W + 1)) % T.n
        d2 = (T.x[cand] - self.x[:, None]) ** 2 + (T.z[cand] - self.z[:, None]) ** 2
        new = cand[np.arange(len(cand)), d2.argmin(1)]
        step = (new - self.idx + T.n // 2) % T.n - T.n // 2
        self.idx = new
        h = T.h[new]
        self.lat = -((self.x - T.x[new]) * np.cos(h) + (self.z - T.z[new]) * np.sin(h))
        return step * T.grid

    def observe(self, wall_m):
        T, q = self.t, math.pi / 2
        o = np.empty((len(self.x), 9))
        o[:, 0] = self.v / 90
        o[:, 1] = np.clip(self.lat / wall_m, -1, 1)
        o[:, 2] = np.clip(wrap(self.h - T.h[self.idx]) / q, -1, 1)
        for k, di in enumerate(self.look_idx):
            o[:, 3 + k] = np.clip(wrap(T.h[(self.idx + di) % T.n] - self.h) / q, -1, 1)
        o[:, 8] = self.steer
        return o

    def step(self, out):
        a = self.alive
        pedal, target = out[:, 0], out[:, 1]
        gas, brake = np.maximum(pedal, 0), np.maximum(-pedal, 0)
        v = self.v
        self.steer = np.where(a, self.steer + (target - self.steer) * self.alpha, self.steer)
        kappa = (self.steer + self.bias) / (np.interp(v, self.cv, self.gain) * self.mis)
        lim = np.interp(v, self.cv, self.lat_max) / np.maximum(v * v, 1.0)
        over = np.clip(np.abs(kappa) / lim - 1, 0, 1)
        kappa = np.clip(kappa, -lim, lim)
        acc = gas * np.interp(v, self.cv, self.acc) - brake * np.interp(v, self.cv, self.brk) \
            - (1 - gas) * (0.3 + 0.0004 * v * v)
        e, i = self.e, self.idx
        off = (self.lat > e["road_r"][i]) | (-self.lat > e["road_l"][i])
        acc -= 6.0 * off + self.scrub * over
        self.off_t += DT * (a & off)
        self.clamp_t += DT * (a & (over > 0))
        self.alive_t += DT * a
        self.v = np.where(a, np.maximum(0, v + acc * DT), v)
        self.h = np.where(a, self.h + kappa * self.v * DT, self.h)
        self.x = np.where(a, self.x - np.sin(self.h) * self.v * DT, self.x)
        self.z = np.where(a, self.z + np.cos(self.h) * self.v * DT, self.z)
        ds = self.locate()
        self.t_el += DT

        R = REWARD
        r = R["progress"] * ds - R["time"] * DT - R["grass"] * DT * off
        self.s += np.where(a, ds, 0)
        better = self.s > self.best_s + 2
        self.best_s = np.where(better, self.s, self.best_s)
        self.stall = np.where(better, 0, self.stall + DT)

        e, i = self.e, self.idx
        crash = a & ((self.lat > e["wall_r"][i]) | (-self.lat > e["wall_l"][i]))
        lap = a & ~crash & (self.s >= self.t.lap_m * 0.999)
        stall = a & ~crash & ~lap & (self.stall > R["stall_time"])
        timeout = a & ~crash & ~lap & ~stall & (self.t_el > R["max_time"])
        r = r - R["crash"] * crash - R["stall"] * stall
        r = r + lap * (R["lap_bonus"] + R["lap_time"] * max(0, R["ref_time"] - self.t_el))
        self.lap_time = np.where(lap, self.t_el, self.lap_time)
        self.R += np.where(a, r, 0)
        for m, why in ((crash, "wall"), (lap, "lap"), (stall, "stall"), (timeout, "time")):
            self.reason[m] = why
        self.alive = a & ~(crash | lap | stall | timeout)


def load_edges(track_path, edge_margin, wall_margin):
    """Road and wall per point from <track>.edges.json, or None for the flat fallback."""
    p = Path(track_path).with_suffix(".edges.json")
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    r, l = np.array(d["right_m"]) + edge_margin, np.array(d["left_m"]) + edge_margin
    return dict(road_r=r, road_l=l, wall_r=r + wall_margin, wall_l=l + wall_margin, file=str(p))


def episode(track, pop, wall_m, edges=None, scrub=SCRUB, steer_noise=0.0, rng=None):
    cars = Population(track, len(pop), edges=edges, scrub=scrub, steer_noise=steer_noise, rng=rng)
    while cars.alive.any():
        cars.step(forward(pop, cars.observe(wall_m)))
    return cars


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", default="tracks/mount_panorama.json")
    ap.add_argument("--gens", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", help="weights JSON (default tracks/<name>.evolved.json)")
    ap.add_argument("--resume", action="store_true", help="seed the population from --out")
    ap.add_argument("--seed-from", help="seed the population from a genome JSON, e.g. ML Track Trainer's "
                                        "Export best model ({\"w\": [...], \"fit\", \"gen\"})")
    ap.add_argument("--edge-margin", type=float, default=0.5, help="m added to the measured road edges")
    ap.add_argument("--wall-margin", type=float, default=3.0, help="m from road edge to wall (crash)")
    ap.add_argument("--steer-noise", type=float, default=0.0,
                    help="per-car steering gain mismatch up to this fraction (e.g. 0.25), plus a small bias")
    ap.add_argument("--flat-edges", action="store_true",
                    help=f"ignore the edges file: flat {ROAD_HALF} m road / {WALL} m wall, no scrub (the old sim)")
    a = ap.parse_args()

    track = TrackLine(a.track)
    edges = None if a.flat_edges else load_edges(a.track, a.edge_margin, a.wall_margin)
    scrub = 0.0 if a.flat_edges else SCRUB
    print(f"Road edges: {edges['file']} +{a.edge_margin} m, wall +{a.wall_margin} m, scrub {scrub}" if edges
          else f"Road edges: flat {ROAD_HALF} m, wall {WALL} m, scrub {scrub}")
    out = Path(a.out or Path(a.track).with_suffix(".evolved.json"))
    rng = np.random.default_rng(a.seed)
    wall_m = WALL
    best = None
    if a.seed_from:
        d = json.loads(Path(a.seed_from).read_text())
        w = np.array(d["w"] if isinstance(d, dict) else d, dtype=float)
        if w.size != SIZE:
            raise SystemExit(f"{a.seed_from} has {w.size} weights; {list(LAYERS)} needs {SIZE}")
        # Its reward came from another track and car, so it isn't comparable: this run's first gen sets best
        pop = evolve([w], 0, rng)
        print(f"Seeded from {a.seed_from} (gen {d.get('gen', '?')} there)" if isinstance(d, dict) else
              f"Seeded from {a.seed_from}")
    elif a.resume and out.exists():
        best = json.loads(out.read_text())
        pop = evolve([np.array(best["w"])], 0, rng)
    else:
        pop = np.array([random_genome(rng) for _ in range(CONFIG["population"])])

    log_path = Path("training_logs") / f"evolve_{out.stem}.csv"
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["gen", "best_reward", "mean_reward", "best_progress", "laps", "best_lap_s", "crashes",
                      "top_off_pct", "top_clamp_pct"])
        for gen in range(a.gens):
            t0 = time.time()
            cars = episode(track, pop, wall_m, edges, scrub, a.steer_noise, rng)
            order = np.argsort(-cars.R)
            top = order[0]
            laps = int((cars.reason == "lap").sum())
            lap_s = np.nanmin(cars.lap_time) if laps else float("nan")
            log.writerow([gen, round(cars.R[top], 1), round(cars.R.mean(), 1),
                          round(cars.best_s.max() / track.lap_m, 3), laps, round(lap_s, 2),
                          int((cars.reason == "wall").sum()),
                          round(100 * cars.off_t[top] / cars.alive_t[top], 1), round(100 * cars.clamp_t[top] / cars.alive_t[top], 1)])
            f.flush()
            print(f"gen {gen:3d}  best {cars.R[top]:8.0f}  progress {cars.best_s.max() / track.lap_m:5.1%}"
                  f"  laps {laps:2d}  best lap {lap_s:6.1f}s  off {100 * cars.off_t[top] / cars.alive_t[top]:4.1f}%"
                  f"  ({time.time() - t0:.1f}s)")
            if best is None or cars.R[top] > best["fit"]:
                best = {"w": pop[top].round(6).tolist(), "fit": float(cars.R[top]), "gen": gen,
                        "wall_m": wall_m, "layers": list(LAYERS), "track": track.meta["name"],
                        "edges": edges["file"] if edges else "flat", "scrub": scrub,
                        "edge_margin": a.edge_margin, "wall_margin": a.wall_margin, "steer_noise": a.steer_noise,
                        "lap_s": None if np.isnan(cars.lap_time[top]) else float(cars.lap_time[top])}
                out.write_text(json.dumps(best))
            pop = evolve([pop[i] for i in order], gen, rng)
    print(f"Best reward {best['fit']:.0f} (gen {best['gen']}), saved {out}. Log: {log_path}")


if __name__ == "__main__":
    main()
