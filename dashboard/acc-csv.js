/* acc-csv.js — parse ac_telemetry reader.py CSVs into laps, racing lines and tracks.
   Shared by the ACC dashboard (ac_telemetry/dashboard) and ML Track Trainer (track import).
   Keep both copies identical. No dependencies. */
(function (global) {
  'use strict';

  const NO_TIME = 2e9;          // ACC uses INT_MAX for "no lap time yet"
  const TEXT = new Set(['session_type', 'lap_time_str']);

  /* ---------- parsing ---------- */

  // Returns { n, names, has(name), col(name) -> array, name }. Text columns stay strings, the rest are numbers.
  function parse(text, name) {
    let start = 0;
    const nextLine = () => {
      if (start >= text.length) return null;
      let end = text.indexOf('\n', start);
      if (end < 0) end = text.length;
      let line = text.slice(start, end);
      if (line.endsWith('\r')) line = line.slice(0, -1);
      start = end + 1;
      return line;
    };
    const header = nextLine().split(',');
    const idx = {};
    header.forEach((c, i) => { if (c) idx[c] = i; });
    if (idx.pos_x === undefined || idx.lap_fraction === undefined) {
      throw new Error('This CSV has no track coordinates. Record a new session with the updated reader.py.');
    }
    const cols = {};
    for (const c in idx) cols[c] = [];
    let line;
    while ((line = nextLine()) !== null) {
      if (!line) continue;
      const f = line.split(',');
      for (const c in idx) cols[c].push(TEXT.has(c) ? f[idx[c]] : +f[idx[c]]);
    }
    const n = cols.timestamp.length;
    return { n, name: name || 'session', names: Object.keys(cols), has: (c) => c in cols, col: (c) => cols[c] };
  }

  /* ---------- laps ---------- */

  const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : NaN; };

  function laps(data) {
    const lapCol = data.col('current_lap'), ts = data.col('timestamp');
    const dist = data.col('distance_m'), last = data.col('last_lap_ms'), valid = data.col('is_valid_lap');
    const pos = data.col('position'), done = data.col('completed_laps');
    const maxDone = Math.max(...done.slice(-50));
    const out = [];
    let i0 = 0;
    for (let i = 1; i <= data.n; i++) {
      if (i < data.n && lapCol[i] === lapCol[i0]) continue;
      out.push({ lap: lapCol[i0], i0, i1: i - 1, complete: lapCol[i0] <= maxDone,
                 length: dist[i - 1] - dist[i0], position: pos[i - 1],
                 valid: valid[Math.max(i0, i - 50)] === 0 ? 0 : 1, timeS: NaN, t0: ts[i0] });
      i0 = i;
    }
    // Official lap time appears in iLastTime once the next lap has started
    out.forEach((L, k) => {
      const nxt = out[k + 1];
      if (!L.complete || !nxt) return;
      for (let i = Math.min(nxt.i1, nxt.i0 + 400); i >= nxt.i0; i--) {
        if (last[i] > 0 && last[i] < NO_TIME) { L.timeS = last[i] / 1000; break; }
      }
      // is_valid_lap resets on the new lap; check the frames just before the line
      let v = 1;
      for (let i = Math.max(L.i0, L.i1 - 50); i <= L.i1; i++) if (valid[i] === 0) v = 0;
      L.valid = v;
    });
    const med = median(out.filter((L) => L.complete).map((L) => L.length));
    const first = out.length ? out[0].lap : 0;
    out.forEach((L) => {
      L.flying = L.complete && L.lap !== first && !isNaN(L.timeS) && Math.abs(L.length - med) < 0.03 * med;
    });
    const pool = out.filter((L) => L.flying && L.valid);
    const best = (pool.length ? pool : out.filter((L) => L.flying)).reduce(
      (b, L) => (!b || L.timeS < b.timeS ? L : b), null);
    return { list: out, lapMeters: med, best: best ? best.lap : null };
  }

  /* ---------- orientation ---------- */

  // World (x, z) -> screen (x, y). ACC steer is −1 left / +1 right, so pick the mirror in which
  // right-hand steering turns the car clockwise on screen.
  function orientation(data, lapInfo) {
    const x = data.col('pos_x'), z = data.col('pos_z'), st = data.col('steer'), sp = data.col('speed_kmh');
    const L = lapInfo.list.find((l) => l.lap === lapInfo.best) || lapInfo.list[0];
    let score = 0, px, py, ph;
    for (let i = L.i0; i <= L.i1; i += 10) {
      if (sp[i] < 50) { ph = undefined; continue; }
      if (px !== undefined) {
        const h = Math.atan2(z[i] - py, x[i] - px);
        if (ph !== undefined) {
          let dh = h - ph; if (dh > Math.PI) dh -= 2 * Math.PI; if (dh < -Math.PI) dh += 2 * Math.PI;
          score += dh * st[i];
        }
        ph = h;
      }
      px = x[i]; py = z[i];
    }
    // In screen coords (y down) a positive heading change is clockwise; keep z as y if that matches steer
    const flip = score < 0 ? -1 : 1;
    return (i) => ({ x: x[i], y: flip * z[i] });
  }

  /* ---------- traces ---------- */

  // Points for one lap with distance-into-lap from the game's track spline position.
  function trace(data, L, lapMeters, toScreen, step) {
    const frac = data.col('lap_fraction'), ts = data.col('timestamp');
    const pts = [];
    for (let i = L.i0; i <= L.i1; i += step || 1) {
      let f = frac[i];
      const t = ts[i] - L.t0;
      if (f > 0.9 && t < 10) f -= 1;          // spline position wraps a few frames before/after the lap counter
      if (f < 0.1 && t > 30) f += 1;
      const p = toScreen(i);
      if (p.x === 0 && p.y === 0) continue;   // car not in the car list (menus)
      pts.push({ i, x: p.x, y: p.y, d: f * lapMeters, t,
                 speed: data.col('speed_kmh')[i], gas: data.col('gas')[i], brake: data.col('brake')[i],
                 steer: data.col('steer')[i], gear: data.col('gear')[i] });
    }
    return pts;
  }

  // Value of key at distance d (linear interpolation, pts sorted by d).
  function sampleAt(pts, d, key) {
    let lo = 0, hi = pts.length - 1;
    if (!pts.length) return NaN;
    if (d <= pts[0].d) return pts[0][key];
    if (d >= pts[hi].d) return pts[hi][key];
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (pts[m].d <= d) lo = m; else hi = m; }
    const a = pts[lo], b = pts[hi], u = (d - a.d) / ((b.d - a.d) || 1);
    return a[key] + (b[key] - a[key]) * u;
  }

  // Mean line across laps on a fixed distance grid, smoothed over ~±10 m, with unit normals
  // and signed curvature (1/m; the curve's centre lies at +normal / curvature).
  // Mean, not per-axis median: taking x and y medians separately can mix laps and zig-zag.
  function reference(traces, lapMeters, stepM) {
    const step = stepM || 2, n = Math.floor(lapMeters / step);
    const ref = [];
    for (let k = 0; k < n; k++) {
      const d = k * step;
      const xs = traces.map((p) => sampleAt(p, d, 'x')), ys = traces.map((p) => sampleAt(p, d, 'y'));
      ref.push({ d, x: xs.reduce((s, v) => s + v, 0) / xs.length, y: ys.reduce((s, v) => s + v, 0) / ys.length });
    }
    smooth(ref, Math.ceil(10 / step));
    const w = Math.max(1, Math.ceil(4 / step));   // tangent from points ~±4 m apart
    const tang = ref.map((_, k) => {
      const a = ref[(k - w + n) % n], b = ref[(k + w) % n], l = Math.hypot(b.x - a.x, b.y - a.y) || 1;
      return { x: (b.x - a.x) / l, y: (b.y - a.y) / l };
    });
    for (let k = 0; k < n; k++) {
      ref[k].nx = -tang[k].y; ref[k].ny = tang[k].x;   // left of travel on screen
      const a = tang[(k - w + n) % n], b = tang[(k + w) % n];
      ref[k].curv = (a.x * b.y - a.y * b.x) / (2 * w * step);   // dθ/ds
    }
    return ref;
  }

  function smooth(pts, r) {
    const n = pts.length, xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
    for (let k = 0; k < n; k++) {
      let sx = 0, sy = 0;
      for (let j = -r; j <= r; j++) { sx += xs[(k + j + n) % n]; sy += ys[(k + j + n) % n]; }
      pts[k].x = sx / (2 * r + 1); pts[k].y = sy / (2 * r + 1);
    }
  }

  // Signed sideways offset (m) of a lap from the reference line; + = left of the reference.
  // Measured against the nearest reference point (searched ±30 m around the lap distance), so
  // small disagreements between lap distance and position don't leak into the sideways offset.
  // Also sets p.k = fractional reference index (projection onto the nearest segment).
  function offsets(pts, ref) {
    const n = ref.length, step = n > 1 ? ref[1].d : 1, W = Math.ceil(30 / step);
    return pts.map((p) => {
      const k0 = Math.round(p.d / step);
      let bk = 0, bd = Infinity;
      for (let j = -W; j <= W; j++) {
        const k = ((k0 + j) % n + n) % n, r = ref[k];
        const dd = (p.x - r.x) ** 2 + (p.y - r.y) ** 2;
        if (dd < bd) { bd = dd; bk = k; }
      }
      // Project onto the segment before or after the nearest point, whichever is closer
      let best = { k: bk, t: 0, dd: bd };
      [bk - 1, bk].forEach((s) => {
        const a = ref[(s + n) % n], b = ref[(s + 1) % n];
        const vx = b.x - a.x, vy = b.y - a.y, l2 = vx * vx + vy * vy || 1;
        const t = Math.max(0, Math.min(1, ((p.x - a.x) * vx + (p.y - a.y) * vy) / l2));
        const dd = (p.x - a.x - vx * t) ** 2 + (p.y - a.y - vy * t) ** 2;
        if (dd < best.dd) best = { k: (s + n) % n, t, dd };
      });
      p.k = best.k + best.t;
      const r = refAt(ref, p.k);
      return (p.x - r.x) * r.nx + (p.y - r.y) * r.ny;
    });
  }

  // Reference position, normal and curvature at a fractional index (linear interpolation, wraps).
  function refAt(ref, kf) {
    const n = ref.length, i = Math.floor(kf), t = kf - i;
    const a = ref[((i % n) + n) % n], b = ref[(((i + 1) % n) + n) % n];
    const nx = a.nx + (b.nx - a.nx) * t, ny = a.ny + (b.ny - a.ny) * t, l = Math.hypot(nx, ny) || 1;
    return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, nx: nx / l, ny: ny / l,
             curv: a.curv + (b.curv - a.curv) * t, d: a.d + (b.d - a.d) * t };
  }

  /* ---------- export for ML Track Trainer ---------- */

  // Closed centreline as ~targetPts control points, scaled to fit `size` px. Start/finish first.
  function trackForTrainer(ref, lapMeters, name, targetPts, size) {
    const N = targetPts || 120, S = size || 700;
    const step = Math.max(1, Math.floor(ref.length / N));
    const pts = ref.filter((_, k) => k % step === 0);
    const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
    const minX = Math.min(...xs), minY = Math.min(...ys);
    const scale = S / Math.max(Math.max(...xs) - minX, Math.max(...ys) - minY);
    return {
      name, lapMeters: Math.round(lapMeters), source: 'ac_telemetry',
      control: pts.map((p) => [Math.round((p.x - minX) * scale + 30), Math.round((p.y - minY) * scale + 30)]),
    };
  }

  // One call: CSV text -> trainer track (median line of the flying laps).
  function trackFromCsv(text, fileName) {
    const data = parse(text, fileName);
    const info = laps(data);
    if (info.best === null) throw new Error('No complete flying lap found in this CSV.');
    const toScreen = orientation(data, info);
    const tr = info.list.filter((L) => L.flying).map((L) => trace(data, L, info.lapMeters, toScreen, 5));
    const ref = reference(tr, info.lapMeters, 4);
    const m = /acc_session_\d+_\d+_(.+)\.csv$/i.exec(fileName || '');
    const track = m ? m[1].replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()) : 'ACC track';
    return trackForTrainer(ref, info.lapMeters, track);
  }

  global.AccCsv = { parse, laps, orientation, trace, sampleAt, reference, offsets, refAt, trackForTrainer, trackFromCsv };
})(typeof window !== 'undefined' ? window : globalThis);
