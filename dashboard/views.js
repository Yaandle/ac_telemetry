/* views.js — Race / Driving / Car / Data views of the ACC dashboard.
   Brings the terminal summary, Streamlit pages and plot_session.py graphs into the browser. */
(function () {
  'use strict';

  const NS = 'http://www.w3.org/2000/svg';
  const $ = (id) => document.getElementById(id);
  const el = (tag, attrs, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  };
  const txt = (parent, x, y, s, attrs) => { const t = el('text', Object.assign({ x, y }, attrs || {}), parent); t.textContent = s; return t; };
  const S = AccDash.state, fmtLap = AccDash.fmtLap;
  const WHEELS = ['fl', 'fr', 'rl', 'rr'];
  const WHEEL_COLOURS = { fl: '#2f6db5', fr: '#c79a1e', rl: '#3a9a5b', rr: '#b5522b' };
  const nice = (v) => Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2);
  let view = 'map';

  /* ---------- tabs ---------- */

  $('tabs').addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    view = b.dataset.view;
    document.querySelectorAll('#tabs button').forEach((x) => x.classList.toggle('on', x === b));
    document.querySelectorAll('.view-body').forEach((v) => v.classList.toggle('hidden', v.id !== 'v-' + view));
    document.querySelectorAll('.map-only').forEach((x) => x.classList.toggle('hidden', view !== 'map'));
    if (view === 'map') { AccDash.refreshMap(); renderMapMore(); } else render();
  });
  // Called after every change; why === 'cursor' when only the inspected point moved
  AccDash.hooks.push((why) => {
    if (view === 'map') renderMapMore(why);
    else if (why !== 'cursor') render();
    else if (view === 'driving') renderInputs();
  });
  window.addEventListener('resize', () => { if (!S.data) return; if (view === 'map') renderMapMore(); else render(); });

  function render() {
    if (!S.data) return;
    if (!S.lapStats || S.lapStats.src !== S.data) S.lapStats = lapStats();
    ({ race: renderRace, driving: renderDriving, car: renderCar, data: renderData })[view]();
  }

  /* ---------- chart primitives ---------- */

  function frame(svg, xs, ys, o) {
    svg.innerHTML = '';
    const r = svg.getBoundingClientRect(), W = r.width, H = r.height;
    const m = { l: 48, r: 12, t: 10, b: 22 };
    let [x0, x1] = xs, [y0, y1] = ys;
    if (o.zero) { y0 = Math.min(y0, 0); y1 = Math.max(y1, 0); }
    if (y0 === y1) { y0 -= 1; y1 += 1; }
    const py = o.tight ? 0 : (y1 - y0) * 0.06; y0 -= py; y1 += py;
    if (x0 === x1) { x0 -= 1; x1 += 1; }
    const X = (x) => m.l + (x - x0) / (x1 - x0) * (W - m.l - m.r);
    const Y = o.invertY ? (y) => m.t + (y - y0) / (y1 - y0) * (H - m.t - m.b)
                        : (y) => m.t + (y1 - y) / (y1 - y0) * (H - m.t - m.b);
    el('line', { class: 'ax', x1: m.l, y1: m.t, x2: m.l, y2: H - m.b }, svg);
    el('line', { class: 'ax', x1: m.l, y1: H - m.b, x2: W - m.r, y2: H - m.b }, svg);
    const yt = o.yTicks || [y0 + py, (y0 + y1) / 2, y1 - py];
    yt.forEach((v) => txt(svg, m.l - 6, Y(v) + 3, (o.yFmt || nice)(v) + (o.yUnit || ''), { 'text-anchor': 'end' }));
    const xt = o.xTicks || [x0, (x0 + x1) / 2, x1];
    xt.forEach((v, k) => txt(svg, X(v), H - 6, (o.xFmt || nice)(v),
      { 'text-anchor': xt.length > 1 && k === xt.length - 1 && X(v) > W - 30 ? 'end' : 'middle' }));
    if (o.xLabel) txt(svg, W - m.r, H - m.b - 4, o.xLabel, { 'text-anchor': 'end' });
    if (o.zero) el('line', { class: 'zero', x1: m.l, y1: Y(0), x2: W - m.r, y2: Y(0) }, svg);
    return { X, Y, W, H, m };
  }

  // series: [{ pts: [[x, y]], colour, width, step, label }]
  function lineChart(svg, series, o) {
    o = o || {};
    const all = series.flatMap((s) => s.pts).filter(([, y]) => isFinite(y));
    if (!all.length) return empty(svg);
    const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
    const f = frame(svg, o.xRange || [Math.min(...xs), Math.max(...xs)], o.yRange || [Math.min(...ys), Math.max(...ys)], o);
    series.forEach((s) => {
      const pts = s.pts.filter(([, y]) => isFinite(y));
      const every = Math.max(1, Math.floor(pts.length / (f.W * 1.5)));
      let d = '';
      pts.filter((_, k) => k % every === 0 || k === pts.length - 1).forEach(([x, y], k, arr) => {
        if (!k) d += `M${f.X(x).toFixed(1)} ${f.Y(y).toFixed(1)}`;
        else if (s.step) d += `H${f.X(x).toFixed(1)}V${f.Y(y).toFixed(1)}`;
        else d += `L${f.X(x).toFixed(1)} ${f.Y(y).toFixed(1)}`;
      });
      el('path', { d, stroke: s.colour, 'stroke-width': s.width || 1.4 }, svg);
      if (s.dots) pts.forEach(([x, y]) => el('circle', { class: 'dot', cx: f.X(x), cy: f.Y(y), r: 2.6, fill: s.colour }, svg));
    });
    legend(svg, series.filter((s) => s.label), f);
    if (o.pick) linkCursor(svg, f, o.xRange);
    return f;
  }

  // Distance charts: show the map's inspected point and set it on click
  function linkCursor(svg, f, xr) {
    if (S.cursor !== null && S.cursor >= xr[0] && S.cursor <= xr[1]) {
      el('line', { class: 'cur', x1: f.X(S.cursor), y1: f.m.t, x2: f.X(S.cursor), y2: f.H - f.m.b }, svg);
    }
    svg.style.cursor = 'crosshair';
    svg.onclick = (e) => {
      const px = e.clientX - svg.getBoundingClientRect().left;
      AccDash.setCursor(xr[0] + (px - f.m.l) / (f.W - f.m.l - f.m.r) * (xr[1] - xr[0]));
    };
  }

  const kmTicks = () => range(0, Math.floor(S.info.lapMeters / 1000)).map((k) => k * 1000);
  const kmFmt = (v) => (v / 1000).toFixed(0) + ' km';
  const bestLap = () => S.info.list.find((L) => L.lap === S.info.best);
  const flyingLaps = () => S.info.list.filter((L) => L.flying && S.traces[L.lap]).map((L) => L.lap);
  // Best lap drawn last so it sits on top
  const lapOrder = (laps) => laps.slice().sort((a, b) => (a === S.info.best) - (b === S.info.best));
  const lapSeries = (laps, pts) => lapOrder(laps).map((lap) => ({
    label: 'L' + lap + (lap === S.info.best ? ' best' : ''), colour: S.colours[lap],
    width: lap === S.info.best ? 2 : 1.1, pts: pts(lap) }));

  // items: [{ label, value, cls, text }]
  function barChart(svg, items, o) {
    o = o || {};
    if (!items.length) return empty(svg);
    const vals = items.map((i) => i.value);
    const lo = o.base !== undefined ? o.base : Math.min(0, ...vals), hi = Math.max(...vals);
    const f = frame(svg, [0, items.length], [lo, hi], Object.assign({ xTicks: [], tight: o.base !== undefined }, o));
    const bw = (f.X(1) - f.X(0)) * 0.7;
    items.forEach((it, k) => {
      const x = f.X(k + 0.5) - bw / 2, y = f.Y(Math.max(it.value, lo));
      el('rect', { class: 'bar ' + (it.cls || ''), x, y, width: bw, height: Math.max(0, f.Y(lo) - y) }, svg);
      txt(svg, x + bw / 2, f.H - 6, it.label, { 'text-anchor': 'middle' });
      if (it.text) txt(svg, x + bw / 2, y - 3, it.text, { 'text-anchor': 'middle' });
    });
  }

  function legend(svg, series, f) {
    let x = f.m.l + 8;
    series.forEach((s) => {
      el('rect', { x, y: f.m.t + 2, width: 8, height: 8, fill: s.colour, stroke: 'none' }, svg);
      const t = txt(svg, x + 12, f.m.t + 9, s.label);
      x += 18 + t.getComputedTextLength();
    });
  }

  function empty(svg, msg) {
    svg.innerHTML = '';
    txt(svg, 12, 20, msg || 'No data for this chart in this CSV.');
  }

  /* ---------- map tab: full-lap charts below the map ---------- */

  function renderMapMore(why) {
    if (!S.data) return;
    const L = S.info.lapMeters, best = bestLap();
    if (why !== 'cursor') {
      const tag = `L${best.lap} · ${fmtLap(best.timeS)}`;
      $('m-best-a').textContent = $('m-best-b').textContent = tag;
      trackMaps();
    }
    const laps = flyingLaps(), xo = { xRange: [0, L], xTicks: kmTicks(), xFmt: kmFmt, pick: true };
    lineChart($('m-speed'), lapSeries(laps, (lap) => S.traces[lap].map((p) => [p.d, p.speed])), xo);
    const ref = S.timeGrid[S.info.best], G = AccDash.GRID_M;
    lineChart($('m-delta'), lapSeries(laps, (lap) => Array.from(S.timeGrid[lap], (t, k) => [k * G, t - ref[k]])),
      Object.assign({ zero: true, yUnit: ' s' }, xo));
  }

  // Best lap drawn in world coordinates, coloured per segment: speed, and throttle (+) / brake (−)
  function trackMaps() {
    const pts = S.traces[S.info.best];
    const vs = pts.map((p) => p.speed).sort((a, b) => a - b);
    const lo = vs[Math.floor(vs.length * 0.02)], hi = vs[Math.floor(vs.length * 0.98)];
    const sc = (v) => AccDash.speedColour(v, lo, hi);
    drawTrackMap($('m-tm-speed'), pts, (p) => sc(Math.round(p.speed / 5) * 5));
    $('m-sp-lo').textContent = Math.round(lo) + ' km/h'; $('m-sp-hi').textContent = Math.round(hi);
    $('m-sp-bar').style.background = `linear-gradient(90deg, ${[0, .25, .5, .75, 1].map((t) => sc(lo + t * (hi - lo))).join(',')})`;
    drawTrackMap($('m-tm-pedal'), pts, (p) => pedalScale(Math.round((p.gas - p.brake) * 10) / 10));
    $('m-pd-bar').style.background = `linear-gradient(90deg, ${[-1, -.5, 0, .5, 1].map(pedalScale).join(',')})`;
  }

  // −1 full brake (red) … 0 coasting (pale) … +1 full throttle (green)
  function pedalScale(v) {
    const red = [192, 57, 43], pale = [232, 226, 196], green = [42, 128, 72];
    const [a, b, t] = v < 0 ? [pale, red, -v] : [pale, green, v];
    return `rgb(${a.map((x, i) => Math.round(x + (b[i] - x) * Math.min(1, t))).join(',')})`;
  }

  function drawTrackMap(svg, pts, colourOf) {
    svg.innerHTML = '';
    const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const pad = Math.max(x1 - x0, y1 - y0) * 0.08;
    svg.setAttribute('viewBox', `${x0 - pad} ${y0 - pad} ${x1 - x0 + 2 * pad} ${y1 - y0 + 2 * pad}`);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    const buckets = {};
    for (let k = 1; k < pts.length; k++) {
      const c = colourOf(pts[k]);
      (buckets[c] = buckets[c] || []).push(`M${pts[k - 1].x.toFixed(1)} ${pts[k - 1].y.toFixed(1)}L${pts[k].x.toFixed(1)} ${pts[k].y.toFixed(1)}`);
    }
    for (const c in buckets) el('path', { d: buckets[c].join(''), stroke: c }, svg);
    // km labels pushed outward from the centre, as on the main map
    const cx = xs.reduce((s, v) => s + v, 0) / xs.length, cy = ys.reduce((s, v) => s + v, 0) / ys.length;
    const fs = (x1 - x0 + 2 * pad) / 38;
    for (let km = 0; km * 1000 < S.info.lapMeters - 200; km++) {
      const p = pts.reduce((b, q) => (Math.abs(q.d - km * 1000) < Math.abs(b.d - km * 1000) ? q : b), pts[0]);
      let dx = p.x - cx, dy = p.y - cy; const l = Math.hypot(dx, dy) || 1; dx /= l; dy /= l;
      if (!km) el('circle', { cx: p.x, cy: p.y, r: fs * 0.35, fill: 'var(--ink)' }, svg);
      txt(svg, p.x + dx * fs * 1.2, p.y + dy * fs * 1.2 + fs * 0.35, km ? km + ' km' : 'S/F',
        { class: 'km', 'font-size': fs, 'text-anchor': dx >= 0 ? 'start' : 'end' });
    }
  }

  /* ---------- driving: best lap inputs and lap consistency ---------- */

  function renderInputs() {
    const best = bestLap(), pts = S.traces[best.lap], L = S.info.lapMeters, d = S.data;
    $('d-in-lap').textContent = `L${best.lap} · ${fmtLap(best.timeS)}`;
    const xo = { xRange: [0, L], xTicks: kmTicks(), xFmt: kmFmt, pick: true };
    const ink = 'var(--ink)';
    lineChart($('d-in-speed'), [{ pts: pts.map((p) => [p.d, p.speed]), colour: '#2f6db5' }], Object.assign({ yUnit: '' }, xo));
    lineChart($('d-in-steer'), [{ pts: pts.map((p) => [p.d, p.steer]), colour: '#c77a1e', width: 1.1 }],
      Object.assign({ zero: true }, xo));
    lineChart($('d-in-gear'), [{ pts: pts.map((p) => [p.d, p.gear]), colour: ink, step: true, width: 1.2 }],
      Object.assign({ yTicks: [1, Math.round(Math.max(...pts.map((p) => p.gear)))], yFmt: (v) => String(Math.round(v)) }, xo));

    // Throttle up, brake down, with TC / ABS intervention shaded behind
    const svg = $('d-in-pedal');
    const f = frame(svg, [0, L], [-1, 1], Object.assign({ tight: true, zero: true, yTicks: [-1, 0, 1], yFmt: (v) => v > 0 ? 'thr' : v < 0 ? 'brk' : '0' }, xo));
    const shade = (col, fill) => {
      if (!d.has(col)) return;
      const c = d.col(col);
      let start = null;
      pts.forEach((p, k) => {
        const on = c[p.i] > 0;
        if (on && start === null) start = p.d;
        if ((!on || k === pts.length - 1) && start !== null) {
          el('rect', { x: f.X(start), y: f.m.t, width: Math.max(1, f.X(p.d) - f.X(start)), height: f.H - f.m.t - f.m.b, fill }, svg);
          start = null;
        }
      });
    };
    shade('tc_active', 'rgba(142,91,181,.25)');
    shade('abs_active', 'rgba(199,154,30,.3)');
    const area = (sign, key) => 'M' + f.X(pts[0].d).toFixed(1) + ' ' + f.Y(0).toFixed(1) +
      pts.map((p) => `L${f.X(p.d).toFixed(1)} ${f.Y(sign * p[key]).toFixed(1)}`).join('') +
      `L${f.X(pts[pts.length - 1].d).toFixed(1)} ${f.Y(0).toFixed(1)}Z`;
    el('path', { d: area(1, 'gas'), style: 'fill:#7fc480;stroke:none' }, svg);
    el('path', { d: area(-1, 'brake'), style: 'fill:#e07a6e;stroke:none' }, svg);
    linkCursor(svg, f, [0, L]);
  }

  function renderConsistency() {
    const rows = S.lapStats.rows;
    const xo = { xLabel: 'lap', xTicks: rows.map((r) => r.lap), xFmt: (v) => String(v) };
    [['d-cs-avg', 'avg', '#2f6db5'], ['d-cs-ft', 'fullThrottle', '#3a9a5b'], ['d-cs-br', 'braking', '#c0392b'],
     ['d-cs-co', 'coasting', '#6b6b6b'], ['d-cs-ov', 'overlap', '#8e5bb5'], ['d-cs-fuel', 'fuelUsed', '#8a5a3b']]
      .forEach(([id, key, colour]) => lineChart($(id), [{ pts: rows.map((r) => [r.lap, r[key]]), colour, dots: true, width: 1.6 }], xo));
  }

  /* ---------- per-lap statistics ---------- */

  function lapStats() {
    const d = S.data, c = (n) => (d.has(n) ? d.col(n) : null);
    const sp = c('speed_kmh'), gas = c('gas'), brk = c('brake'), fuel = c('fuel_kg');
    const rows = S.info.list.filter((L) => L.complete).map((L, k, arr) => {
      let n = 0, sum = 0, top = 0, ft = 0, br = 0, co = 0, ov = 0, tc = 0, ab = 0;
      const means = {};
      for (let i = L.i0; i <= L.i1; i++) {
        if (sp[i] <= 5) continue;
        n++; sum += sp[i]; top = Math.max(top, sp[i]);
        if (gas[i] > 0.95) ft++;
        if (brk[i] > 0.05) br++;
        if (gas[i] < 0.05 && brk[i] < 0.05) co++;
        if (gas[i] > 0.1 && brk[i] > 0.1) ov++;
        if (d.has('tc_active') && d.col('tc_active')[i] > 0) tc++;
        if (d.has('abs_active') && d.col('abs_active')[i] > 0) ab++;
      }
      ['tyre_temp', 'tyre_press', 'brake_temp'].forEach((m) => WHEELS.forEach((w) => {
        const col = c(`${m}_${w}`);
        if (!col) return;
        let s = 0, q = 0;
        for (let i = L.i0; i <= L.i1; i++) if (sp[i] > 5) { s += col[i]; q++; }
        means[`${m}_${w}`] = q ? s / q : NaN;
      }));
      const next = S.info.list[S.info.list.indexOf(L) + 1];
      const pct = (v) => (n ? v / n * 100 : NaN);
      return Object.assign({
        lap: L.lap, timeS: L.timeS, valid: L.valid, flying: L.flying, position: L.position,
        avg: n ? sum / n : NaN, top, fullThrottle: pct(ft), braking: pct(br), coasting: pct(co), overlap: pct(ov),
        tc: pct(tc), abs: pct(ab), fuelUsed: next && fuel ? fuel[L.i0] - fuel[next.i0] : NaN,
      }, means);
    });
    return { src: S.data, rows };
  }

  /* ---------- race ---------- */

  function raceStart() {
    const d = S.data, ts = d.col('timestamp');
    if (d.has('green_flag')) {
      const g = d.col('green_flag'), fl = d.col('flag');
      for (let i = 0; i < d.n; i++) if (g[i] === 1 || fl[i] === 7) return ts[i];
    }
    const sp = d.col('speed_kmh'), gas = d.col('gas');
    for (let i = 0; i < d.n; i++) if (gas[i] > 0.95 && sp[i] > 70) return ts[i];
    return ts[0];
  }

  function timeSeries(name, filter, every) {
    const d = S.data, ts = d.col('timestamp'), v = d.col(name), t0 = S.raceStart, out = [];
    for (let i = 0; i < d.n; i += every || 25) if (ts[i] >= t0 && (!filter || filter(v[i]))) out.push([(ts[i] - t0) / 60, v[i]]);
    return out;
  }

  function renderRace() {
    S.raceStart = raceStart();
    const rows = S.lapStats.rows.filter((r) => isFinite(r.timeS));
    const best = Math.min(...rows.filter((r) => r.flying && r.valid).map((r) => r.timeS));
    barChart($('r-laps'), rows.map((r) => ({
      label: 'L' + r.lap, value: r.timeS, text: fmtLap(r.timeS) + (r.valid ? '' : ' ✗'),
      cls: r.timeS === best ? 'best' : r.valid ? '' : 'bad' })),
      { base: Math.min(...rows.map((r) => r.timeS)) * 0.985, yFmt: (v) => fmtLap(v) });

    const ink = 'var(--ink)';
    if (S.data.has('position')) {
      const pos = timeSeries('position', (v) => v > 0, 10);
      const ps = pos.map((p) => p[1]);
      const lo = Math.min(...ps), hi = Math.max(...ps);
      lineChart($('r-pos'), [{ pts: pos, colour: ink, step: true, width: 1.8 }],
        { invertY: true, yRange: [lo, hi], yTicks: range(lo, hi), yFmt: (v) => 'P' + v, xLabel: 'min since green' });
      // ACC reports gapBehind as a negative number of ms
      const gap = (name) => timeSeries(name, (v) => v !== 0 && Math.abs(v) < 60000).map(([x, y]) => [x, Math.abs(y) / 1000]);
      lineChart($('r-gap'), [
        { label: 'Ahead', pts: gap('gap_ahead_ms'), colour: 'var(--accent)' },
        { label: 'Behind', pts: S.data.has('gap_behind_ms') ? gap('gap_behind_ms') : [], colour: '#3a9a5b' },
      ], { yUnit: ' s', xLabel: 'min since green' });
    } else { empty($('r-pos')); empty($('r-gap')); }
    lineChart($('r-fuel'), [{ pts: timeSeries('fuel_kg', (v) => v > 0), colour: ink }], { xLabel: 'min since green' });

    const fmtP = (v) => (isFinite(v) ? v.toFixed(1) : '—');
    $('r-table').innerHTML = '<thead><tr><th>Lap</th><th>Time</th><th>Pos</th><th>Avg km/h</th><th>Top</th><th>Full thr %</th><th>Braking %</th><th>Coast %</th><th>Overlap %</th><th>Fuel kg</th></tr></thead><tbody>' +
      S.lapStats.rows.map((r) => `<tr><td>${r.lap}${r.flying ? '' : ' <span class="muted">' + (r.lap === S.info.list[0].lap ? 'start' : 'out') + '</span>'}</td>` +
        `<td class="${r.valid ? '' : 'bad'}">${fmtLap(r.timeS)}${r.valid ? '' : ' ✗'}</td><td>${r.position > 0 ? 'P' + r.position : ''}</td>` +
        `<td>${fmtP(r.avg)}</td><td>${fmtP(r.top)}</td><td>${fmtP(r.fullThrottle)}</td><td>${fmtP(r.braking)}</td>` +
        `<td>${fmtP(r.coasting)}</td><td>${fmtP(r.overlap)}</td><td>${isFinite(r.fuelUsed) ? r.fuelUsed.toFixed(2) : '—'}</td></tr>`).join('') + '</tbody>';
  }

  const range = (a, b) => Array.from({ length: b - a + 1 }, (_, k) => a + k);

  /* ---------- driving ---------- */

  function selectedFrames(every) {
    const out = [];
    S.info.list.filter((L) => S.selected.has(L.lap)).forEach((L) => {
      for (let i = L.i0; i <= L.i1; i += every || 1) if (S.data.col('speed_kmh')[i] > 5) out.push({ i, lap: L.lap });
    });
    return out;
  }

  function hist(svg, frames, col, edges, labelFmt) {
    const v = S.data.col(col), counts = new Array(edges.length - 1).fill(0);
    frames.forEach(({ i }) => {
      for (let b = 0; b < counts.length; b++) {
        if (v[i] >= edges[b] && (v[i] < edges[b + 1] || (b === counts.length - 1 && v[i] <= edges[b + 1]))) { counts[b]++; break; }
      }
    });
    const n = frames.length || 1;
    barChart(svg, counts.map((c, b) => ({ label: labelFmt(edges[b], edges[b + 1]), value: c / n * 100 })), { yUnit: '%' });
  }

  function renderDriving() {
    renderInputs();
    renderConsistency();
    const frames = selectedFrames(1);
    if (!frames.length) {
      ['d-gg', 'd-gas', 'd-brk', 'd-steer', 'd-gear'].forEach((id) => empty($(id), 'Select laps in the Laps table.'));
      $('d-brake').innerHTML = '';
      return;
    }
    // G-G
    const svg = $('d-gg'), lat = S.data.col('accel_lateral'), lon = S.data.col('accel_longitudinal');
    const every = Math.max(1, Math.floor(frames.length / 6000));
    const pts = frames.filter((_, k) => k % every === 0).filter(({ i }) => Math.abs(lat[i]) < 4 && Math.abs(lon[i]) < 4);
    const lim = Math.max(...pts.map(({ i }) => Math.max(Math.abs(lat[i]), Math.abs(lon[i])))) * 1.05 || 1;
    const f = frame(svg, [-lim, lim], [-lim, lim], { tight: true, yUnit: ' G', xLabel: 'lateral G', xTicks: [-lim, 0, lim] });
    el('line', { class: 'zero', x1: f.X(0), y1: f.m.t, x2: f.X(0), y2: f.H - f.m.b }, svg);
    el('line', { class: 'zero', x1: f.m.l, y1: f.Y(0), x2: f.W - f.m.r, y2: f.Y(0) }, svg);
    pts.forEach(({ i, lap }) => el('circle', { class: 'dot', cx: f.X(lat[i]), cy: f.Y(lon[i]), r: 1.1, fill: S.colours[lap], 'fill-opacity': 0.35 }, svg));

    const pct = (a, b) => `${Math.round(a * 100)}`;
    hist($('d-gas'), frames, 'gas', [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1], pct);
    hist($('d-brk'), frames.filter(({ i }) => S.data.col('brake')[i] > 0.02), 'brake', [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1], pct);
    hist($('d-steer'), frames, 'steer', [-1, -.5, -.3, -.2, -.1, -.03, .03, .1, .2, .3, .5, 1], (a) => a.toFixed(2).replace('0.', '.'));
    const gears = frames.map(({ i }) => Math.round(S.data.col('gear')[i]));
    const gmax = Math.max(...gears);
    hist($('d-gear'), frames, 'gear', range(1, gmax + 1).map((g) => g - 0.5), (a) => String(a + 0.5));

    renderBraking();
  }

  // Brake onsets (brake rising past 0.2 after being below 0.05), as distance into the lap
  function brakeZones(pts) {
    const out = []; let armed = true, last = -1e9;
    pts.forEach((p) => {
      if (armed && p.brake > 0.2 && p.d - last > 150) { out.push(p); last = p.d; armed = false; }
      else if (p.brake < 0.05) armed = true;
    });
    return out;
  }

  function renderBraking() {
    const best = S.info.best, ref = brakeZones(S.traces[best]);
    const laps = [...S.selected].filter((l) => S.traces[l]).sort((a, b) => a - b);
    const head = '<thead><tr><th>Zone</th><th>At</th>' + laps.map((l) => `<th><span class="sw" style="background:${S.colours[l]}"></span>L${l}</th>`).join('') + '<th>Spread</th><th>Min km/h</th></tr></thead>';
    const body = ref.map((z, k) => {
      const cells = [], mins = [];
      laps.forEach((l) => {
        const on = brakeZones(S.traces[l]).reduce((b, p) => (Math.abs(p.d - z.d) < Math.abs((b || { d: 1e9 }).d - z.d) ? p : b), null);
        if (!on || Math.abs(on.d - z.d) > 150) { cells.push('<td>—</td>'); return; }
        const dd = on.d - z.d;
        cells.push(`<td>${l === best ? '0' : (dd >= 0 ? '+' : '') + dd.toFixed(0)} m</td>`);
        const seg = S.traces[l].filter((p) => p.d >= on.d && p.d <= on.d + 400);
        mins.push({ d: on.d, v: Math.min(...seg.map((p) => p.speed)) });
      });
      const ds = mins.map((m) => m.d), vs = mins.map((m) => m.v);
      return `<tr><td>Z${k + 1}</td><td>${(z.d / 1000).toFixed(2)} km</td>${cells.join('')}` +
        `<td>${ds.length > 1 ? (Math.max(...ds) - Math.min(...ds)).toFixed(0) + ' m' : '—'}</td>` +
        `<td>${vs.length ? Math.min(...vs).toFixed(0) + '–' + Math.max(...vs).toFixed(0) : '—'}</td></tr>`;
    }).join('');
    $('d-brake').innerHTML = head + '<tbody>' + body + '</tbody>';
  }

  /* ---------- car ---------- */

  function renderCar() {
    const rows = S.lapStats.rows;
    const perWheel = (svg, m, unit) => {
      if (!S.data.has(`${m}_fl`)) return empty(svg);
      lineChart(svg, WHEELS.map((w) => ({ label: w.toUpperCase(), colour: WHEEL_COLOURS[w], dots: true,
        pts: rows.map((r) => [r.lap, r[`${m}_${w}`]]) })), { yUnit: unit, xLabel: 'lap', xTicks: rows.map((r) => r.lap), xFmt: (v) => String(v) });
    };
    perWheel($('k-ttemp'), 'tyre_temp', '°');
    perWheel($('k-press'), 'tyre_press', '');
    perWheel($('k-btemp'), 'brake_temp', '°');
    if (S.data.has('tc_active')) {
      lineChart($('k-aids'), [
        { label: 'TC', colour: '#8e5bb5', dots: true, pts: rows.map((r) => [r.lap, r.tc]) },
        { label: 'ABS', colour: '#c79a1e', dots: true, pts: rows.map((r) => [r.lap, r.abs]) },
      ], { yUnit: '%', zero: true, xLabel: 'lap', xTicks: rows.map((r) => r.lap), xFmt: (v) => String(v) });
    } else empty($('k-aids'));

    const sp = S.data.col('speed_kmh'), med = (col, scale, filt) => {
      const v = [];
      for (let i = 0; i < S.data.n; i += 3) if (sp[i] > 5 && (!filt || filt(i))) v.push(col[i] * scale);
      v.sort((a, b) => a - b);
      return v[v.length >> 1];
    };
    const valid = S.data.has('wheel_slip_valid') ? S.data.col('wheel_slip_valid') : null;
    barChart($('k-slip'), WHEELS.map((w) => ({ label: w.toUpperCase(), value: med(S.data.col('wheel_slip_' + w), 1, valid && ((i) => valid[i] === 1)) }))
      .map((it) => Object.assign(it, { text: nice(it.value) })));
    barChart($('k-susp'), WHEELS.map((w) => ({ label: w.toUpperCase(), value: med(S.data.col('suspension_travel_' + w), 1000) }))
      .map((it) => Object.assign(it, { text: nice(it.value) })));
  }

  /* ---------- data ---------- */

  function renderData() {
    const d = S.data, ts = d.col('timestamp'), sp = d.col('speed_kmh');
    const dur = ts[d.n - 1] - ts[0];
    let latMax = 0, lonMax = 0, still = 0, noPos = 0;
    const lat = d.col('accel_lateral'), lon = d.col('accel_longitudinal'), px = d.col('pos_x'), pz = d.col('pos_z');
    for (let i = 0; i < d.n; i++) {
      // |G| < 4 skips single-frame spikes (kerbs, contact)
      if (sp[i] > 20 && Math.abs(lat[i]) < 4 && Math.abs(lon[i]) < 4) { latMax = Math.max(latMax, Math.abs(lat[i])); lonMax = Math.max(lonMax, Math.abs(lon[i])); }
      if (sp[i] <= 5) still++;
      if (px[i] === 0 && pz[i] === 0) noPos++;
    }
    const fuel = d.col('fuel_kg').filter((v) => v > 0);
    const bestRow = S.info.list.find((L) => L.lap === S.info.best);
    const kv = (o) => Object.entries(o).map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
    $('x-summary').innerHTML = kv({
      File: S.fileName, Session: d.has('session_type') ? d.col('session_type')[0] : '—',
      Duration: (dur / 60).toFixed(1) + ' min', Samples: d.n.toLocaleString(), 'Sample rate': (d.n / dur).toFixed(0) + ' Hz',
      'Best lap': fmtLap(bestRow.timeS) + ' (L' + bestRow.lap + ')', 'Top speed': Math.max(...sp).toFixed(1) + ' km/h',
      'Max G': `${lonMax.toFixed(2)} long · ${latMax.toFixed(2)} lat`,
      'Fuel used': fuel.length ? (fuel[0] - fuel[fuel.length - 1]).toFixed(2) + ' kg' : '—',
      Finish: S.info.list[S.info.list.length - 1].position > 0 ? 'P' + S.info.list[S.info.list.length - 1].position : '—',
    });
    const share = (k) => (k / d.n * 100).toFixed(1) + '%';
    $('x-quality').innerHTML = kv({
      Channels: d.names.length, 'Stationary frames': share(still), 'Frames without position': share(noPos),
      'Wheel slip flagged invalid': d.has('wheel_slip_valid') ? share(d.col('wheel_slip_valid').filter((v) => v === 0).length) : '—',
      'Invalid laps': S.info.list.filter((L) => L.complete && !L.valid).length,
      'Flying laps used for reference': S.info.list.filter((L) => L.flying).length,
    });
    const rows = d.names.map((name) => {
      const v = d.col(name);
      if (typeof v[0] === 'string') {
        const u = new Set(v.slice(0, 5000));
        return `<tr><td>${name}</td><td colspan="4" class="muted">text · ${u.size}${u.size >= 5000 ? '+' : ''} values · e.g. ${[...u].slice(0, 3).join(', ')}</td></tr>`;
      }
      let lo = Infinity, hi = -Infinity, s = 0, z = 0, bad = 0;
      for (let i = 0; i < v.length; i++) {
        const x = v[i];
        if (!isFinite(x)) { bad++; continue; }
        lo = Math.min(lo, x); hi = Math.max(hi, x); s += x; if (x === 0) z++;
      }
      const n = v.length - bad;
      return `<tr><td>${name}</td><td>${nice(lo)}</td><td>${nice(s / (n || 1))}</td><td>${nice(hi)}</td><td>${(z / (n || 1) * 100).toFixed(0)}%${bad ? ' · ' + bad + ' NaN' : ''}</td></tr>`;
    });
    $('x-cols').innerHTML = '<thead><tr><th>Channel</th><th>Min</th><th>Mean</th><th>Max</th><th>Zeros</th></tr></thead><tbody>' + rows.join('') + '</tbody>';
  }
})();
