/* dashboard.js — ACC session dashboard: zoomable track map, racing lines per lap, linked charts. */
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
  const fmtLap = (s) => isNaN(s) ? '—' : `${Math.floor(s / 60)}:${(s % 60).toFixed(3).padStart(6, '0')}`;
  const PALETTE = ['#2f6db5', '#3a9a5b', '#8e5bb5', '#c79a1e', '#2a9d9d', '#6b6b6b', '#c4467c', '#5a7d2a'];
  const GRID_M = 5;          // distance grid for deltas
  const S = {};              // session state

  /* ---------- loading ---------- */

  function load(text, fileName) {
    const data = AccCsv.parse(text, fileName);
    const info = AccCsv.laps(data);
    if (info.best === null) throw new Error('No complete flying lap in this session.');
    const toScreen = AccCsv.orientation(data, info);
    const flying = info.list.filter((L) => L.flying);
    const traces = {};
    info.list.forEach((L) => {
      if (L.complete) traces[L.lap] = AccCsv.trace(data, L, info.lapMeters, toScreen, 2);
    });
    const ref = AccCsv.reference(flying.map((L) => traces[L.lap]), info.lapMeters, 2);
    const offsets = {}, timeGrid = {};
    const nGrid = Math.floor(info.lapMeters / GRID_M);
    for (const lap in traces) {
      offsets[lap] = AccCsv.offsets(traces[lap], ref);
      const o = offsets[lap], m = o.length;
      traces[lap].forEach((p, k) => {
        p.off = o[k];
        // ±5 samples (~±5 m at speed) moving average: used only when the spread exaggerates offsets
        let s = 0, c = 0;
        for (let j = Math.max(0, k - 5); j <= Math.min(m - 1, k + 5); j++) { s += o[j]; c++; }
        p.offS = s / c;
      });
      const g = new Float64Array(nGrid);
      for (let k = 0; k < nGrid; k++) g[k] = AccCsv.sampleAt(traces[lap], k * GRID_M, 't');
      timeGrid[lap] = g;
    }
    const m = /acc_session_(\d{8})_\d+_?(.*)\.csv$/i.exec(fileName || '');
    const track = m && m[2] ? m[2].replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()) : 'Unknown track';

    Object.assign(S, { data, info, traces, ref, timeGrid, track, fileName, cursor: null,
                       colour: $('colour').value, spread: +$('spread').value, selected: new Set() });
    const best = info.best;
    S.selected.add(best);
    // Also show the two laps closest to the best for comparison
    flying.filter((L) => L.lap !== best).sort((a, b) => a.timeS - b.timeS).slice(0, 2)
      .forEach((L) => S.selected.add(L.lap));
    S.colours = {};
    let c = 0;
    info.list.forEach((L) => { S.colours[L.lap] = L.lap === best ? 'var(--accent)' : PALETTE[c++ % PALETTE.length]; });

    renderSession();
    buildMap();
    resetView();
    renderAll();
    $('empty').classList.add('hidden');
    $('export').disabled = false;
  }

  function openFile(file) {
    const r = new FileReader();
    $('file-name').textContent = 'Loading ' + file.name + '…';
    r.onload = () => {
      try { load(r.result, file.name); $('file-name').textContent = file.name; }
      catch (e) { $('file-name').textContent = '⚠ ' + e.message; }
    };
    r.readAsText(file);
  }

  // When served over http: index.html?csv=../telemetry_logs/<file>.csv opens that file directly
  const qs = new URLSearchParams(location.search).get('csv');
  if (qs && location.protocol.startsWith('http')) {
    $('file-name').textContent = 'Loading ' + qs + '…';
    fetch(qs).then((r) => { if (!r.ok) throw new Error(r.status + ' ' + r.statusText); return r.text(); })
      .then((t) => { load(t, qs.split('/').pop()); $('file-name').textContent = qs.split('/').pop(); })
      .catch((e) => { $('file-name').textContent = '⚠ ' + e.message; });
  }

  $('file').addEventListener('change', (e) => e.target.files[0] && openFile(e.target.files[0]));
  document.addEventListener('dragover', (e) => { e.preventDefault(); document.body.classList.add('drop'); });
  document.addEventListener('dragleave', () => document.body.classList.remove('drop'));
  document.addEventListener('drop', (e) => {
    e.preventDefault(); document.body.classList.remove('drop');
    if (e.dataTransfer.files[0]) openFile(e.dataTransfer.files[0]);
  });

  /* ---------- side panel ---------- */

  function renderSession() {
    const { info, data } = S;
    const bestLap = info.list.find((L) => L.lap === info.best);
    const last = info.list[info.list.length - 1];
    $('s-track').textContent = $('track-name').textContent = S.track;
    $('s-type').textContent = data.has('session_type') ? data.col('session_type')[0] : '—';
    $('s-len').textContent = (info.lapMeters / 1000).toFixed(3) + ' km';
    $('s-laps').textContent = info.list.filter((L) => L.complete).length + ' complete';
    $('s-pos').textContent = last.position > 0 ? 'P' + last.position : '—';
    $('s-best').textContent = fmtLap(bestLap.timeS) + ' · L' + bestLap.lap;

    const tb = $('laps');
    tb.innerHTML = '';
    info.list.filter((L) => L.complete).forEach((L) => {
      const tr = document.createElement('tr');
      const d = L.timeS - bestLap.timeS;
      const note = !L.flying ? ' <span class="muted">' + (L.lap === info.list[0].lap ? 'start' : 'out') + '</span>' : '';
      tr.innerHTML = `<td><span class="sw" style="background:${S.colours[L.lap]}"></span>${L.lap}${note}</td>` +
        `<td class="${L.valid ? '' : 'bad'}">${fmtLap(L.timeS)}${L.valid ? '' : ' ✗'}</td>` +
        `<td>${L.lap === info.best ? '—' : isNaN(d) ? '' : (d >= 0 ? '+' : '') + d.toFixed(3)}</td>` +
        `<td>${L.position > 0 ? 'P' + L.position : ''}</td>`;
      tr.dataset.lap = L.lap;
      tr.onclick = () => {
        S.selected.has(L.lap) ? S.selected.delete(L.lap) : S.selected.add(L.lap);
        renderAll();
      };
      tb.appendChild(tr);
    });
  }

  function syncLapRows() {
    document.querySelectorAll('#laps tr').forEach((tr) => {
      tr.classList.toggle('off', !S.selected.has(+tr.dataset.lap));
    });
  }

  $('colour').addEventListener('change', (e) => { S.colour = e.target.value; renderLines(); renderLegend(); });
  $('show-ref').addEventListener('change', (e) => S.refPath && S.refPath.classList.toggle('hidden', !e.target.checked));
  $('show-road').addEventListener('change', (e) => S.roadGroup && S.roadGroup.classList.toggle('hidden', !e.target.checked));

  $('export').addEventListener('click', () => {
    const t = AccCsv.trackForTrainer(S.ref, S.info.lapMeters, S.track);
    const blob = new Blob([JSON.stringify(t, null, 1)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = S.track.toLowerCase().replace(/\s+/g, '_') + '.track.json';
    a.click();
    URL.revokeObjectURL(a.href);
  });

  /* ---------- map ---------- */

  const svg = $('map');
  let world, linesG, cursorG, kmLabels = [];
  let view = { x: 0, y: 0, w: 1, h: 1 };

  const pathOf = (pts, close) => 'M' + pts.map((p) => p.x.toFixed(1) + ' ' + p.y.toFixed(1)).join('L') + (close ? 'Z' : '');

  // Exaggerated offset, kept inside each corner's radius so lines can't fold over on the inside
  function spreadOffset(r, o) {
    const v = o * S.spread;
    if (v * r.curv > 0) {
      const lim = 0.7 / Math.abs(r.curv);
      return Math.sign(v) * Math.min(Math.abs(v), lim);
    }
    return v;
  }

  // Where a lap point is drawn: its real position, or with the sideways offset exaggerated by S.spread
  function shown(p) {
    if (!S.spread || S.spread === 1) return p;
    const r = AccCsv.refAt(S.ref, p.k), o = spreadOffset(r, p.offS);
    return { x: r.x + r.nx * o, y: r.y + r.ny * o };
  }

  function buildMap() {
    svg.innerHTML = '';
    world = el('g', {}, svg);
    const ref = S.ref, n = ref.length;

    // Used track width: envelope of the flying laps' offsets, smoothed, plus a margin
    const lo = new Float64Array(n).fill(0), hi = new Float64Array(n).fill(0);
    const step = ref[1].d;
    for (const lap of S.info.list.filter((L) => L.flying).map((L) => L.lap)) {
      S.traces[lap].forEach((p) => {
        const k = Math.round(p.k) % n;
        if (Math.abs(p.offS) > 25) return;
        lo[k] = Math.min(lo[k], p.offS); hi[k] = Math.max(hi[k], p.offS);
      });
    }
    // Rolling max/min (±12 m) for the envelope, then a rolling mean (±12 m) so the edges are smooth
    const sm = (a, f) => Array.from(a, (_, k) => {
      let v = a[k];
      for (let j = -6; j <= 6; j++) v = f(v, a[(k + j + n) % n]);
      return v;
    });
    const mean = (a) => Array.from(a, (_, k) => {
      let s = 0;
      for (let j = -6; j <= 6; j++) s += a[(k + j + n) % n];
      return s / 13;
    });
    const L = mean(sm(hi, Math.max)).map((v) => v + 1.0), R = mean(sm(lo, Math.min)).map((v) => v - 1.0);
    S.band = { L, R };
    S.roadGroup = el('g', {}, world);
    drawRoad();
    S.roadGroup.classList.toggle('hidden', !$('show-road').checked);

    S.refPath = el('path', { class: 'ref', d: pathOf(ref, true) }, world);
    S.refPath.classList.toggle('hidden', !$('show-ref').checked);

    const r0 = ref[0];
    el('line', { class: 'sf', x1: r0.x + r0.nx * L[0], y1: r0.y + r0.ny * L[0],
                 x2: r0.x + r0.nx * R[0], y2: r0.y + r0.ny * R[0] }, world);

    // km markers, pushed outward from the track centre
    const cx = ref.reduce((s, p) => s + p.x, 0) / n, cy = ref.reduce((s, p) => s + p.y, 0) / n;
    kmLabels = [];
    for (let km = 0; km * 1000 < S.info.lapMeters - 200; km++) {
      const r = ref[Math.min(n - 1, Math.round(km * 1000 / step))];
      let dx = r.x - cx, dy = r.y - cy; const l = Math.hypot(dx, dy) || 1; dx /= l; dy /= l;
      const t = el('text', { class: 'km', 'text-anchor': dx >= 0 ? 'start' : 'end' }, world);
      t.textContent = km === 0 ? 'S/F' : km + ' km';
      kmLabels.push({ t, x: r.x, y: r.y, dx, dy });
    }

    linesG = el('g', {}, world);
    cursorG = el('g', {}, world);
    const xs = ref.map((p) => p.x), ys = ref.map((p) => p.y);
    S.bounds = { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
  }

  // Used-width band around the reference line, widened with the line spread
  function drawRoad() {
    const { L, R } = S.band;
    const edge = (r, o) => (S.spread > 1 ? spreadOffset(r, o) : o);
    const left = S.ref.map((r, k) => { const o = edge(r, L[k]); return { x: r.x + r.nx * o, y: r.y + r.ny * o }; });
    const right = S.ref.map((r, k) => { const o = edge(r, R[k]); return { x: r.x + r.nx * o, y: r.y + r.ny * o }; });
    S.roadGroup.innerHTML = '';
    el('path', { class: 'road', d: pathOf(left, true) + pathOf(right, true) }, S.roadGroup);
    el('path', { class: 'edge', d: pathOf(left, true) }, S.roadGroup);
    el('path', { class: 'edge', d: pathOf(right, true) }, S.roadGroup);
  }

  $('spread').addEventListener('input', (e) => {
    S.spread = +e.target.value;
    $('spread-v').textContent = '×' + S.spread;
    if (!S.ref) return;
    drawRoad(); renderLines(); renderCursor();
  });

  function speedColour(v, lo, hi) {
    const t = Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
    // dark blue (slow) -> soft yellow (fast)
    const a = [38, 52, 110], b = [92, 160, 120], c = [232, 196, 64];
    const m = t < 0.5 ? [a, b, t * 2] : [b, c, (t - 0.5) * 2];
    return `rgb(${m[0].map((x, i) => Math.round(x + (m[1][i] - x) * m[2])).join(',')})`;
  }
  const pedalColour = (p) => p.brake > 0.05 ? '#c0392b' : p.gas > 0.95 ? '#3a9a5b' : p.gas > 0.05 ? '#a8c47a' : '#c9c6bd';

  function renderLines() {
    linesG.innerHTML = '';
    const laps = [...S.selected].sort((a, b) => (a === S.info.best) - (b === S.info.best));   // best on top
    if (S.colour === 'lap') {
      laps.forEach((lap) => el('path', { class: 'line', d: pathOf(S.traces[lap].map(shown)), stroke: S.colours[lap] }, linesG));
      return;
    }
    // Speed / pedal: one path per colour bucket per lap
    const all = laps.flatMap((lap) => S.traces[lap]);
    const vs = all.map((p) => p.speed).sort((a, b) => a - b);
    const lo = vs[Math.floor(vs.length * 0.02)] || 0, hi = vs[Math.floor(vs.length * 0.98)] || 1;
    S.speedRange = [lo, hi];
    laps.forEach((lap) => {
      const raw = S.traces[lap], pts = raw.map(shown), buckets = {};
      for (let k = 1; k < pts.length; k++) {
        const col = S.colour === 'speed'
          ? speedColour(Math.round(raw[k].speed / 10) * 10, lo, hi) : pedalColour(raw[k]);
        (buckets[col] = buckets[col] || []).push(`M${pts[k - 1].x.toFixed(1)} ${pts[k - 1].y.toFixed(1)}L${pts[k].x.toFixed(1)} ${pts[k].y.toFixed(1)}`);
      }
      for (const col in buckets) el('path', { class: 'line', d: buckets[col].join(''), stroke: col }, linesG);
    });
  }

  function renderLegend() {
    const lg = $('legend');
    lg.classList.toggle('hidden', S.colour === 'lap');
    if (S.colour === 'speed' && S.speedRange) {
      const [lo, hi] = S.speedRange;
      $('leg-lo').textContent = Math.round(lo) + ' km/h'; $('leg-hi').textContent = Math.round(hi);
      $('leg-bar').style.background = `linear-gradient(90deg, ${[0, .25, .5, .75, 1].map((t) => speedColour(lo + t * (hi - lo), lo, hi)).join(',')})`;
    } else if (S.colour === 'pedal') {
      $('leg-lo').textContent = 'Brake'; $('leg-hi').textContent = 'Full throttle';
      $('leg-bar').style.background = 'linear-gradient(90deg,#c0392b 0 25%,#c9c6bd 25% 50%,#a8c47a 50% 75%,#3a9a5b 75%)';
    }
  }

  /* ---------- pan / zoom ---------- */

  const aspect = () => { const r = svg.getBoundingClientRect(); return r.height / r.width || 0.6; };
  function resetView() {
    const b = S.bounds, pad = Math.max(b.x1 - b.x0, b.y1 - b.y0) * 0.08, a = aspect();
    const w = Math.max(b.x1 - b.x0 + 2 * pad, (b.y1 - b.y0 + 2 * pad) / a);
    view = { x: (b.x0 + b.x1) / 2 - w / 2, y: (b.y0 + b.y1) / 2 - w * a / 2, w, h: w * a };
    fullW = w;
    applyView();
  }
  // Racing lines are drawn in screen pixels; thicken them as you zoom so they read without zooming all the way in
  let fullW = 1;
  const LINE_W = 1.6, LINE_W_MAX = 7;
  function applyView() {
    svg.setAttribute('viewBox', `${view.x} ${view.y} ${view.w} ${view.h}`);
    const zoom = Math.max(1, fullW / view.w);
    svg.style.setProperty('--line-w', Math.min(LINE_W_MAX, LINE_W * Math.pow(zoom, 0.55)).toFixed(2) + 'px');
    const fs = view.w / 70;
    kmLabels.forEach((k) => {
      k.t.setAttribute('font-size', fs);
      k.t.setAttribute('x', k.x + k.dx * fs * 1.6); k.t.setAttribute('y', k.y + k.dy * fs * 1.6 + fs * 0.35);
    });
    renderCursor();
    renderCharts();
  }
  const toWorld = (px, py) => {
    const p = svg.createSVGPoint(); p.x = px; p.y = py;
    return p.matrixTransform(svg.getScreenCTM().inverse());
  };
  function zoomAt(wx, wy, f) {
    view = { x: wx - (wx - view.x) * f, y: wy - (wy - view.y) * f, w: view.w * f, h: view.h * f };
    applyView();
  }

  svg.addEventListener('wheel', (e) => {
    if (!S.ref) return;
    e.preventDefault();
    const p = toWorld(e.clientX, e.clientY);
    zoomAt(p.x, p.y, e.deltaY > 0 ? 1.18 : 1 / 1.18);
  }, { passive: false });

  let drag = null;
  svg.addEventListener('pointerdown', (e) => {
    if (!S.ref) return;
    drag = { x: e.clientX, y: e.clientY, v: { ...view }, moved: false };
    svg.setPointerCapture(e.pointerId);
  });
  svg.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const r = svg.getBoundingClientRect(), k = view.w / r.width;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) { drag.moved = true; svg.classList.add('drag'); }
    if (drag.moved) { view.x = drag.v.x - dx * k; view.y = drag.v.y - dy * k; applyView(); }
  });
  // Click = inspect point; double-click = zoom to ~320 m around it.
  // (Handled here because pointer capture stops the browser's dblclick event.)
  let lastClick = 0;
  svg.addEventListener('pointerup', (e) => {
    if (drag && !drag.moved) {
      const r = nearestRef(toWorld(e.clientX, e.clientY));
      setCursor(r.d);
      if (e.timeStamp - lastClick < 350) {
        const w = Math.min(320, view.w), a = aspect();
        view = { x: r.x - w / 2, y: r.y - w * a / 2, w, h: w * a };
        applyView();
      }
      lastClick = e.timeStamp;
    }
    drag = null; svg.classList.remove('drag');
  });
  $('reset').addEventListener('click', () => S.ref && resetView());
  window.addEventListener('resize', () => S.ref && applyView());

  function nearestRef(p) {
    let best = S.ref[0], bd = Infinity;
    for (const r of S.ref) { const d = (r.x - p.x) ** 2 + (r.y - p.y) ** 2; if (d < bd) { bd = d; best = r; } }
    return best;
  }

  /* ---------- cursor (selected point) ---------- */

  function setCursor(d) { S.cursor = d; renderCursor(); renderCharts(); renderPoint(); }

  function renderCursor() {
    if (!cursorG) return;
    cursorG.innerHTML = '';
    if (S.cursor === null) return;
    const step = S.ref[1].d, k = Math.min(S.ref.length - 1, Math.round(S.cursor / step)), r = S.ref[k];
    const edge = (o) => (S.spread > 1 ? spreadOffset(r, o) : o);
    const L = edge(S.band.L[k]) + 3, R = edge(S.band.R[k]) - 3;
    el('line', { class: 'cut', x1: r.x + r.nx * L, y1: r.y + r.ny * L, x2: r.x + r.nx * R, y2: r.y + r.ny * R }, cursorG);
    const rad = view.w / 260;
    S.selected.forEach((lap) => {
      const pts = S.traces[lap];
      const at = S.spread > 1
        ? (() => { const o = spreadOffset(r, AccCsv.sampleAt(pts, S.cursor, 'offS')); return { x: r.x + r.nx * o, y: r.y + r.ny * o }; })()
        : { x: AccCsv.sampleAt(pts, S.cursor, 'x'), y: AccCsv.sampleAt(pts, S.cursor, 'y') };
      el('circle', { class: 'pin', r: rad, fill: S.colours[lap], cx: at.x, cy: at.y }, cursorG);
    });
  }

  function deltaAt(lap, d) {
    const k = Math.min(S.timeGrid[lap].length - 1, Math.max(0, Math.round(d / GRID_M)));
    return S.timeGrid[lap][k] - S.timeGrid[S.info.best][k];
  }

  function renderPoint() {
    if (S.cursor === null) return;
    $('pt-where').textContent = `${(S.cursor / 1000).toFixed(2)} km into the lap`;
    $('b-dist').textContent = (S.cursor / 1000).toFixed(2) + ' km';
    $('pt-table').classList.remove('hidden');
    const rows = [...S.selected].sort((a, b) => a - b).map((lap) => {
      const pts = S.traces[lap], at = (k) => AccCsv.sampleAt(pts, S.cursor, k);
      const dl = deltaAt(lap, S.cursor);
      return `<tr><td><span class="sw" style="background:${S.colours[lap]}"></span>${lap}</td>` +
        `<td>${at('speed').toFixed(0)}</td><td>${Math.round(at('gear'))}</td>` +
        `<td>${(at('off') >= 0 ? '+' : '') + at('off').toFixed(1)}</td>` +
        `<td>${lap === S.info.best ? '—' : (dl >= 0 ? '+' : '') + dl.toFixed(2)}</td></tr>`;
    });
    $('pt-rows').innerHTML = rows.join('');
  }

  /* ---------- charts ---------- */

  // Distance window: the part of the reference line inside the map view (full lap when zoomed out).
  function chartWindow() {
    const L = S.info.lapMeters;
    const inside = S.ref.filter((r) => r.x >= view.x && r.x <= view.x + view.w && r.y >= view.y && r.y <= view.y + view.h);
    if (!inside.length || inside.length > S.ref.length * 0.85) return [0, L];
    let d0 = Math.min(...inside.map((r) => r.d)), d1 = Math.max(...inside.map((r) => r.d));
    if (d1 - d0 > L * 0.6) return [0, L];   // view straddles start/finish; show everything
    return [Math.max(0, d0 - 50), Math.min(L, d1 + 50)];
  }

  function chart(svgEl, title, unit, series, win, opts) {
    svgEl.innerHTML = '';
    const r = svgEl.getBoundingClientRect(), W = r.width, H = r.height;
    if (W < 10 || H < 10) return;
    const m = { l: 46, r: 12, t: 16, b: 14 };
    let lo = Infinity, hi = -Infinity;
    series.forEach((s) => s.pts.forEach(([x, y]) => { if (x >= win[0] && x <= win[1] && isFinite(y)) { lo = Math.min(lo, y); hi = Math.max(hi, y); } }));
    if (opts && opts.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    const pad = (hi - lo) * 0.08 || 1; lo -= pad; hi += pad;
    const X = (x) => m.l + (x - win[0]) / (win[1] - win[0]) * (W - m.l - m.r);
    const Y = (y) => m.t + (hi - y) / (hi - lo) * (H - m.t - m.b);
    el('line', { class: 'ax', x1: m.l, y1: m.t, x2: m.l, y2: H - m.b }, svgEl);
    const tt = el('text', { class: 'title', x: m.l + 6, y: 11 }, svgEl); tt.textContent = title;
    [hi - pad, lo + pad].forEach((v) => {
      const t = el('text', { x: m.l - 6, y: Y(v) + 3, 'text-anchor': 'end' }, svgEl);
      t.textContent = (Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(1)) + unit;
    });
    if (opts && opts.zero) el('line', { class: 'zero', x1: m.l, y1: Y(0), x2: W - m.r, y2: Y(0) }, svgEl);
    series.forEach((s) => {
      const pts = s.pts.filter(([x]) => x >= win[0] && x <= win[1]);
      const every = Math.max(1, Math.floor(pts.length / (W * 1.5)));
      const d = pts.filter((_, k) => k % every === 0).map(([x, y], k) => (k ? 'L' : 'M') + X(x).toFixed(1) + ' ' + Y(y).toFixed(1)).join('');
      el('path', { d, stroke: s.colour, 'stroke-width': s.width || 1.3 }, svgEl);
    });
    if (S.cursor !== null && S.cursor >= win[0] && S.cursor <= win[1]) {
      el('line', { class: 'cur', x1: X(S.cursor), y1: m.t, x2: X(S.cursor), y2: H - m.b }, svgEl);
    }
    svgEl.onclick = (e) => {
      const x = e.clientX - svgEl.getBoundingClientRect().left;
      setCursor(win[0] + (x - m.l) / (W - m.l - m.r) * (win[1] - win[0]));
    };
  }

  function renderCharts() {
    if (!S.ref) return;
    const win = chartWindow();
    $('b-win').textContent = win[1] - win[0] >= S.info.lapMeters - 1 ? 'Full lap'
      : `${(win[0] / 1000).toFixed(2)}–${(win[1] / 1000).toFixed(2)} km`;
    const laps = [...S.selected].sort((a, b) => (a === S.info.best) - (b === S.info.best));
    const mk = (key) => laps.map((lap) => ({ colour: S.colours[lap], width: lap === S.info.best ? 1.8 : 1.2,
      pts: S.traces[lap].map((p) => [p.d, p[key]]) }));
    chart($('c-speed'), 'Speed', '', mk('speed'), win);
    chart($('c-line'), 'Line · offset from reference (+ left)', ' m', mk('off'), win, { zero: true });
    const delta = laps.map((lap) => ({ colour: S.colours[lap], width: lap === S.info.best ? 1.8 : 1.2,
      pts: Array.from(S.timeGrid[lap], (t, k) => [k * GRID_M, t - S.timeGrid[S.info.best][k]]) }));
    chart($('c-delta'), 'Delta to best (+ slower)', ' s', delta, win, { zero: true });
  }

  function renderAll() {
    syncLapRows();
    renderLines();
    renderLegend();
    renderCursor();
    renderCharts();
    renderPoint();
    $('b-info').textContent = `${S.selected.size} laps shown · ${S.info.list.filter((L) => L.complete).length} complete`;
    AccDash.hooks.forEach((f) => f());
  }

  // Shared with views.js: session state, helpers, and render hooks (called after every change)
  window.AccDash = { state: S, fmtLap, hooks: [], refreshMap: () => S.ref && applyView() };
})();
