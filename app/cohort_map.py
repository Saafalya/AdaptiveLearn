"""Interactive cohort map: every held-out student as a dot, clickable, with a what-if ghost.

Built on Streamlit's inline components (st.components.v2). The JavaScript runs on every
rerun with fresh `data`; it keeps its canvas and animation state on the mount point so
the load animation plays once and later updates glide instead of redrawing from scratch.
"""
from pathlib import Path

import numpy as np
import streamlit as st

LOADED_MTIME = Path(__file__).stat().st_mtime

CSS = """
:host { display: block; }
.wrap { position: relative; width: 100%; height: var(--h, 480px); font-family: var(--body); }
canvas { display: block; width: 100%; height: 100%; border-radius: 22px; cursor: crosshair; }
canvas.is-pointing { cursor: pointer; }
canvas:focus-visible { outline: 2px solid var(--blue-soft); outline-offset: 3px; }
.tip {
  position: absolute; pointer-events: none; z-index: 2; min-width: 168px;
  padding: 10px 12px; border-radius: 12px; font-size: 12.5px; line-height: 1.45; color: var(--text);
  background: var(--tip-bg); border: 1px solid var(--line-strong);
  box-shadow: 0 18px 40px -16px var(--shadow);
  opacity: 0; transform: translateY(4px); transition: opacity 120ms ease-out, transform 120ms ease-out;
}
.tip.on { opacity: 1; transform: translateY(0); }
.tip b { font-weight: 600; }
.tip .o0 { color: var(--blue-soft); } .tip .o1 { color: var(--red-soft); }
.tip small { display: block; margin-top: 4px; color: var(--mute); }
@media (max-width: 640px) { .wrap { height: 380px; } }
"""

JS = r"""
export default function (component) {
  const { data, parentElement, setStateValue } = component;
  let S = parentElement.__al;
  if (!S) { S = init(parentElement); parentElement.__al = S; }
  S.setState = setStateValue;
  S.update(data);
}

function init(root) {
  const wrap = document.createElement('div'); wrap.className = 'wrap';
  const canvas = document.createElement('canvas');
  canvas.tabIndex = 0;
  canvas.setAttribute('role', 'img');
  const tip = document.createElement('div'); tip.className = 'tip';
  wrap.append(canvas, tip); root.appendChild(wrap);
  const ctx = canvas.getContext('2d');
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

  let BLUE = [91, 134, 255], RED = [255, 74, 102], C = {};
  const FONT = "'Geist', 'Segoe UI', system-ui, sans-serif";
  const rgb = (hex) => { const h = hex.trim().replace('#', ''); return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)); };
  function readTheme() {
    const cs = getComputedStyle(wrap), v = (n) => cs.getPropertyValue(n).trim();
    C = { top: v('--canvas-top'), bottom: v('--canvas-bottom'), grid: v('--grid'), tick: v('--tick'),
          label: v('--mute'), text: v('--text'), ring: v('--text'), guide: v('--guide') };
    if (v('--blue')) BLUE = rgb(v('--blue'));
    if (v('--red')) RED = rgb(v('--red'));
    S.dotAlpha = parseFloat(v('--dot-alpha')) || 0.42;
  }
  const PAD = { l: 58, r: 22, t: 42, b: 80 };
  const S = { pts: [], sel: null, ghost: null, ghostFrom: null, ghostAt: 0, base: null,
              born: 0, mouse: null, hover: -1, W: 0, H: 0, dpr: 1, dirty: true, meta: null };

  const ease = (t) => 1 - Math.pow(1 - t, 4);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  function sx(x) { return PAD.l + (x / S.meta.xmax) * (S.W - PAD.l - PAD.r); }
  // scores 0-100 use the upper area; "nothing submitted yet" (-1) sits in a band at the bottom
  function sy(y) {
    const top = PAD.t, band = S.H - PAD.b + 26, plotBottom = S.H - PAD.b - 6;
    return y < 0 ? band : plotBottom - (y / 100) * (plotBottom - top);
  }

  function resize() {
    const r = wrap.getBoundingClientRect();
    S.dpr = Math.min(window.devicePixelRatio || 1, 2);
    S.W = r.width; S.H = r.height;
    canvas.width = Math.round(r.width * S.dpr); canvas.height = Math.round(r.height * S.dpr);
    S.dirty = true;
  }
  new ResizeObserver(resize).observe(wrap);
  if (document.fonts) document.fonts.ready.then(() => { S.dirty = true; });   // redraw labels in Geist once loaded

  function nearest(mx, my, within) {
    let best = -1, bd = within * within;
    for (let i = 0; i < S.pts.length; i++) {
      const p = S.pts[i], dx = p.px - mx, dy = p.py - my, d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }

  canvas.addEventListener('pointermove', (e) => {
    const r = canvas.getBoundingClientRect();
    S.mouse = { x: e.clientX - r.left, y: e.clientY - r.top };
    const i = nearest(S.mouse.x, S.mouse.y, 12);
    if (i !== S.hover) { S.hover = i; showTip(); }
    canvas.classList.toggle('is-pointing', i >= 0);
    S.dirty = true;
  });
  canvas.addEventListener('pointerleave', () => { S.mouse = null; S.hover = -1; showTip(); S.dirty = true; });
  canvas.addEventListener('click', () => {
    if (S.hover < 0 || !S.setState) return;
    S.setState('selected', S.pts[S.hover].k);
  });

  function showTip() {
    if (S.hover < 0) { tip.classList.remove('on'); return; }
    const p = S.pts[S.hover], m = S.meta;
    const outcome = m.results[p.r];
    const score = p.y < 0 ? 'nothing submitted yet' : `scoring ${Math.round(p.y)}% so far`;
    tip.innerHTML = `<b>Module ${m.modules[p.m]}, ${m.pres[p.pr]}</b><br>` +
      `<span class="o${p.o}">${outcome}</span> at the end of the course<br>` +
      `${p.c.toLocaleString()} clicks by day 90, ${score}` +
      `<small>Click to open this student's forecast</small>`;
    const left = clamp(p.px + 14, 8, S.W - 190), top = clamp(p.py - 18, 8, S.H - 110);
    tip.style.left = left + 'px'; tip.style.top = top + 'px';
    tip.classList.add('on');
  }

  function drawAxes() {
    const m = S.meta;
    ctx.save();
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
    ctx.fillStyle = C.tick; ctx.font = '11px ' + FONT;
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (const v of [0, 25, 50, 75, 100]) {
      const y = sy(v); ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(S.W - PAD.r, y); ctx.stroke();
      ctx.fillText(v + '%', PAD.l - 10, y);
    }
    ctx.fillText('none', PAD.l - 10, sy(-1));
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (const c of m.xticks) {
      const x = sx(Math.log1p(c));
      ctx.fillText(c >= 1000 ? (c / 1000) + 'k' : String(c), x, S.H - PAD.b + 46);
    }
    ctx.fillStyle = C.label; ctx.font = '12px ' + FONT;
    ctx.textAlign = 'right'; ctx.textBaseline = 'alphabetic';
    ctx.fillText('Clicks on course pages by day 90 (log scale)', S.W - PAD.r, S.H - 12);
    ctx.textAlign = 'left';
    ctx.fillText('Average assessment score so far', PAD.l - 34, 22);
    ctx.restore();
  }

  function frame(now) {
    if (!wrap.isConnected) return;               // unmounted: stop the loop
    requestAnimationFrame(frame);
    const intro = reduce ? 1 : clamp((now - S.born) / 1700, 0, 1);
    const gt = reduce ? 1 : clamp((now - S.ghostAt) / 520, 0, 1);
    if (!S.dirty && intro >= 1 && gt >= 1) return;
    S.dirty = false;
    if (!S.meta || !S.W) return;

    ctx.setTransform(S.dpr, 0, 0, S.dpr, 0, 0);
    ctx.clearRect(0, 0, S.W, S.H);
    const g = ctx.createLinearGradient(0, 0, 0, S.H);
    g.addColorStop(0, C.top); g.addColorStop(1, C.bottom);
    ctx.fillStyle = g; ctx.fillRect(0, 0, S.W, S.H);
    drawAxes();

    const mx = S.mouse ? S.mouse.x : -999, my = S.mouse ? S.mouse.y : -999;
    const floor = sy(0) + 6;
    for (let i = 0; i < S.pts.length; i++) {
      const p = S.pts[i];
      // intro: dots rise from the baseline, staggered left to right
      const local = reduce ? 1 : ease(clamp((intro * 1.45) - p.x / S.meta.xmax * 0.45, 0, 1));
      const tx = sx(p.x) + p.jx, ty = sy(p.y) + p.jy;
      p.px = tx; p.py = floor + (ty - floor) * local;
      const d = Math.hypot(p.px - mx, p.py - my);
      const near = d < 70 ? 1 - d / 70 : 0;
      const c = p.o ? RED : BLUE;
      ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${S.dotAlpha + near * (1 - S.dotAlpha)})`;
      ctx.beginPath(); ctx.arc(p.px, p.py, 1.7 + near * 1.9, 0, 6.2832); ctx.fill();
    }

    if (S.hover >= 0) {
      const p = S.pts[S.hover];
      ctx.strokeStyle = C.ring; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(p.px, p.py, 6, 0, 6.2832); ctx.stroke();
    }

    if (S.sel && intro >= 1) {
      const x = sx(S.sel.x), y = sy(S.sel.y);
      ctx.strokeStyle = C.guide; ctx.setLineDash([3, 4]); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, S.H - PAD.b + 30);
      ctx.moveTo(PAD.l, y); ctx.lineTo(S.W - PAD.r, y); ctx.stroke(); ctx.setLineDash([]);
      ctx.strokeStyle = C.ring; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, 8, 0, 6.2832); ctx.stroke();

      // the sliders round to whole numbers, so ignore differences smaller than that
      const moved = S.ghost && (Math.abs(S.ghost.x - S.sel.x) > 0.02 || Math.abs(S.ghost.y - S.sel.y) > 1);
      if (moved) {
        const from = S.ghostFrom || S.sel, k = ease(gt);
        const gx = sx(from.x + (S.ghost.x - from.x) * k), gy = sy(from.y + (S.ghost.y - from.y) * k);
        ctx.strokeStyle = C.label; ctx.setLineDash([2, 5]); ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(gx, gy); ctx.stroke(); ctx.setLineDash([]);
        const rg = ctx.createRadialGradient(gx - 3, gy - 3, 1, gx, gy, 11);
        const col = S.ghost.risk >= 0.5 ? RED : BLUE;
        rg.addColorStop(0, '#ffffff'); rg.addColorStop(0.45, `rgb(${col.join(',')})`);
        rg.addColorStop(1, `rgba(${col.join(',')},0.15)`);
        ctx.fillStyle = rg; ctx.beginPath(); ctx.arc(gx, gy, 10, 0, 6.2832); ctx.fill();
        ctx.fillStyle = C.text; ctx.font = '600 12px ' + FONT;
        ctx.textAlign = gx > S.W - 140 ? 'right' : 'left'; ctx.textBaseline = 'bottom';
        ctx.fillText(`What-if: ${Math.round(S.ghost.risk * 100)}% risk`, gx + (gx > S.W - 140 ? -14 : 14), gy - 8);
      }
    }
  }

  S.update = (d) => {
    if (d.height) wrap.style.setProperty('--h', d.height + 'px');
    readTheme(); requestAnimationFrame(() => { readTheme(); S.dirty = true; });
    if (!S.meta) {
      S.meta = d.meta;
      const n = d.x.length;
      for (let i = 0; i < n; i++) {
        const k = d.k[i];
        // stable jitter per student so overlapping dots separate without moving between renders
        const h1 = Math.sin(k * 12.9898) * 43758.5453, h2 = Math.sin(k * 78.233) * 12345.6789;
        S.pts.push({ x: d.x[i] / 1000, y: d.y[i], o: d.o[i], r: d.r[i], m: d.m[i], pr: d.p[i],
                     c: d.c[i], k, jx: (h1 - Math.floor(h1) - 0.5) * 3, jy: (h2 - Math.floor(h2) - 0.5) * (d.y[i] < 0 ? 18 : 3) });
      }
      canvas.setAttribute('aria-label', `Map of ${n} held-out students by activity and early score. ` +
        'Blue students passed, red failed or withdrew. Use the controls below to choose a student.');
      S.born = performance.now();
      requestAnimationFrame(frame);
    }
    const prevGhost = S.ghost && S.ghost.key === (d.sel && d.sel.k) ? S.ghost : null;
    S.sel = d.sel ? { x: d.sel.x / 1000, y: d.sel.y, k: d.sel.k } : null;
    const ng = d.ghost ? { x: d.ghost.x / 1000, y: d.ghost.y, risk: d.ghost.risk, key: d.sel && d.sel.k } : null;
    if (ng && (!prevGhost || prevGhost.x !== ng.x || prevGhost.y !== ng.y || prevGhost.risk !== ng.risk)) {
      S.ghostFrom = prevGhost ? { x: prevGhost.x, y: prevGhost.y } : null;
      S.ghostAt = performance.now();
    }
    S.ghost = ng;
    S.dirty = true;
  };

  return S;
}
"""

_component = st.components.v2.component("adaptivelearn_cohort_map", css=CSS, js=JS)

RESULTS = ["Distinction", "Pass", "Fail", "Withdrawn"]


def _xy(clicks, score):
    x = int(round(np.log1p(max(float(clicks), 0.0)) * 1000))
    y = -1 if score is None or (isinstance(score, float) and np.isnan(score)) else float(score)
    return x, y


@st.cache_data
def cohort_payload(test):
    """Compact column arrays for the held-out students (sent once, cached)."""
    modules = sorted(test.code_module.unique())
    pres = sorted(test.code_presentation.unique())
    x = (np.log1p(test.total_clicks.clip(lower=0)) * 1000).round().astype(int)
    y = test.mean_score_early.fillna(-1).round(1)
    xmax = float(np.log1p(test.total_clicks.clip(lower=0).quantile(0.995))) * 1.04
    return {
        "x": x.tolist(), "y": y.tolist(), "k": test.key.astype(int).tolist(),
        "o": test.final_result.isin(["Fail", "Withdrawn"]).astype(int).tolist(),
        "r": test.final_result.map(RESULTS.index).astype(int).tolist(),
        "m": test.code_module.map(modules.index).astype(int).tolist(),
        "p": test.code_presentation.map(pres.index).astype(int).tolist(),
        "c": test.total_clicks.astype(int).tolist(),
        "meta": {"modules": modules, "pres": pres, "results": RESULTS, "xmax": xmax,
                 "xticks": [1, 10, 100, 1000]},
    }


def cohort_map(test, selected_row, whatif_row, whatif_risk, on_pick, theme="dark", height=480):
    payload = dict(cohort_payload(test))
    payload["theme"] = theme
    payload["height"] = height
    sx, sy = _xy(selected_row.total_clicks, selected_row.mean_score_early)
    gx, gy = _xy(whatif_row["total_clicks"], whatif_row["mean_score_early"])
    payload["sel"] = {"x": sx, "y": sy, "k": int(selected_row.key)}
    payload["ghost"] = None if whatif_risk is None else {"x": gx, "y": gy, "risk": round(float(whatif_risk), 3)}
    return _component(data=payload, key="cohort", on_selected_change=on_pick)
