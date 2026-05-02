"""Build script for the standalone GRB Detection Rate HTML app.

Zips grb_detect/ + standalone_bridge.py, base64-encodes the zip,
injects it into the HTML template, and writes grb_detection_rate.html.

Usage:
    python build_standalone.py
"""
from __future__ import annotations

import base64
import io
import pathlib
import zipfile

ROOT = pathlib.Path(__file__).parent

# ---------------------------------------------------------------------------
# Custom slider helper — generates .cs-wrap HTML blocks
# %%SLIDER_X%% placeholders in HTML_TEMPLATE are replaced in build()
# ---------------------------------------------------------------------------

def _cs_slider(sid: str, smin: float, smax: float, step: float,
               default: float, marks: list) -> str:
    """Generate a .cs-wrap custom slider block with correct thumb/fill alignment."""
    def pct(v: float) -> float:
        return (v - smin) / (smax - smin)

    def align(p: float) -> float:
        return 0 if p < 1e-6 else (1 if p > 1 - 1e-6 else 0.5)

    def fmt(v: float) -> str:
        return f'{v:g}'

    ticks: list = []
    mark_spans: list = []
    for val, lbl in marks:
        p = pct(val)
        is_dflt = abs(val - default) < step * 0.5
        cls = ' cs-mark-dflt' if is_dflt else ''
        a = align(p)
        mark_spans.append(
            f'<span class="cs-mark{cls}" style="--mpct:{p:.4f};--align:{a}">{lbl}</span>')
        if 1e-6 < p < 1 - 1e-6:
            ticks.append(
                f'<div class="cs-tick" data-pct="{p:.4f}" style="--tick-pct:{p:.4f}"></div>')

    dflt_pct = pct(default)
    return (
        f'<div class="cs-wrap" data-id="{sid}" style="--val-pct:{dflt_pct:.4f}">'
        f'<div class="cs-track-area">'
        f'<div class="cs-track-bg"></div>'
        f'<div class="cs-track-fill"></div>'
        + ''.join(ticks)
        + f'<div class="cs-thumb" tabindex="0" role="slider"'
        f' aria-valuemin="{fmt(smin)}" aria-valuemax="{fmt(smax)}"'
        f' aria-valuenow="{fmt(default)}"></div>'
        f'</div>'
        f'<div class="cs-marks">{"".join(mark_spans)}</div>'
        f'<input type="range" id="{sid}_slider" class="cs-hidden" tabindex="-1"'
        f' min="{fmt(smin)}" max="{fmt(smax)}" step="{fmt(step)}"'
        f' value="{fmt(default)}">'
        f'</div>'
    )

_SLIDERS = [
    # (KEY, sid, min, max, step, default, marks)
    ('I',           'i',           2,      100,     1,      10,      [(2,'2'),(10,'10'),(30,'30'),(100,'100')]),
    ('FLIVE',       'flive',       0.01,   1,       0.01,   0.2,     [(0.01,'0.01'),(0.2,'0.2'),(0.5,'0.5'),(1,'1')]),
    ('ALOG',        'Alog',        -12,    -2,      0.01,   -4.68,   [(-12,'-12'),(-8,'-8'),(-4.68,'-4.68'),(-2,'-2')]),
    ('OMEGAEXP',    'omegaexp',    1,      200,     1,      47,      [(1,'1'),(47,'47'),(100,'100'),(200,'200')]),
    ('TOH',         'toh',         0,      30,      0.5,    0,       [(0,'0'),(15,'15'),(30,'30')]),
    ('OMEGA_SRV',   'omega_srv',   100,    41253,   100,    27500,   [(100,'100'),(10000,'10k'),(27500,'27.5k'),(41253,'41k')]),
    ('TNIGHT',      'tnight',      4,      14,      0.25,   10,      [(4,'4'),(6,'6'),(8,'8'),(10,'10'),(12,'12'),(14,'14')]),
    ('P',           'p',           2.01,   3,       0.01,   2.5,     [(2.01,'2'),(2.5,'2.5'),(3,'3')]),
    ('NU_LOG',      'nu_log',      14.3,   15.1,    0.01,   14.7,    [(14.3,'14.3'),(14.7,'14.7'),(15.1,'15.1')]),
    ('EKISO_LOG',   'Ekiso_log',   51,     55,      0.1,    53,      [(51,'51'),(52,'52'),(53,'53'),(54,'54'),(55,'55')]),
    ('N0_LOG',      'n0_log',      -3,     2,       0.1,    0,       [(-3,'-3'),(-2,'-2'),(-1,'-1'),(0,'0'),(1,'1'),(2,'2')]),
    ('GAMMA0_LOG',  'gamma0_log',  2,      3.5,     0.05,   2.5,     [(2,'2'),(2.5,'2.5'),(3,'3'),(3.5,'3.5')]),
    ('THETAJ',      'thetaj',      0.01,   0.5,     0.01,   0.1,     [(0.01,'0.01'),(0.1,'0.1'),(0.3,'0.3'),(0.5,'0.5')]),
    ('EPSE',        'epse',        -2,     -0.3,    0.05,   -1,      [(-2,'-2'),(-1,'-1'),(-0.3,'-0.3')]),
    ('EPSB',        'epsB',        -3,     -1,      0.05,   -2,      [(-3,'-3'),(-2,'-2'),(-1,'-1')]),
    ('DEUC',        'deuc',        1,      12,      0.01,   5.28,    [(1,'1'),(5.28,'5.28'),(8,'8'),(12,'12')]),
    ('RHO_GRB_LOG', 'rho_grb_log', 1,      3.3,     0.005,  2.415,   [(1,'10'),(2,'100'),(2.415,'260'),(3,'1k'),(3.3,'2k')]),
]

# ---------------------------------------------------------------------------
# HTML template — %%PHYSICS_ZIP_B64%% and %%SLIDER_*%% are replaced in build()
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GRB Detection Rate Explorer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/pyodide/v0.27.0/full/pyodide.js"></script>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
/* ── Design tokens ────────────────────────────────────────────────────────── */
:root {
  --bg:#080c14; --surface-0:#0e1421; --surface-1:#141c2e;
  --surface-2:#1c2640; --surface-3:#243050;
  --border:rgba(255,255,255,0.07); --border-hover:rgba(255,255,255,0.14);
  --text-hi:#f0f4ff; --text-mid:#8ba0c0; --text-lo:#445570;
  --accent:#6d9eff; --accent-dim:rgba(109,158,255,0.10); --accent-glow:rgba(109,158,255,0.25);
  --amber:#fbbf24; --coral:#f87171; --green:#34d399;
  --font-ui:'Inter',system-ui,sans-serif;
  --font-mono:'JetBrains Mono','Cascadia Code','Fira Code',monospace;
  --r-sm:5px; --r-md:8px; --r-lg:12px;
  --shadow-panel:0 4px 24px rgba(0,0,0,0.50);
  --t-fast:120ms ease; --t-mid:200ms ease;
  --t-slow:250ms cubic-bezier(0.4,0,0.2,1);
}
[data-theme="light"] {
  --bg:#f0f4f8; --surface-0:#ffffff; --surface-1:#f8fafc;
  --surface-2:#eef2f7; --surface-3:#e4eaf4;
  --border:rgba(0,0,0,0.08); --border-hover:rgba(0,0,0,0.16);
  --text-hi:#0f172a; --text-mid:#4b6080; --text-lo:#94aec8;
  --accent:#3b6fff; --accent-dim:rgba(59,111,255,0.08); --accent-glow:rgba(59,111,255,0.20);
}
/* ── Reset & base ─────────────────────────────────────────────────────────── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden}
body{font-family:var(--font-ui);background:var(--bg);color:var(--text-hi);font-size:13px;line-height:1.5}
button{cursor:pointer;border:none;background:none;font-family:inherit}
select{font-family:inherit}
input{font-family:inherit}
/* ── App shell ───────────────────────────────────────────────────────────── */
#app{display:flex;flex-direction:column;height:100vh;overflow:hidden}
/* ── Navbar ──────────────────────────────────────────────────────────────── */
.app-navbar{
  display:flex;align-items:center;gap:10px;
  min-height:52px;height:52px;flex-shrink:0;
  background:var(--surface-1);border-bottom:1px solid var(--border);
  box-shadow:var(--shadow-panel);padding:0 12px 0 8px;z-index:100;
}
.app-title{flex:1;font-size:14px;font-weight:600;letter-spacing:0.4px;color:var(--text-hi);overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.navbar-right{display:flex;gap:8px;flex-shrink:0;align-items:center}
.nav-btn,.icon-btn{
  display:inline-flex;align-items:center;justify-content:center;
  height:30px;padding:0 10px;
  border:1px solid var(--border);border-radius:var(--r-sm);
  background:var(--surface-2);color:var(--text-mid);
  font-family:var(--font-ui);font-size:12px;font-weight:500;
  cursor:pointer;white-space:nowrap;outline:none;user-select:none;
  transition:background var(--t-fast),color var(--t-fast),border-color var(--t-fast);
}
.nav-btn:hover,.icon-btn:hover{background:var(--surface-3);color:var(--text-hi);border-color:var(--border-hover)}
.preset-select{
  height:30px;padding:3px 8px;
  background:var(--surface-2);border:1px solid var(--border);border-radius:var(--r-sm);
  color:var(--text-mid);font-size:12px;font-weight:500;
  transition:border-color var(--t-fast),color var(--t-fast);
}
.preset-select:hover{color:var(--text-hi);border-color:var(--border-hover)}
.preset-select:focus{border-color:var(--accent);outline:none}
.view-controls{display:flex;height:30px;border:1px solid var(--border);border-radius:var(--r-md);background:var(--surface-2);overflow:hidden}
.view-btn{
  height:30px;padding:0 10px;border-right:1px solid var(--border);
  background:transparent;color:var(--text-mid);
  font-family:var(--font-ui);font-size:12px;font-weight:500;
  cursor:pointer;outline:none;user-select:none;white-space:nowrap;
  transition:background var(--t-fast),color var(--t-fast);
}
.view-btn:last-child{border-right:none}
.view-btn:hover{background:var(--surface-3);color:var(--text-hi)}
.view-btn.active{background:var(--accent);color:#fff;font-weight:600}
/* ── Metrics strip ───────────────────────────────────────────────────────── */
.metrics-strip{
  display:flex;align-items:center;
  min-height:36px;height:36px;flex-shrink:0;
  padding:0 16px;
  background:var(--surface-0);border-bottom:1px solid var(--border);
  overflow-x:auto;overflow-y:hidden;white-space:nowrap;
}
.metrics-strip::-webkit-scrollbar{display:none}
.metric-badge{display:flex;align-items:baseline;gap:5px;padding:0 12px;white-space:nowrap}
.metric-sep{width:1px;height:16px;background:var(--border);flex-shrink:0}
.metric-sep-prominent{width:1px;height:26px;background:var(--text-lo);flex-shrink:0}
.metric-label{font-family:var(--font-mono);font-size:10px;font-weight:500;letter-spacing:0.4px;color:var(--text-lo)}
.metric-value{font-family:var(--font-mono);font-size:13px;font-weight:500;color:var(--text-mid)}
.metric-amber{color:var(--amber)}
.metric-coral{color:var(--coral)}
.metric-muted{color:var(--text-mid)}
.metric-gain{color:var(--text-mid)}
.metric-gain.positive{color:var(--green)}
.metric-gain.negative{color:var(--coral)}
/* ── Body area ───────────────────────────────────────────────────────────── */
.app-body{display:flex;flex:1;min-height:0;overflow:hidden}
/* ── Sidebar ─────────────────────────────────────────────────────────────── */
.app-sidebar{
  width:0;flex-shrink:0;overflow:hidden;
  background:var(--surface-1);border-right:1px solid var(--border);
  z-index:10;transition:width var(--t-slow);
}
.app-sidebar.open{width:300px}
.sidebar-inner{width:300px;height:100%;overflow-y:auto;overflow-x:hidden;padding:8px 0 16px;scrollbar-width:thin;scrollbar-color:var(--border) transparent}
.sidebar-inner::-webkit-scrollbar{width:10px}
.sidebar-inner::-webkit-scrollbar-track{background:transparent}
.sidebar-inner::-webkit-scrollbar-thumb{background:var(--border);border-radius:6px}
.sidebar-inner::-webkit-scrollbar-thumb:hover{background:var(--border-hover)}
/* ── Accordion ───────────────────────────────────────────────────────────── */
details.acc-item{border-bottom:1px solid var(--border)}
details.acc-item summary{
  display:flex;align-items:center;justify-content:space-between;
  padding:10px 14px;cursor:pointer;list-style:none;
  font-size:10px;font-weight:600;letter-spacing:1.2px;text-transform:uppercase;
  color:var(--text-lo);transition:color var(--t-fast),background var(--t-fast);
  user-select:none;
}
details.acc-item summary::-webkit-details-marker{display:none}
details.acc-item summary::after{content:"›";font-size:16px;line-height:1;transition:transform var(--t-fast)}
details.acc-item[open] summary{color:var(--accent);background:var(--accent-dim)}
details.acc-item[open] summary::after{transform:rotate(90deg)}
.acc-body{padding:8px 14px 12px}
/* ── Parameter blocks ────────────────────────────────────────────────────── */
.param-block{margin-bottom:10px}
.param-row{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px}
.param-label{font-size:11px;font-weight:500;color:var(--text-mid);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:195px}
.param-input{
  width:72px;min-width:72px;padding:3px 4px;
  border:1px solid var(--border);border-radius:var(--r-sm);
  background:var(--surface-2);color:var(--text-hi);
  font-family:var(--font-mono);font-size:12px;text-align:center;
  transition:border-color var(--t-fast),box-shadow var(--t-fast);
}
.param-input:focus{border-color:var(--accent);box-shadow:0 0 0 2px var(--accent-glow);outline:none}
/* ── Custom sliders (cs-*) ───────────────────────────────────────────────── */
/* All positions use calc(--cs-r + val * (100% - 2*--cs-r)) so thumb, fill,
   ticks and labels share one coordinate system — no browser thumb-offset hacks. */
:root{--cs-r:8px;--cs-track-h:4px;--cs-thumb-sz:16px}
.cs-wrap{position:relative;padding:4px 0 20px}
.cs-track-area{
  position:relative;height:var(--cs-thumb-sz);cursor:pointer;
}
.cs-track-bg{
  position:absolute;
  left:var(--cs-r);right:var(--cs-r);
  top:50%;height:var(--cs-track-h);transform:translateY(-50%);
  background:var(--surface-3);border-radius:2px;pointer-events:none;
}
.cs-track-fill{
  position:absolute;
  left:var(--cs-r);top:50%;
  width:calc(var(--val-pct,0) * (100% - 2*var(--cs-r)));
  height:var(--cs-track-h);transform:translateY(-50%);
  background:var(--accent);border-radius:2px;pointer-events:none;
}
.cs-tick{
  position:absolute;
  left:calc(var(--cs-r) + var(--tick-pct,0) * (100% - 2*var(--cs-r)));
  top:50%;transform:translate(-50%,-50%);
  width:6px;height:6px;border-radius:50%;
  background:var(--surface-2);border:1px solid var(--surface-3);
  pointer-events:none;transition:border-color var(--t-fast);
}
.cs-tick.active{border-color:var(--accent)}
.cs-thumb{
  position:absolute;
  left:calc(var(--cs-r) + var(--val-pct,0) * (100% - 2*var(--cs-r)));
  top:50%;transform:translate(-50%,-50%);
  width:var(--cs-thumb-sz);height:var(--cs-thumb-sz);border-radius:50%;
  background:var(--accent);border:2px solid var(--surface-1);
  box-shadow:0 0 0 3px var(--accent-glow);cursor:grab;z-index:1;
  transition:box-shadow var(--t-fast);outline:none;
}
.cs-thumb:hover,.cs-wrap.cs-dragging .cs-thumb{box-shadow:0 0 0 6px var(--accent-glow)}
.cs-wrap.cs-dragging .cs-thumb{cursor:grabbing}
.cs-thumb:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.cs-marks{position:relative;height:14px;margin-top:1px}
.cs-mark{
  position:absolute;
  left:calc(var(--cs-r) + var(--mpct,0) * (100% - 2*var(--cs-r)));
  top:0;transform:translateX(calc(-1 * var(--align,0.5) * 100%));
  font-family:var(--font-mono);font-size:10px;color:var(--text-mid);
  white-space:nowrap;pointer-events:none;line-height:14px;
}
.cs-mark-dflt{font-weight:700;font-size:11px;color:var(--text-hi)}
.cs-hidden{position:absolute;width:1px;height:1px;opacity:0;
           pointer-events:none;overflow:hidden;clip:rect(0,0,0,0)}
/* ── Switches ────────────────────────────────────────────────────────────── */
.param-switch-row{display:flex;align-items:center;gap:8px;margin:6px 0}
.toggle-switch{position:relative;display:inline-block;width:34px;height:18px;flex-shrink:0}
.toggle-switch input{opacity:0;width:0;height:0}
.toggle-track{
  position:absolute;inset:0;border-radius:9px;
  background:var(--surface-3);border:1px solid var(--border-hover);
  transition:background var(--t-fast);cursor:pointer;
}
.toggle-thumb{
  position:absolute;top:2px;left:2px;
  width:12px;height:12px;border-radius:50%;
  background:var(--text-lo);
  transition:transform var(--t-fast),background var(--t-fast);
}
.toggle-switch input:checked + .toggle-track{background:var(--accent);border-color:var(--accent)}
.toggle-switch input:checked + .toggle-track .toggle-thumb{transform:translateX(16px);background:#fff}
.switch-label{font-size:12px;color:var(--text-mid);cursor:pointer}
.toggle-hint{font-size:11px;color:var(--text-lo);line-height:1.45;margin-top:2px;margin-bottom:6px}
/* ── Group labels ────────────────────────────────────────────────────────── */
.param-group-label{font-size:10px;font-weight:600;letter-spacing:0.05em;text-transform:uppercase;color:var(--text-lo);margin:6px 0 2px}
/* ── Derived value display ───────────────────────────────────────────────── */
.grb-derived-row{margin-top:4px;gap:6px;flex-wrap:wrap}
.grb-derived-val{font-family:var(--font-mono);font-size:11px;color:var(--text-mid);background:var(--surface-2);border:1px solid var(--border);border-radius:var(--r-sm);padding:2px 6px;white-space:nowrap}
/* ── Main area ───────────────────────────────────────────────────────────── */
.app-main{display:flex;flex-direction:column;flex:1;min-width:0;overflow:hidden;background:var(--bg)}
.view-panel{flex:1;min-height:0;display:flex;flex-direction:column}
.view-panel.hidden{display:none}
.plot-div{flex:1;min-height:0}
/* ── Status bar ──────────────────────────────────────────────────────────── */
.status-bar{flex-shrink:0;font-family:var(--font-mono);font-size:11px;color:var(--text-lo);padding:5px 16px;border-top:1px solid var(--border);background:var(--surface-0)}
.status-bar.empty{padding:0;border-top:none}
/* ── Spinner ─────────────────────────────────────────────────────────────── */
@keyframes spin{to{transform:rotate(360deg)}}
.spinner{display:inline-block;width:10px;height:10px;border:2px solid var(--accent-dim);border-top-color:var(--accent);border-radius:50%;animation:spin 0.7s linear infinite;margin-right:6px;vertical-align:middle}
/* ── Derived info displays (sidebar constraints / strategy) ──────────────── */
.derived-info{display:inline-block;font-family:var(--font-mono);font-size:11px;color:var(--text-mid);background:var(--surface-2);border:1px solid var(--border);border-radius:var(--r-sm);padding:3px 8px;margin:0 0 8px}
.derived-info.warning{color:var(--amber);border-color:rgba(251,191,36,0.30);background:rgba(251,191,36,0.06)}
/* ── Slice position controls (below N-slice / t-slice graphs) ───────────── */
.slice-ctrl{display:flex;align-items:center;gap:10px;padding:6px 20px 12px 20px;border-top:1px solid var(--border);background:var(--surface-0);flex-shrink:0}
.slice-ctrl-label{font-family:var(--font-mono);font-size:11px;color:var(--text-mid);white-space:nowrap;min-width:80px;flex-shrink:0}
.slice-pos-slider{flex:1;display:flex;flex-direction:column;gap:2px}
.slice-pos-slider input[type=range]{width:100%;margin:0}
.slice-pos-marks{display:flex;justify-content:space-between;padding:0 var(--cs-r);font-family:var(--font-mono);font-size:10px;color:var(--text-lo);pointer-events:none}
.slice-pos-value{font-family:var(--font-mono);font-size:11px;color:var(--text-mid);min-width:70px;text-align:right;flex-shrink:0}
</style>
</head>
<body>
<div id="app">

<!-- ── Navbar ────────────────────────────────────────────────────────────── -->
<nav class="app-navbar">
  <button class="icon-btn" id="sidebar-toggle" title="Toggle sidebar">☰</button>
  <span class="app-title">GRB Detection Rate Explorer</span>
  <div class="navbar-right">
    <select class="preset-select" id="preset-select" title="Load preset parameters">
      <option value="none" disabled selected>— Presets —</option>
      <option value="ztf">ZTF</option>
      <option value="rubin">Rubin LSST</option>
    </select>
    <button class="nav-btn" id="export-btn" title="Download surface as CSV">⬇ CSV</button>
    <div class="view-controls">
      <button class="view-btn active" data-tab="3d" title="3D surface view">3D</button>
      <button class="view-btn" data-tab="nslice" title="R vs N_exp at fixed t_cad">N<sub>exp</sub> slice</button>
      <button class="view-btn" data-tab="tslice" title="R vs t_cad at fixed N_exp">t<sub>cad</sub> slice</button>
    </div>
    <button class="nav-btn" id="theme-toggle">☾ Dark</button>
  </div>
</nav>

<!-- ── Metrics strip ─────────────────────────────────────────────────────── -->
<div class="metrics-strip" id="metrics-strip">
  <div class="metric-badge"><span class="metric-label">R<sub>det,opt</sub></span><span class="metric-value metric-amber" id="m-R-opt">—</span></div>
  <div class="metric-sep"></div>
  <div class="metric-badge"><span class="metric-label">t<sub>cad,opt</sub></span><span class="metric-value metric-muted" id="m-tcad-opt">—</span></div>
  <div class="metric-sep"></div>
  <div class="metric-badge"><span class="metric-label">N<sub>exp,opt</sub></span><span class="metric-value metric-muted" id="m-N-opt">—</span></div>
  <div class="metric-sep"></div>
  <div class="metric-badge"><span class="metric-label">t<sub>exp,opt</sub></span><span class="metric-value metric-muted" id="m-texp-opt">—</span></div>
  <div class="metric-sep"></div>
  <div class="metric-sep-prominent"></div>
  <div class="metric-badge"><span class="metric-label">R<sub>det,ZTF</sub></span><span class="metric-value metric-coral" id="m-R-ztf">—</span></div>
  <div class="metric-sep"></div>
  <div class="metric-badge"><span class="metric-label">t<sub>cad,ZTF</sub></span><span class="metric-value metric-muted" id="m-tcad-ztf">—</span></div>
  <div class="metric-sep"></div>
  <div class="metric-badge"><span class="metric-label">N<sub>exp,ZTF</sub></span><span class="metric-value metric-muted" id="m-N-ztf">—</span></div>
  <div class="metric-sep"></div>
  <div class="metric-badge"><span class="metric-label">t<sub>exp,ZTF</sub></span><span class="metric-value metric-muted" id="m-texp-ztf">—</span></div>
  <div class="metric-sep"></div>
  <div class="metric-sep-prominent"></div>
  <div class="metric-badge"><span class="metric-label">Gain</span><span class="metric-value metric-gain" id="m-gain">—</span></div>
</div>

<!-- ── App body ───────────────────────────────────────────────────────────── -->
<div class="app-body">

<!-- ── Sidebar ───────────────────────────────────────────────────────────── -->
<aside class="app-sidebar open" id="app-sidebar">
<div class="sidebar-inner">

<!-- Strategy -->
<details class="acc-item" open>
<summary>Strategy</summary>
<div class="acc-body">
  <div class="param-block">
    <div class="param-row">
      <label class="param-label" for="i_slider" title="Number of detections required per GRB to count as a detection event">i</label>
      <input type="number" id="i_input" class="param-input" min="2" max="100" step="1" value="10">
    </div>
    %%SLIDER_I%%
  </div>
  <div class="param-block">
    <div class="param-row">
      <label class="param-label" for="flive_slider" title="Fraction of total time the telescope is actually observing (0=never, 1=always)">f<sub>live</sub></label>
      <input type="number" id="flive_input" class="param-input" min="0.01" max="1" step="0.01" value="0.2">
    </div>
    %%SLIDER_FLIVE%%
  </div>
</div>
</details>

<!-- Instrument -->
<details class="acc-item" open>
<summary>Instrument</summary>
<div class="acc-body">
  <div class="param-block">
    <div class="param-row">
      <label class="param-label" for="Alog_slider" title="log₁₀ of reference limiting flux in Jansky at t_exp_ref = 30 s">log A [Jy]</label>
      <input type="number" id="Alog_input" class="param-input" min="-12" max="-2" step="0.01" value="-4.68">
    </div>
    %%SLIDER_ALOG%%
  </div>
  <div class="param-block">
    <div class="param-row">
      <label class="param-label" for="omegaexp_slider" title="Single-exposure field of view of the instrument in square degrees">Ω<sub>exp</sub> [deg²]</label>
      <input type="number" id="omegaexp_input" class="param-input" min="1" max="200" step="1" value="47">
    </div>
    %%SLIDER_OMEGAEXP%%
  </div>
  <div class="param-block">
    <div class="param-row">
      <label class="param-label" for="toh_slider" title="Per-exposure overhead (readout, slew, settle) in seconds">t<sub>OH</sub> [sec]</label>
      <input type="number" id="toh_input" class="param-input" min="0" max="30" step="0.5" value="0">
    </div>
    %%SLIDER_TOH%%
  </div>
</div>
</details>

<!-- Constraints -->
<details class="acc-item" open>
<summary>Constraints</summary>
<div class="acc-body">
  <div class="param-block">
    <div class="param-row">
      <label class="param-label" for="omega_srv_slider" title="Maximum surveyable sky area per cadence cycle in square degrees">Ω<sub>srv,max</sub> [deg²]</label>
      <input type="number" id="omega_srv_input" class="param-input" min="100" max="41253" step="100" value="27500">
    </div>
    %%SLIDER_OMEGA_SRV%%
    <div id="nexpmax-display"></div>
  </div>
  <div class="param-block" id="tnight-block" style="display:none">
    <div class="param-row">
      <label class="param-label" for="tnight_slider" title="Length of the observable astronomical night in hours">t<sub>night</sub> [hr]</label>
      <input type="number" id="tnight_input" class="param-input" min="4" max="14" step="0.25" value="10">
    </div>
    %%SLIDER_TNIGHT%%
    <div id="subnight-limit-display"></div>
  </div>
</div>
</details>

<!-- Physics -->
<details class="acc-item">
<summary>Parameters</summary>
<div class="acc-body">
  <div class="param-group-label">Physical model</div>
  <div class="param-block">
    <div class="param-row">
      <label class="param-label" for="p_slider" title="Electron power-law index (must be > 2). Exponentially amplified in flux; strongly affects spectral slope and all detection ranges.">p</label>
      <input type="number" id="p_input" class="param-input" min="2.01" max="3" step="0.01" value="2.5">
    </div>
    %%SLIDER_P%%
  </div>
  <div class="param-block">
    <div class="param-row">
      <label class="param-label" for="nu_log_slider" title="log₁₀ of observing frequency in Hz. Optical/near-IR window: 10^14.3 Hz (~1 µm, J-band) to 10^15.1 Hz (~80 nm, near-UV). Default 5×10¹⁴ Hz = optical V-band (550 nm).">log ν [Hz]</label>
      <input type="number" id="nu_log_input" class="param-input" min="14.3" max="15.1" step="0.01" value="14.7">
    </div>
    %%SLIDER_NU_LOG%%
  </div>
  <div class="param-block">
    <div class="param-row">
      <label class="param-label" for="Ekiso_log_slider" title="log₁₀ of isotropic-equivalent kinetic energy in erg. Scales flux as E^{(p+3)/4} and deceleration time as E^{1/3}.">log E<sub>k,iso</sub> [erg]</label>
      <input type="number" id="Ekiso_log_input" class="param-input" min="51" max="55" step="0.1" value="53">
    </div>
    %%SLIDER_EKISO_LOG%%
  </div>
  <div class="param-block">
    <div class="param-row">
      <label class="param-label" for="n0_log_slider" title="log₁₀ of ISM number density in cm⁻³. Affects flux (∝ n^{1/2}) and deceleration time (∝ n^{-1/3}).">log n<sub>0</sub> [cm⁻³]</label>
      <input type="number" id="n0_log_input" class="param-input" min="-3" max="2" step="0.1" value="0">
    </div>
    %%SLIDER_N0_LOG%%
  </div>
  <div class="param-block">
    <div class="param-row">
      <label class="param-label" for="gamma0_log_slider" title="log₁₀ of initial bulk Lorentz factor. Affects deceleration time (∝ Γ₀^{-8/3}) and the on-axis beaming cone angle.">log Γ<sub>0</sub></label>
      <input type="number" id="gamma0_log_input" class="param-input" min="2" max="3.5" step="0.05" value="2.5">
    </div>
    %%SLIDER_GAMMA0_LOG%%
  </div>
  <div class="param-block">
    <div class="param-row">
      <label class="param-label" for="thetaj_slider" title="Jet half-opening angle in radians. Sets the jet-break time and the beaming fraction f_b = θ_j²/2.">θ<sub>j</sub> [rad]</label>
      <input type="number" id="thetaj_input" class="param-input" min="0.01" max="0.5" step="0.01" value="0.1">
    </div>
    %%SLIDER_THETAJ%%
  </div>
  <div class="param-block">
    <div class="param-row">
      <label class="param-label" for="epse_slider" title="log₁₀ of electron energy fraction. Enters flux as (ε_e·(p−2)/(p−1))^{p−1}.">log ε<sub>e</sub></label>
      <input type="number" id="epse_input" class="param-input" min="-2" max="-0.3" step="0.05" value="-1">
    </div>
    %%SLIDER_EPSE%%
  </div>
  <div class="param-block">
    <div class="param-row">
      <label class="param-label" for="epsB_slider" title="log₁₀ of magnetic energy fraction. Enters flux as ε_B^{(p+1)/4}.">log ε<sub>B</sub></label>
      <input type="number" id="epsB_input" class="param-input" min="-3" max="-1" step="0.05" value="-2">
    </div>
    %%SLIDER_EPSB%%
  </div>
  <div class="param-group-label" style="margin-top:10px">Cosmological model</div>
  <div class="param-block">
    <div class="param-row">
      <label class="param-label" for="deuc_slider" title="Euclidean calibration distance in Gpc. Sets the volume for the rate integral. Default 5.28 Gpc ≈ z = 2 in flat ΛCDM.">D<sub>Euc</sub> [Gpc]</label>
      <input type="number" id="deuc_input" class="param-input" min="1" max="12" step="0.01" value="5.28">
    </div>
    %%SLIDER_DEUC%%
  </div>
  <div class="param-block">
    <div class="param-row">
      <label class="param-label" for="rho_grb_log_slider" title="log₁₀ of GRB volumetric rate density. Scales the total detection rate linearly.">log ℛ [Gpc⁻³ yr⁻¹]</label>
      <input type="number" id="rho_grb_log_input" class="param-input" min="1" max="3.3" step="0.005" value="2.415">
    </div>
    %%SLIDER_RHO_GRB_LOG%%
  </div>
  <div class="param-row grb-derived-row">
    <span id="grb-ntotal-display" class="grb-derived-val">R<sub>int</sub> = —</span>
    <span id="grb-ntoward-display" class="grb-derived-val">f<sub>b</sub>R<sub>int</sub> = —</span>
  </div>
</div>
</details>

<!-- Settings -->
<details class="acc-item">
<summary>Settings</summary>
<div class="acc-body">
  <div class="param-switch-row">
    <label class="toggle-switch" for="optical-switch">
      <input type="checkbox" id="optical-switch">
      <span class="toggle-track"><span class="toggle-thumb"></span></span>
    </label>
    <span class="switch-label" onclick="document.getElementById('optical-switch').click()">Optical survey mode</span>
  </div>
  <p class="toggle-hint">Requires ≥ i detections in one night or a single detection each night. Multi-day effective live fraction is reduced by f<sub>night</sub> = t<sub>night</sub> / 24 h; the t<sub>night</sub> slider becomes active.</p>
  <div class="param-switch-row">
    <label class="toggle-switch" for="toh-approx-switch">
      <input type="checkbox" id="toh-approx-switch">
      <span class="toggle-track"><span class="toggle-thumb"></span></span>
    </label>
    <span class="switch-label" onclick="document.getElementById('toh-approx-switch').click()">t<sub>OH</sub> approximation</span>
  </div>
  <p class="toggle-hint">Uses the naive t<sub>exp</sub> = f<sub>live</sub>·t<sub>cad</sub>/N<sub>exp</sub> equation (ignoring t<sub>OH</sub> in the exposure formula) for fully analytic optimal strategies, while still enforcing f<sub>live</sub>·t<sub>cad</sub>/N<sub>exp</sub> &gt; t<sub>OH</sub> as the validity boundary.</p>
  <div class="param-switch-row">
    <label class="toggle-switch" for="regime-color-switch">
      <input type="checkbox" id="regime-color-switch">
      <span class="toggle-track"><span class="toggle-thumb"></span></span>
    </label>
    <span class="switch-label" onclick="document.getElementById('regime-color-switch').click()">Color by detection regime</span>
  </div>
  <p class="toggle-hint">Colors the surface by analytical detection regime. Warm colors (red→amber) indicate flux-limited (D<sub>Euc</sub>) regimes; cool colors (blue→teal) indicate cadence-limited (D<sub>i</sub>) regimes; gray indicates the singly-limited (D<sub>dec</sub>) regime. More vibrant colors correspond to higher detection ranges.</p>
  <div class="param-switch-row">
    <label class="toggle-switch" for="full-integral-switch">
      <input type="checkbox" id="full-integral-switch">
      <span class="toggle-track"><span class="toggle-thumb"></span></span>
    </label>
    <span class="switch-label" onclick="document.getElementById('full-integral-switch').click()">Full integral mode</span>
  </div>
  <p class="toggle-hint">Computes R<sub>det</sub> using the exact integral over all viewing angles q ∈ [0, q<sub>nr</sub>] (thesis Eq. 39), instead of the dominant-term approximation. Adds the tail contribution beyond q<sub>Euc</sub> (flux-limited) or q<sub>i</sub> (cadence-limited). Slower but more accurate, especially near range boundaries.</p>
  <div class="param-switch-row">
    <label class="toggle-switch" for="off-axis-switch">
      <input type="checkbox" id="off-axis-switch">
      <span class="toggle-track"><span class="toggle-thumb"></span></span>
    </label>
    <span class="switch-label" onclick="document.getElementById('off-axis-switch').click()">Off-axis detections only</span>
  </div>
  <p class="toggle-hint">Show only GRBs detected from outside the relativistic beaming cone (viewing angle q &gt; q<sub>dec</sub>). Subtracts the on-axis contribution (q &lt; q<sub>dec</sub>) from the rate integral. Regions where no off-axis detection is possible are masked out.</p>
</div>
</details>

</div><!-- sidebar-inner -->
</aside><!-- app-sidebar -->

<!-- ── Main area ─────────────────────────────────────────────────────────── -->
<main class="app-main">
  <div id="panel-3d" class="view-panel">
    <div id="plot-3d" class="plot-div"></div>
  </div>
  <div id="panel-nslice" class="view-panel hidden">
    <div id="plot-nslice" class="plot-div"></div>
    <div class="slice-ctrl">
      <span class="slice-ctrl-label">Fixed t<sub>cad</sub></span>
      <div class="slice-pos-slider">
        <input type="range" id="nslice-tfix-slider" min="2" max="8" step="0.05" value="4.937">
        <div class="slice-pos-marks">
          <span>100 s</span><span>1 hr</span><span>1 day</span><span>1 wk</span><span>1 yr</span>
        </div>
      </div>
      <span class="slice-pos-value" id="nslice-tfix-value">1 day</span>
    </div>
  </div>
  <div id="panel-tslice" class="view-panel hidden">
    <div id="plot-tslice" class="plot-div"></div>
    <div class="slice-ctrl">
      <span class="slice-ctrl-label">Fixed N<sub>exp</sub></span>
      <div class="slice-pos-slider">
        <input type="range" id="tslice-nfix-slider" min="0" max="4" step="0.05" value="2.0">
        <div class="slice-pos-marks">
          <span>1</span><span>10</span><span>100</span><span>1k</span><span>10k</span>
        </div>
      </div>
      <span class="slice-pos-value" id="tslice-nfix-value">100</span>
    </div>
  </div>
  <div class="status-bar" id="status-bar">Initializing Pyodide…</div>
</main>

</div><!-- app-body -->
</div><!-- app -->

<!-- ── Embedded physics package ───────────────────────────────────────────── -->
<script id="physics-zip" type="application/octet-stream">%%PHYSICS_ZIP_B64%%</script>

<!-- ── Application JavaScript ────────────────────────────────────────────── -->
<script>
'use strict';

// ── Constants ──────────────────────────────────────────────────────────────
const DAY_S = 86400;
const REGIME_HEX = ['#FF1744','#FF9100','#FFD740','#2979FF','#00E5FF','#1DE9B6','#9E9E9E'];
const REGIME_LABELS = [
  'Saturated · Range IV','Distance-limited · Range III','Distance-limited · Range II',
  'Cadence-limited · Range IV','Cadence-limited · Range III','Cadence-limited · Range II',
  'Flux-limited · Range I'
];
const TCAD_TICKVALS_H = [1/3600, 1/60, 1, 6, 24, 168, 730, 8760];
const TCAD_TICKTEXT   = ['1 sec','1 min','1 hr','6 hr','1 day','1 wk','1 mo','1 yr'];
const AMBER = '#fbbf24';
const CORAL = '#f87171';
const ZMIN_LOG = -2;

// Explicit Plasma colorscale (from plotly.colors.sequential.Plasma resolved via
// plotly.express.colors.make_colorscale). Passing the explicit array — instead
// of the string 'Plasma' — guarantees Plotly.js renders identically to Plotly.py,
// which is what Dash does behind the scenes.
const PLASMA_SCALE = [
  [0.0,                'rgb(13, 8, 135)'],
  [0.1111111111111111, 'rgb(70, 3, 159)'],
  [0.2222222222222222, 'rgb(114, 1, 168)'],
  [0.3333333333333333, 'rgb(156, 23, 158)'],
  [0.4444444444444444, 'rgb(189, 55, 134)'],
  [0.5555555555555556, 'rgb(216, 87, 107)'],
  [0.6666666666666666, 'rgb(237, 121, 83)'],
  [0.7777777777777777, 'rgb(251, 159, 58)'],
  [0.8888888888888888, 'rgb(253, 202, 38)'],
  [1.0,                'rgb(240, 249, 33)'],
];

// Presets (match callbacks/ui.py) — touch only: i, f_live, A_log, omega_exp, t_oh, omega_srv, optical.
const PRESETS = {
  ztf:   {i:10, f_live:0.2,  A_log:-4.68, omega_exp:47,  t_oh:15, omega_srv:27500, optical:true},
  rubin: {i:10, f_live:0.7,  A_log:-7.0,  omega_exp:9.6, t_oh:30, omega_srv:18000, optical:true},
};
// Map preset keys to DOM slider/switch IDs.
const PRESET_MAP = {
  i:         'i_slider',
  f_live:    'flive_slider',
  A_log:     'Alog_slider',
  omega_exp: 'omegaexp_slider',
  t_oh:      'toh_slider',
  omega_srv: 'omega_srv_slider',
};
let _activePresetKey = null;     // which preset is currently active (drift detection)
let _presetApplying  = false;    // suppress drift detection while we apply a preset

// ── State ──────────────────────────────────────────────────────────────────
let pyodide = null;
let pyComputeAll = null;
let pyComputeNslice = null;
let pyComputeTslice = null;
let _computing = false;
let _pendingUpdate = false;
let _debounceTimer = null;
let _sliceDebounceTimer = null;   // independent debounce for slice-position drags
let _sliceComputing = { nslice: false, tslice: false };
let _sliceQueued    = { nslice: false, tslice: false };
let _lastData = null;
let _currentTab = '3d';
let _currentTheme = 'dark';

// ── Status bar ─────────────────────────────────────────────────────────────
function setStatus(msg, spinning = false) {
  const bar = document.getElementById('status-bar');
  bar.innerHTML = spinning
    ? '<span class="spinner"></span>' + msg
    : msg;
  bar.className = 'status-bar' + (msg ? '' : ' empty');
}

// ── Theme ──────────────────────────────────────────────────────────────────
function setTheme(theme) {
  _currentTheme = theme;
  document.documentElement.dataset.theme = theme;
  document.getElementById('theme-toggle').textContent = theme === 'dark' ? '☾ Dark' : '☀ Light';
  if (_lastData) rerenderAll(_lastData);
}

document.getElementById('theme-toggle').addEventListener('click', () => {
  setTheme(_currentTheme === 'dark' ? 'light' : 'dark');
});

// ── Sidebar ────────────────────────────────────────────────────────────────
document.getElementById('sidebar-toggle').addEventListener('click', () => {
  document.getElementById('app-sidebar').classList.toggle('open');
  setTimeout(() => Plotly.Plots && Plotly.Plots.resize && Plotly.Plots.resize(document.getElementById('plot-' + _currentTab)), 260);
});

// ── Tab switching ──────────────────────────────────────────────────────────
document.querySelectorAll('.view-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    if (tab === _currentTab) return;
    _currentTab = tab;
    document.querySelectorAll('.view-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
    document.querySelectorAll('.view-panel').forEach(p => p.classList.toggle('hidden', p.id !== 'panel-' + tab));
    if (_lastData) {
      if (tab === 'nslice') renderNSlice(_lastData);
      else if (tab === 'tslice') renderTSlice(_lastData);
    }
  });
});

// ── Slider ↔ input sync ────────────────────────────────────────────────────
const SLIDER_IDS = [
  'i','flive','Alog','omegaexp','toh','omega_srv','tnight',
  'p','nu_log','Ekiso_log','n0_log','gamma0_log','thetaj','epse','epsB','deuc','rho_grb_log'
];

// Visual helpers: paint filled track up to current value + toggle active-dot class.
function _sliderPct(sl) {
  const mn = parseFloat(sl.min), mx = parseFloat(sl.max);
  if (!isFinite(mn) || !isFinite(mx) || mx === mn) return 0;
  const v = parseFloat(sl.value);
  return Math.min(100, Math.max(0, (v - mn) / (mx - mn) * 100));
}
function updateSliderVisual(sl) {
  if (!sl) return;
  const pct = _sliderPct(sl) / 100;  // 0–1
  const wrap = sl.closest('.cs-wrap');
  if (!wrap) return;
  wrap.style.setProperty('--val-pct', pct.toFixed(4));
  wrap.querySelector('.cs-thumb')?.setAttribute('aria-valuenow', sl.value);
  wrap.querySelectorAll('.cs-tick').forEach(t => {
    const tp = parseFloat(t.dataset.pct);
    t.classList.toggle('active', isFinite(tp) && tp <= pct + 1e-6);
  });
}

function syncFromSlider(id) {
  const sl = document.getElementById(id + '_slider');
  const inp = document.getElementById(id + '_input');
  if (inp) inp.value = sl.value;
  updateSliderVisual(sl);
  if (id === 'deuc' || id === 'thetaj' || id === 'rho_grb_log') updateGrbCounts();
  _checkPresetDrift();
  triggerUpdate();
}

function syncFromInput(id) {
  const inp = document.getElementById(id + '_input');
  const sl = document.getElementById(id + '_slider');
  if (!inp || !sl) return;
  const raw = inp.value;
  const v = parseFloat(raw);
  // NaN / empty: revert to current slider value, do nothing
  if (!isFinite(v)) {
    inp.value = sl.value;
    return;
  }
  // Clamp to [slider.min, slider.max] — matches Dash callbacks/sync.py
  const mn = parseFloat(sl.min), mx = parseFloat(sl.max);
  const clamped = Math.min(Math.max(v, mn), mx);
  inp.value = clamped;
  sl.value = clamped;
  updateSliderVisual(sl);
  if (id === 'deuc' || id === 'thetaj' || id === 'rho_grb_log') updateGrbCounts();
  _checkPresetDrift();
  triggerUpdate();
}

// Wire drag/click/touch/keyboard on every .cs-wrap, then prime initial visuals.
function _initCustomSliders() {
  const CS_R = 8; // px — must match --cs-r CSS variable
  document.querySelectorAll('.cs-wrap').forEach(wrap => {
    const sl    = wrap.querySelector('.cs-hidden');
    const area  = wrap.querySelector('.cs-track-area');
    const thumb = wrap.querySelector('.cs-thumb');
    if (!sl || !area) return;
    updateSliderVisual(sl); // prime fill + ticks to initial value

    function valFromX(clientX) {
      const rect = area.getBoundingClientRect();
      const p = Math.min(1, Math.max(0, (clientX - rect.left - CS_R) / (rect.width - 2 * CS_R)));
      const mn = parseFloat(sl.min), mx = parseFloat(sl.max), st = parseFloat(sl.step) || 1;
      let v = mn + p * (mx - mn);
      if (st > 0) v = Math.round((v - mn) / st) * st + mn;
      return Math.min(mx, Math.max(mn, v));
    }
    function applyVal(v) {
      sl.value = v;
      sl.dispatchEvent(new Event('input', { bubbles: true }));
    }

    let dragging = false;
    area.addEventListener('mousedown', e => {
      dragging = true; wrap.classList.add('cs-dragging');
      applyVal(valFromX(e.clientX)); e.preventDefault();
    });
    window.addEventListener('mousemove', e => { if (dragging) applyVal(valFromX(e.clientX)); });
    window.addEventListener('mouseup',   () => { dragging = false; wrap.classList.remove('cs-dragging'); });

    area.addEventListener('touchstart', e => {
      dragging = true; wrap.classList.add('cs-dragging');
      applyVal(valFromX(e.touches[0].clientX)); e.preventDefault();
    }, { passive: false });
    window.addEventListener('touchmove', e => {
      if (dragging) { applyVal(valFromX(e.touches[0].clientX)); e.preventDefault(); }
    }, { passive: false });
    window.addEventListener('touchend', () => { dragging = false; wrap.classList.remove('cs-dragging'); });

    if (thumb) thumb.addEventListener('keydown', e => {
      const mn = parseFloat(sl.min), mx = parseFloat(sl.max), st = parseFloat(sl.step) || 1;
      let v = parseFloat(sl.value);
      if      (e.key === 'ArrowRight' || e.key === 'ArrowUp')   v += st;
      else if (e.key === 'ArrowLeft'  || e.key === 'ArrowDown') v -= st;
      else if (e.key === 'Home') v = mn;
      else if (e.key === 'End')  v = mx;
      else return;
      e.preventDefault();
      applyVal(Math.min(mx, Math.max(mn, v)));
    });
  });
}
_initCustomSliders();

SLIDER_IDS.forEach(id => {
  const sl = document.getElementById(id + '_slider');
  const inp = document.getElementById(id + '_input');
  if (sl) sl.addEventListener('input', () => syncFromSlider(id));
  if (inp) inp.addEventListener('change', () => syncFromInput(id));
});

// Optical switch shows/hides t_night
document.getElementById('optical-switch').addEventListener('change', function() {
  document.getElementById('tnight-block').style.display = this.checked ? 'block' : 'none';
  updateSubnightLimitDisplay();
  _checkPresetDrift();
  triggerUpdate();
});

// Other toggles
['toh-approx-switch','regime-color-switch','full-integral-switch','off-axis-switch'].forEach(id => {
  document.getElementById(id).addEventListener('change', triggerUpdate);
});

// Derived-info display wiring (instant, no Python)
document.getElementById('omegaexp_slider').addEventListener('input', updateNexpMaxDisplay);
document.getElementById('omegaexp_input').addEventListener('change', updateNexpMaxDisplay);
document.getElementById('omega_srv_slider').addEventListener('input', updateNexpMaxDisplay);
document.getElementById('omega_srv_input').addEventListener('change', updateNexpMaxDisplay);
document.getElementById('i_slider').addEventListener('input', updateSubnightLimitDisplay);
document.getElementById('i_input').addEventListener('change', updateSubnightLimitDisplay);
document.getElementById('tnight_slider').addEventListener('input', updateSubnightLimitDisplay);
document.getElementById('tnight_input').addEventListener('change', updateSubnightLimitDisplay);

// Slice-position sliders: live label + debounced partial recompute
function triggerSliceUpdate(which) {
  clearTimeout(_sliceDebounceTimer);
  _sliceDebounceTimer = setTimeout(() => runSliceUpdate(which), 120);
}
async function runSliceUpdate(which) {
  if (!_lastData) return;  // nothing to render against
  const fn = which === 'nslice' ? pyComputeNslice : pyComputeTslice;
  if (!fn) return;
  if (_sliceComputing[which]) { _sliceQueued[which] = true; return; }
  _sliceComputing[which] = true;
  _sliceQueued[which] = false;
  try {
    const params = readParams();
    const pyParams = pyodide.toPy(params);
    let pyResult;
    if (which === 'nslice') {
      const t_cad_fix_s = Math.pow(10, parseFloat(document.getElementById('nslice-tfix-slider').value));
      pyResult = fn(pyParams, t_cad_fix_s);
    } else {
      const N_fix = Math.pow(10, parseFloat(document.getElementById('tslice-nfix-slider').value));
      pyResult = fn(pyParams, N_fix);
    }
    const payload = pyResult.toJs({ dict_converter: Object.fromEntries });
    pyResult.destroy();
    pyParams.destroy();
    if (payload.error) {
      setStatus('Slice error: ' + String(payload.error).slice(0, 200));
    } else {
      // Merge slice payload into _lastData so renderers use fresh values.
      Object.assign(_lastData, payload);
      if (which === 'nslice' && _currentTab === 'nslice') renderNSlice(_lastData);
      if (which === 'tslice' && _currentTab === 'tslice') renderTSlice(_lastData);
    }
  } catch (e) {
    setStatus('Slice compute failed: ' + e.message);
    console.error(e);
  }
  _sliceComputing[which] = false;
  if (_sliceQueued[which]) { _sliceQueued[which] = false; setTimeout(() => runSliceUpdate(which), 0); }
}

document.getElementById('nslice-tfix-slider').addEventListener('input', function() {
  updateNsliceTfixDisplay();
  triggerSliceUpdate('nslice');
});
document.getElementById('tslice-nfix-slider').addEventListener('input', function() {
  updateTsliceNfixDisplay();
  triggerSliceUpdate('tslice');
});

// ── Preset loader ──────────────────────────────────────────────────────────
// Matches callbacks/ui.py: only 7 controls are touched.
function _setSliderValue(id, val) {
  const sl  = document.getElementById(id);
  const inp = document.getElementById(id.replace(/_slider$/, '_input'));
  if (sl)  sl.value  = val;
  if (inp) inp.value = val;
  if (sl) updateSliderVisual(sl);
}
document.getElementById('preset-select').addEventListener('change', function() {
  const key = this.value;
  if (!key || !PRESETS[key]) return;
  const p = PRESETS[key];
  _presetApplying = true;
  for (const [k, domId] of Object.entries(PRESET_MAP)) {
    _setSliderValue(domId, p[k]);
  }
  // Optical toggle: drives t_night visibility
  const opticalSwitch = document.getElementById('optical-switch');
  const newOptical = !!p.optical;
  if (opticalSwitch.checked !== newOptical) {
    opticalSwitch.checked = newOptical;
    document.getElementById('tnight-block').style.display = newOptical ? 'block' : 'none';
  }
  _activePresetKey = key;
  _presetApplying = false;
  updateGrbCounts();
  triggerUpdate();
});

// Drift detection: if the user edits any preset-controlled value, clear the preset dropdown.
function _checkPresetDrift() {
  if (_presetApplying || _activePresetKey === null) return;
  const p = PRESETS[_activePresetKey];
  if (!p) return;
  const sel = document.getElementById('preset-select');
  const opticalSwitch = document.getElementById('optical-switch');
  for (const [k, domId] of Object.entries(PRESET_MAP)) {
    const cur = parseFloat(document.getElementById(domId).value);
    if (Math.abs(cur - p[k]) > 1e-9) {
      _activePresetKey = null;
      sel.value = 'none';
      return;
    }
  }
  if (opticalSwitch.checked !== !!p.optical) {
    _activePresetKey = null;
    sel.value = 'none';
  }
}

// ── Read parameters ────────────────────────────────────────────────────────
function readParams() {
  const v = id => parseFloat(document.getElementById(id).value);
  const b = id => document.getElementById(id).checked;
  return {
    i_det:           Math.round(v('i_slider')),
    A_log:           v('Alog_slider'),
    f_live:          v('flive_slider'),
    t_overhead_s:    v('toh_slider'),
    omega_exp_deg2:  v('omegaexp_slider'),
    omega_srv_deg2:  v('omega_srv_slider'),
    t_night_h:       v('tnight_slider'),
    p:               v('p_slider'),
    nu_log10:        v('nu_log_slider'),
    E_kiso_log10:    v('Ekiso_log_slider'),
    n0_log10:        v('n0_log_slider'),
    epsilon_e_log10: v('epse_slider'),
    epsilon_B_log10: v('epsB_slider'),
    theta_j_rad:     v('thetaj_slider'),
    gamma0_log10:    v('gamma0_log_slider'),
    D_euc_gpc:       v('deuc_slider'),
    rho_grb_log10:   v('rho_grb_log_slider'),
    optical_survey:  b('optical-switch'),
    color_regimes:   b('regime-color-switch'),
    full_integral:   b('full-integral-switch'),
    off_axis:        b('off-axis-switch'),
    toh_approx:      b('toh-approx-switch'),
    nslice_tfix_log: v('nslice-tfix-slider'),
    tslice_nfix_log: v('tslice-nfix-slider'),
    nx: 120, ny: 150,
  };
}

// ── Slice-position helpers ────────────────────────────────────────────────
function _fmtTcad(t_cad_s) {
  if (!isFinite(t_cad_s) || t_cad_s <= 0) return '—';
  if (t_cad_s < 60)        return t_cad_s.toFixed(0) + ' s';
  if (t_cad_s < 3600)      return (t_cad_s / 60).toFixed(1) + ' min';
  if (t_cad_s < 86400)     return (t_cad_s / 3600).toFixed(2) + ' hr';
  if (t_cad_s < 86400 * 7) return (t_cad_s / 86400).toFixed(2) + ' day';
  if (t_cad_s < 86400 * 30)return (t_cad_s / (86400 * 7)).toFixed(2) + ' wk';
  if (t_cad_s < 86400 * 365)return (t_cad_s / (86400 * 30)).toFixed(2) + ' mo';
  return (t_cad_s / (86400 * 365.25)).toFixed(2) + ' yr';
}
function _fmtNexp(n) {
  if (!isFinite(n) || n <= 0) return '—';
  if (n < 1)    return n.toFixed(2);
  if (n < 10)   return n.toFixed(1);
  if (n < 1000) return Math.round(n).toString();
  if (n < 1e6)  return (n / 1000).toFixed(n < 1e4 ? 2 : 1) + 'k';
  return (n / 1e6).toFixed(1) + 'M';
}

function updateNsliceTfixDisplay() {
  const sl = document.getElementById('nslice-tfix-slider');
  const el = document.getElementById('nslice-tfix-value');
  if (sl && el) el.textContent = _fmtTcad(Math.pow(10, parseFloat(sl.value)));
}
function updateTsliceNfixDisplay() {
  const sl = document.getElementById('tslice-nfix-slider');
  const el = document.getElementById('tslice-nfix-value');
  if (sl && el) el.textContent = _fmtNexp(Math.pow(10, parseFloat(sl.value)));
}

// ── Derived sidebar displays (instant, no Python needed) ──────────────────
function updateNexpMaxDisplay() {
  const el = document.getElementById('nexpmax-display');
  if (!el) return;
  const oe = parseFloat(document.getElementById('omegaexp_slider').value);
  const os = parseFloat(document.getElementById('omega_srv_slider').value);
  if (!isFinite(oe) || !isFinite(os)) { el.innerHTML = ''; return; }
  if (oe > os) {
    el.innerHTML = '<span class="derived-info warning">⚠ Ω<sub>exp</sub> &gt; Ω<sub>srv,max</sub> — survey impossible</span>';
    return;
  }
  const nmax = Math.floor(os / oe);
  el.innerHTML = '<span class="derived-info">Max N<sub>exp</sub>: ' + nmax + '</span>';
}
function updateSubnightLimitDisplay() {
  const el = document.getElementById('subnight-limit-display');
  if (!el) return;
  const optical = document.getElementById('optical-switch').checked;
  const i_val   = parseFloat(document.getElementById('i_slider').value);
  const tnight  = parseFloat(document.getElementById('tnight_slider').value);
  if (!optical || !isFinite(i_val) || !isFinite(tnight) || i_val <= 0) { el.innerHTML = ''; return; }
  const limit_h = tnight / i_val;
  el.innerHTML = '<span class="derived-info">Sub-night limit: '
               + limit_h.toFixed(2) + ' hr  (t<sub>night</sub> / i)</span>';
}

// ── Derived GRB count display (instant, no Python needed) ─────────────────
function updateGrbCounts() {
  const rho = Math.pow(10, parseFloat(document.getElementById('rho_grb_log_slider').value));
  const D   = parseFloat(document.getElementById('deuc_slider').value);
  const tj  = parseFloat(document.getElementById('thetaj_slider').value);
  const V   = (4/3) * Math.PI * D * D * D;
  const N_total = rho * V;
  const fb  = tj * tj / 2;
  const N_toward_day = N_total * fb / 365.25;
  const fmt = x => x >= 1e6 ? (x/1e6).toFixed(1)+'M' : x >= 1e3 ? (x/1e3).toFixed(1)+'k' : x.toFixed(1);
  document.getElementById('grb-ntotal-display').innerHTML = 'R<sub>int</sub> = ' + fmt(N_total) + ' yr⁻¹';
  document.getElementById('grb-ntoward-display').innerHTML = 'f<sub>b</sub>R<sub>int</sub> = ' + fmt(N_toward_day) + ' day⁻¹';
}

// ── Debounced update trigger ───────────────────────────────────────────────
function triggerUpdate() {
  if (!pyComputeAll) return;
  clearTimeout(_debounceTimer);
  _debounceTimer = setTimeout(runUpdate, 300);
}

// ── Main async compute + render ───────────────────────────────────────────
async function runUpdate() {
  if (_computing) { _pendingUpdate = true; return; }
  if (!pyComputeAll) return;

  const params = readParams();
  _computing = true;
  _pendingUpdate = false;
  setStatus('Computing…', true);

  let data;
  try {
    const pyParams = pyodide.toPy(params);
    const pyResult = pyComputeAll(pyParams);
    data = pyResult.toJs({ dict_converter: Object.fromEntries });
    pyResult.destroy();
    pyParams.destroy();
  } catch (e) {
    setStatus('Error: ' + e.message);
    _computing = false;
    return;
  }

  if (data.error) {
    setStatus('Python error: ' + String(data.error).slice(0, 300));
    _computing = false;
    return;
  }

  _lastData = data;
  // Shared reshape layer — compute once, reuse across 3D + slices + metrics.
  data._shared = buildSharedData(data);

  render3DSurface(data, params);
  updateMetricsBar(data);
  updateDerivedDisplays(data);

  if (_currentTab === 'nslice') renderNSlice(data);
  else if (_currentTab === 'tslice') renderTSlice(data);

  const rStr = data.R_opt != null ? data.R_opt.toFixed(3) + ' yr⁻¹' : '—';
  const tStr = data.t_cad_opt_h != null ? fmtT(data.t_cad_opt_s) : '—';
  setStatus('Done · Opt: R=' + rStr + ', t_cad=' + tStr);

  _computing = false;
  if (_pendingUpdate) { _pendingUpdate = false; setTimeout(runUpdate, 0); }
}

// ── Reshape flat array → 2D ────────────────────────────────────────────────
function reshape(flat, rows, cols) {
  const out = [];
  for (let r = 0; r < rows; r++) out.push(flat.slice(r * cols, (r + 1) * cols));
  return out;
}

// ── Hex → rgba(…) for regime-coloured overlays (matches _hex_to_rgba) ──────
function hexToRgba(hex, alpha) {
  if (!hex || typeof hex !== 'string') return 'rgba(0,0,0,' + alpha + ')';
  const h = hex.charAt(0) === '#' ? hex.substring(1) : hex;
  if (h.length !== 6) return 'rgba(0,0,0,' + alpha + ')';
  const r = parseInt(h.substring(0, 2), 16);
  const g = parseInt(h.substring(2, 4), 16);
  const b = parseInt(h.substring(4, 6), 16);
  return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
}

// ── Shared data reshape layer ──────────────────────────────────────────────
// Called once per compute; shapes flat bridge arrays into the 2D grids used
// by the 3D figure AND (for Chunk 3) the slice figures. Fields:
//   shape            : [ny, nx]
//   X2d, Y2d, Z2d    : (ny, nx) grids of N_exp, t_cad_h, R_det
//   Zlog2d           : (ny, nx) log10 R_det (for surfacecolor)
//   regime2d         : (ny, nx) regime id ∈ {1..7} or null (or null if absent)
//   tExp2d           : (ny, nx) t_exp_s per point
//   qMed2d           : (ny, nx) q_med per point
//   dMed2d           : (ny, nx) D_med_Gpc per point
//   surfCustomdata3D : (nx, ny, 3) — Plotly 3D surface customdata order
//   dayLine          : null when optical=false, else:
//       {days:[…], n_days, n_N,
//        N2d, R2d, regime2d, tExp2d, qMed2d, dMed2d   — each (n_days, n_N)}
function buildSharedData(data) {
  if (!data || !data.shape) return null;
  const [ny, nx] = data.shape;
  const shared = {shape: [ny, nx], ny: ny, nx: nx};

  shared.X2d    = reshape(data.X_flat,    ny, nx);
  shared.Y2d    = reshape(data.Y_flat,    ny, nx);
  shared.Z2d    = reshape(data.Z_flat,    ny, nx);
  shared.Zlog2d = reshape(data.Z_log_flat, ny, nx);
  shared.regime2d = data.regime_flat ? reshape(data.regime_flat, ny, nx) : null;
  shared.tExp2d = data.t_exp_flat    ? reshape(data.t_exp_flat,    ny, nx) : null;
  shared.qMed2d = data.q_med_flat    ? reshape(data.q_med_flat,    ny, nx) : null;
  shared.dMed2d = data.D_med_Gpc_flat ? reshape(data.D_med_Gpc_flat, ny, nx) : null;

  // Plotly's 3D Surface customdata is indexed [j, i, k] (transposed from the
  // (ny, nx, 3) per-cell tensor). Build (nx, ny, 3).
  if (shared.tExp2d && shared.qMed2d && shared.dMed2d) {
    const cd = new Array(nx);
    for (let j = 0; j < nx; j++) {
      const col = new Array(ny);
      for (let i = 0; i < ny; i++) {
        col[i] = [shared.tExp2d[i][j], shared.qMed2d[i][j], shared.dMed2d[i][j]];
      }
      cd[j] = col;
    }
    shared.surfCustomdata3D = cd;
  } else {
    shared.surfCustomdata3D = null;
  }

  // Day-line payload (optical-only). Bridge returns empty lists when optical
  // is off; day_line_shape = [0, 0] in that case.
  const dlShape = data.day_line_shape;
  if (dlShape && dlShape.length === 2 && dlShape[0] > 0 && dlShape[1] > 0) {
    const nd = dlShape[0], nn = dlShape[1];
    shared.dayLine = {
      days:     data.day_line_t_cad_days || [],
      n_days:   nd,
      n_N:      nn,
      N2d:      reshape(data.day_line_N_flat,         nd, nn),
      R2d:      reshape(data.day_line_R_flat,         nd, nn),
      regime2d: data.day_line_regime_flat ? reshape(data.day_line_regime_flat, nd, nn) : null,
      tExp2d:   reshape(data.day_line_t_exp_flat,     nd, nn),
      qMed2d:   reshape(data.day_line_q_med_flat,     nd, nn),
      dMed2d:   reshape(data.day_line_D_med_Gpc_flat, nd, nn),
    };
  } else {
    shared.dayLine = null;
  }
  return shared;
}

// ── Standardized hovertemplates (ported verbatim from components/figures.py)
const XYZ_HOVER =
  'N<sub>exp</sub> = %{x:.4g}<br>' +
  't<sub>cad</sub> = %{y:.4g} hr<br>' +
  't<sub>exp</sub> = %{customdata[0]:.3g} s<br>' +
  'q<sub>med</sub> = %{customdata[1]:.3g}<br>' +
  'D<sub>med</sub> = %{customdata[2]:.3g} Gpc<br>' +
  'R<sub>det</sub> = %{z:.4g} yr ⁻¹' +
  '<extra></extra>';

// N-slice: N_exp=x, R_det=y; customdata=[t_cad_hr, t_exp, q_med, D_med_Gpc]
const N_HOVER =
  'N<sub>exp</sub> = %{x:.4g}<br>' +
  't<sub>cad</sub> = %{customdata[0]:.4g} hr<br>' +
  't<sub>exp</sub> = %{customdata[1]:.3g} s<br>' +
  'q<sub>med</sub> = %{customdata[2]:.3g}<br>' +
  'D<sub>med</sub> = %{customdata[3]:.3g} Gpc<br>' +
  'R<sub>det</sub> = %{y:.4g} yr ⁻¹' +
  '<extra></extra>';

// T-slice: t_cad=x, R_det=y; customdata=[N_exp, t_exp, q_med, D_med_Gpc]
const T_HOVER =
  'N<sub>exp</sub> = %{customdata[0]:.4g}<br>' +
  't<sub>cad</sub> = %{x:.4g} hr<br>' +
  't<sub>exp</sub> = %{customdata[1]:.3g} s<br>' +
  'q<sub>med</sub> = %{customdata[2]:.3g}<br>' +
  'D<sub>med</sub> = %{customdata[3]:.3g} Gpc<br>' +
  'R<sub>det</sub> = %{y:.4g} yr ⁻¹' +
  '<extra></extra>';

// ── Regime-segmentation helper for 2D slice line traces ────────────────────
// Mirrors components/figures.py::_draw_regime_segments_2d. Walks x/y/regime
// arrays in lockstep, emitting one Plotly scatter trace per contiguous run of
// equal regime IDs; optionally bridges to the first point of the next segment
// so neighbouring segments visually connect. `customdata` is a parallel array
// of per-point hover payloads (length == xArr.length).
function segmentsByRegime(xArr, yArr, regimeArr, customdata, opts) {
  opts = opts || {};
  const lineWidth = opts.lineWidth != null ? opts.lineWidth : 2.5;
  const opacity   = opts.opacity   != null ? opts.opacity   : 1.0;
  const hovertemplate = opts.hovertemplate || null;
  const hoverlabel    = opts.hoverlabel || null;
  const n = xArr.length;
  const out = [];
  let i = 0;
  while (i < n) {
    const xi = xArr[i], yi = yArr[i], ri = regimeArr[i];
    const finite = (v) => v != null && isFinite(v);
    if (!(finite(xi) && finite(yi) && finite(ri))) { i++; continue; }
    const cur = ri | 0;
    let j = i;
    while (j < n && finite(xArr[j]) && finite(yArr[j]) && finite(regimeArr[j]) && (regimeArr[j] | 0) === cur) j++;
    const segX = xArr.slice(i, j);
    const segY = yArr.slice(i, j);
    const bridged = j < n && finite(xArr[j]) && finite(yArr[j]);
    if (bridged) { segX.push(xArr[j]); segY.push(yArr[j]); }
    const col = (cur >= 1 && cur <= 7)
      ? hexToRgba(REGIME_HEX[cur - 1], opacity)
      : ('rgba(128,128,128,' + opacity + ')');
    const trace = {
      type: 'scatter',
      x: segX, y: segY,
      mode: 'lines',
      line: {color: col, width: lineWidth},
      showlegend: false,
    };
    if (customdata) {
      let segCd = customdata.slice(i, j);
      if (bridged) segCd = segCd.concat([customdata[j]]);
      trace.customdata = segCd;
    }
    if (hovertemplate) {
      trace.hovertemplate = hovertemplate;
      if (hoverlabel) trace.hoverlabel = hoverlabel;
    } else {
      trace.hoverinfo = 'skip';
    }
    out.push(trace);
    i = j;
  }
  return out;
}

// ── Regime legend (2D phantom traces) ──────────────────────────────────────
function regimeLegendTraces2D(colorOn) {
  if (!colorOn) return [];
  return REGIME_HEX.map((col, idx) => ({
    type: 'scatter',
    x: [null], y: [null],
    mode: 'markers',
    marker: {size: 8, color: col, symbol: 'square'},
    name: REGIME_LABELS[idx],
    showlegend: true,
    hoverinfo: 'skip',
  }));
}

// Match Dash's _marker_hover: only emit the optional lines when the scalar is finite.
function markerHover3D(label, t_exp_s, q_med, D_med_Gpc) {
  const okTexp = t_exp_s   != null && isFinite(t_exp_s);
  const okQ    = q_med     != null && isFinite(q_med);
  const okD    = D_med_Gpc != null && isFinite(D_med_Gpc);
  const toG = (v) => {
    // Python "%.3g" → JS toPrecision(3) with trailing-zero trim is close enough; the
    // template actually formats at display time via Plotly, but here we need the number
    // inline because these are scalar markers, not per-point customdata. Use Number.
    const s = Number(v).toPrecision(3);
    // Strip trailing zeros after decimal point, and trailing '.' if it ends with one.
    return s.indexOf('.') >= 0 && s.indexOf('e') < 0 ? s.replace(/\.?0+$/, '') : s;
  };
  const texpStr = okTexp ? ('<br>t<sub>exp</sub> = ' + toG(t_exp_s) + ' s') : '';
  const qStr    = okQ    ? ('<br>q<sub>med</sub> = ' + toG(q_med)) : '';
  const dStr    = okD    ? ('<br>D<sub>med</sub> = ' + toG(D_med_Gpc) + ' Gpc') : '';
  return (
    label +
    '<br>N<sub>exp</sub> = %{x:.4g}' +
    '<br>t<sub>cad</sub> = %{y:.4g} hr' +
    texpStr + qStr + dStr +
    '<br>R<sub>det</sub> = %{z:.4g} yr ⁻¹<extra></extra>'
  );
}

// ── Format helpers ─────────────────────────────────────────────────────────
function fmtR(r) {
  if (r == null || !isFinite(r)) return '—';
  if (r >= 100) return r.toFixed(0);
  if (r >= 10) return r.toFixed(1);
  return r.toFixed(2);
}
// Emulate Python's `%.2g`: 2 significant figures, scientific notation when
// exponent < -4 or >= 2 (matching CPython), otherwise fixed; trailing zeros
// are stripped. Exponent is zero-padded to at least 2 digits to match Python.
function fmt2g(x) {
  if (x == null || !isFinite(x)) return '—';
  if (x === 0) return '0';
  const abs = Math.abs(x);
  let e = Math.floor(Math.log10(abs));
  let scientific = (e < -4 || e >= 2);
  if (!scientific) {
    // Fixed: (precision - 1 - e) decimals, clamped at 0.
    const decimals = Math.max(0, 1 - e);
    let s = x.toFixed(decimals);
    // Rounding may have pushed the value into the scientific regime (e.g.
    // 99.9 → "100" → exp 2). If so, re-emit as scientific.
    const rounded = parseFloat(s);
    if (rounded !== 0) {
      const e2 = Math.floor(Math.log10(Math.abs(rounded)));
      if (e2 >= 2) {
        scientific = true;
        e = e2;
      }
    }
    if (!scientific) {
      if (s.indexOf('.') >= 0) s = s.replace(/\.?0+$/, '');
      return s;
    }
  }
  // Scientific: mantissa rounded to 1 decimal (precision - 1 = 1).
  let mantissa = x / Math.pow(10, e);
  let mStr = mantissa.toFixed(1);
  let mNum = parseFloat(mStr);
  let finalExp = e;
  if (Math.abs(mNum) >= 10) {
    mNum = mNum / 10;
    finalExp = e + 1;
    mStr = mNum.toFixed(1);
  }
  if (mStr.indexOf('.') >= 0) mStr = mStr.replace(/\.?0+$/, '');
  const sign = finalExp >= 0 ? '+' : '-';
  const expAbs = Math.abs(finalExp);
  const expStr = (expAbs < 10 ? '0' : '') + expAbs;
  return mStr + 'e' + sign + expStr;
}
function fmtT(t_s) {
  if (t_s == null || !isFinite(t_s)) return '—';
  if (t_s >= DAY_S) return fmt2g(t_s / DAY_S) + ' day';
  if (t_s >= 3600) return fmt2g(t_s / 3600) + ' hr';
  if (t_s >= 60) return fmt2g(t_s / 60) + ' min';
  return fmt2g(t_s) + ' sec';
}
function fmtN(n) { return n == null || !isFinite(n) ? '—' : n.toFixed(0); }

// ── Layout helpers ─────────────────────────────────────────────────────────
function darkMode() { return _currentTheme !== 'light'; }
function gridCol() { return darkMode() ? 'rgba(255,255,255,0.10)' : 'rgba(0,0,0,0.10)'; }
function gridColLight() { return darkMode() ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)'; }
function fontCol() { return darkMode() ? '#e2e8f0' : '#1e293b'; }
function hoverBg() { return darkMode() ? '#1e2533' : '#e8f0f8'; }
function hoverFontCol() { return darkMode() ? '#e2e8f0' : '#1e293b'; }
function plotBg() { return 'rgba(0,0,0,0)'; }
function annotCol() { return darkMode() ? '#8ba0c0' : '#4b6080'; }

// ── Discrete-day overlay lines (3D, optical mode only) ────────────────────
// Mirrors components/figures.py::_add_discrete_day_lines. Days with no valid
// (unmasked) points are skipped. In regime-colour mode each consecutive run
// of equal regime IDs becomes its own segment; in height-colour mode a
// single light uniform line is drawn per day.
function addDay3DLines(traces, shared, params, zmaxLog) {
  const dl = shared && shared.dayLine;
  if (!dl || dl.n_days === 0) return;
  const regimeMode = !!params.color_regimes;
  const heightCmax = (zmaxLog != null && isFinite(zmaxLog)) ? zmaxLog : 1.0;

  // Height-colour mode: collect marker traces separately so they are appended
  // AFTER every per-day line trace. Matches components/figures.py:L319-L337 —
  // WebGL depth-sort ties break in favour of later draw calls, so deferring the
  // markers keeps dots painted above the faint connecting lines regardless of
  // camera angle.
  const pendingMarkers = [];

  for (let d = 0; d < dl.n_days; d++) {
    const Nrow   = dl.N2d[d];
    const Rrow   = dl.R2d[d];
    const tErow  = dl.tExp2d[d];
    const qErow  = dl.qMed2d[d];
    const dErow  = dl.dMed2d[d];
    const ridRow = dl.regime2d ? dl.regime2d[d] : null;

    // Select "good" points: finite N and finite R (bridge set NaN→null for masked cells).
    const good = [];
    for (let j = 0; j < dl.n_N; j++) {
      if (Nrow[j] != null && isFinite(Nrow[j]) && Rrow[j] != null && isFinite(Rrow[j])) good.push(j);
    }
    if (good.length < 2) continue;

    const yHr = dl.days[d] * 24.0;  // constant y for this day (t_cad in hours)
    const xG   = good.map(j => Nrow[j]);
    const yG   = good.map(_ => yHr);
    const zG   = good.map(j => Rrow[j]);
    const cdG  = good.map(j => [tErow[j], qErow[j], dErow[j]]);

    if (regimeMode && ridRow) {
      const ridG = good.map(j => ridRow[j]);
      let start = 0;
      while (start < ridG.length) {
        if (ridG[start] == null || !isFinite(ridG[start])) { start++; continue; }
        const k = ridG[start] | 0;
        let end = start + 1;
        while (end < ridG.length && ridG[end] != null && isFinite(ridG[end]) && (ridG[end] | 0) === k) end++;
        if (end - start >= 2 && k >= 1 && k <= 7) {
          const col = hexToRgba(REGIME_HEX[k - 1], 0.80);
          traces.push({
            type: 'scatter3d',
            x: xG.slice(start, end),
            y: yG.slice(start, end),
            z: zG.slice(start, end),
            customdata: cdG.slice(start, end),
            mode: 'lines+markers',
            line: {color: col, width: 6},
            marker: {size: 0, opacity: 0.001},
            showlegend: false,
            hovertemplate: XYZ_HOVER,
          });
        }
        start = end;
      }
    } else {
      // Height-colour mode: uniform faint line NOW, deferred marker trace LATER.
      traces.push({
        type: 'scatter3d',
        x: xG, y: yG, z: zG,
        customdata: cdG,
        mode: 'lines',
        line: {color: 'rgba(255,255,255,0.12)', width: 1.0},
        showlegend: false,
        hovertemplate: XYZ_HOVER,
      });
      pendingMarkers.push({
        type: 'scatter3d',
        x: xG, y: yG, z: zG,
        customdata: cdG,
        mode: 'markers',
        marker: {
          size: 2,
          color: zG.map(v => Math.log10(Math.max(v, 1e-30))),
          colorscale: PLASMA_SCALE,
          cmin: -2.0,
          cmax: heightCmax,
          opacity: 1.0,
          showscale: false,
        },
        showlegend: false,
        hovertemplate: XYZ_HOVER,
      });
    }
  }

  for (let i = 0; i < pendingMarkers.length; i++) traces.push(pendingMarkers[i]);
}

// ── 3D Surface render ──────────────────────────────────────────────────────
function render3DSurface(data, params) {
  const shared = data._shared || (data._shared = buildSharedData(data));
  if (!shared) return;
  const {ny, nx, X2d, Y2d, Z2d, Zlog2d, regime2d, surfCustomdata3D} = shared;
  const traces = [];
  const dark = darkMode();

  // Discrete day-cadence overlay lines (optical only); Dash adds these BEFORE the
  // main surface so the surface z-order wins on overlapping regions.
  if (params.optical_survey) addDay3DLines(traces, shared, params, data.zmax_log10);

  // Optical mode: exclude discrete-day rows (y ≥ 24 h) from the surface — they
  // are rendered as 1D scatter3d lines by addDay3DLines instead.
  // Mirrors callbacks/surface.py:385-398 where Dash keeps only rows with y_s < DAY_S.
  let Z2dSurf = Z2d, Zlog2dSurf = Zlog2d, regime2dSurf = regime2d;
  if (params.optical_survey) {
    const nullRow = new Array(nx).fill(null);
    Z2dSurf      = Y2d.map((row, r) => (row[0] != null && row[0] >= 24.0) ? nullRow : Z2d[r]);
    Zlog2dSurf   = Y2d.map((row, r) => (row[0] != null && row[0] >= 24.0) ? nullRow : Zlog2d[r]);
    regime2dSurf = regime2d
      ? Y2d.map((row, r) => (row[0] != null && row[0] >= 24.0) ? nullRow : regime2d[r])
      : null;
  }

  if (params.color_regimes && regime2d) {
    for (let k = 1; k <= 7; k++) {
      const Zk = Z2dSurf.map((row, r) => row.map((v, c) => (regime2dSurf[r][c] === k ? v : null)));
      if (Zk.every(row => row.every(v => v === null))) continue;
      const col = REGIME_HEX[k - 1];
      const trace = {
        type: 'surface', x: X2d, y: Y2d, z: Zk,
        colorscale: [[0, col], [1, col]], cmin: 0, cmax: 1,
        showscale: false, connectgaps: false, showlegend: false,
        lighting: {ambient: 0.75, diffuse: 0.75, specular: 0.08, roughness: 0.95},
        lightposition: {x: 100, y: 200, z: 0},
        hovertemplate: XYZ_HOVER,
      };
      if (surfCustomdata3D) trace.customdata = surfCustomdata3D;
      traces.push(trace);
    }
    REGIME_HEX.forEach((col, i) => traces.push({
      type: 'scatter3d', x: [null], y: [null], z: [null],
      mode: 'markers', marker: {size: 7, color: col},
      name: REGIME_LABELS[i], showlegend: true, hoverinfo: 'skip',
    }));
  } else {
    const zmax = data.zmax_log10 || 0;
    const trace = {
      type: 'surface', x: X2d, y: Y2d, z: Z2dSurf,
      surfacecolor: Zlog2dSurf,
      cmin: ZMIN_LOG, cmax: zmax,
      colorscale: PLASMA_SCALE, showscale: true, connectgaps: false,
      lighting: {ambient: 0.75, diffuse: 0.75, specular: 0.08, roughness: 0.95},
      lightposition: {x: 100, y: 200, z: 0},
      colorbar: {
        title: {text: 'R<sub>det</sub> [yr ⁻¹]', side: 'top'},
        x: 0.88, xanchor: 'left', thickness: 18, len: 0.80,
        y: 0.5, yanchor: 'middle',
        tickvals: [-2, -1, 0, 1, 2, 3, 4],
        ticktext: ['0.01', '0.1', '1', '10', '100', '1k', '10k'],
      },
      hovertemplate: XYZ_HOVER,
    };
    if (surfCustomdata3D) trace.customdata = surfCustomdata3D;
    traces.push(trace);
  }

  // Optimal marker
  if (data.N_opt != null && data.R_opt != null && isFinite(data.R_opt)) {
    traces.push({
      type: 'scatter3d',
      x: [data.N_opt], y: [data.t_cad_opt_h], z: [data.R_opt],
      mode: 'markers+text',
      marker: {size: 10, color: AMBER, symbol: 'diamond'},
      text: ['Optimum'], textposition: 'top center',
      name: 'Optimum',
      hovertemplate: markerHover3D('Grid optimum', data.t_exp_opt_s, data.q_med_opt, data.D_med_Gpc_opt),
    });
  }

  // ZTF marker
  if (data.R_ztf != null && isFinite(data.R_ztf)) {
    traces.push({
      type: 'scatter3d',
      x: [data.N_ztf], y: [data.t_cad_ztf_h], z: [data.R_ztf],
      mode: 'markers+text',
      marker: {size: 9, color: CORAL, symbol: 'circle'},
      text: ['ZTF'], textposition: 'top center',
      name: 'ZTF (2 day cadence)',
      hovertemplate: markerHover3D('ZTF strategy', data.t_exp_ztf_s, data.q_med_ztf, data.D_med_Gpc_ztf),
    });
  }

  const Rmax = data.R_opt != null ? Math.max(data.R_opt, data.R_ztf || 0, 0.11) : 0.11;

  // Match Dash's 3D scene grid colour (slightly stronger than the 2D grids).
  const grid3d = dark ? 'rgba(255,255,255,0.14)' : 'rgba(0,0,0,0.12)';

  const layout = {
    uirevision: 'keep-view-v1',
    template: dark ? 'plotly_dark' : 'plotly_white',
    paper_bgcolor: plotBg(),
    hoverlabel: {bgcolor: hoverBg(), font: {color: hoverFontCol()}, bordercolor: 'rgba(0,0,0,0)'},
    scene: {
      bgcolor: plotBg(),
      xaxis: {title: 'N<sub>exp</sub>', type: 'log', gridcolor: grid3d, showbackground: false},
      yaxis: {
        title: 't<sub>cad</sub>', type: 'log',
        tickvals: TCAD_TICKVALS_H, ticktext: TCAD_TICKTEXT,
        gridcolor: grid3d, showbackground: false,
      },
      zaxis: {
        title: 'R<sub>det</sub> [yr ⁻¹]', type: 'log',
        range: [ZMIN_LOG, Math.log10(Math.max(Rmax, 0.11)) + 0.05],
        gridcolor: grid3d, showbackground: false,
      },
      aspectmode: 'manual', aspectratio: {x: 1.2, y: 1.2, z: 0.9},
      // Plotly.js uses scene.camera (not top-level scene_camera which is a Plotly.py convention).
      // Must live inside `scene` for the initial view to actually apply.
      camera: {eye: {x: -1.48, y: -1.48, z: 0.70}, up: {x: 0, y: 0, z: 1}, center: {x: 0, y: 0, z: -0.2}},
    },
    margin: {l: 0, r: 0, b: 0, t: 0},
    legend: {orientation: 'v', x: 0.01, y: 0.98, xanchor: 'left', yanchor: 'top', font: {size: 11}, bgcolor: 'rgba(0,0,0,0)'},
    font: {family: "'JetBrains Mono','Cascadia Code',monospace", color: fontCol(), size: 12},
  };

  Plotly.react('plot-3d', traces, layout, {responsive: true});
}

// ── N-slice render (R vs N_exp at user-chosen t_cad) ───────────────────────
// Ports components/figures.py::build_nslice_figure. Slice position is driven
// by the nslice-tfix-slider (payload t_cad_fix_h), so the render no longer
// early-returns when the optimizer fails. The amber-diamond marker tracks the
// optimizer's N_opt on the CURRENT curve (R value read at argmin|N_sweep - N_opt|);
// when the slice is not near the optimum t_cad, an annotation calls that out.
function renderNSlice(data) {
  // Slice position comes from the bridge payload (falls back to optimum on
  // first render if the slice-position slider has not yet been consulted).
  const tCadRow = (data.t_cad_fix_h != null && isFinite(data.t_cad_fix_h))
    ? data.t_cad_fix_h
    : (data.t_cad_opt_h != null && isFinite(data.t_cad_opt_h) ? data.t_cad_opt_h : null);

  // Chunk 7: consume dedicated high-res N-sweep (800 points, logspaced)
  // computed by the bridge. Mirrors callbacks/surface.py:441-451.
  const xArr  = data.N_sweep_flat          || [];
  const zArr  = data.N_sweep_R_flat        || [];
  const teArr = data.N_sweep_t_exp_flat    || [];
  const qmArr = data.N_sweep_q_med_flat    || [];
  const dmArr = data.N_sweep_D_med_Gpc_flat|| [];
  const ridArr= data.N_sweep_regime_flat   || [];
  const nSweep = xArr.length;

  // Build parallel valid arrays (drop null/NaN R values)
  const xv = [], zv = [], ridV = [], cdV = [];
  for (let c = 0; c < nSweep; c++) {
    const x = xArr[c], z = zArr[c];
    if (x == null || !isFinite(x) || z == null || !isFinite(z) || !(z > 0)) continue;
    xv.push(x);
    zv.push(z);
    ridV.push(ridArr.length ? ridArr[c] : null);
    cdV.push([
      tCadRow,
      teArr.length ? teArr[c] : null,
      qmArr.length ? qmArr[c] : null,
      dmArr.length ? dmArr[c] : null,
    ]);
  }

  const dark = darkMode();
  const accent = dark ? '#6d9eff' : '#3b6fff';
  const hl = {bgcolor: hoverBg(), font: {color: hoverFontCol()}, bordercolor: 'rgba(0,0,0,0)'};
  const traces = [];

  // Regime legend first (so labels sort to top)
  regimeLegendTraces2D(!!data.color_regimes).forEach(t => traces.push(t));

  if (xv.length === 0) {
    Plotly.react('plot-nslice', traces, {
      template: dark ? 'plotly_dark' : 'plotly_white',
      paper_bgcolor: plotBg(), plot_bgcolor: plotBg(),
      annotations: [{text: 'No valid data in N-slice', xref: 'paper', yref: 'paper',
                     x: 0.5, y: 0.5, showarrow: false,
                     font: {size: 14, color: annotCol()}}],
    }, {responsive: true});
    return;
  }

  // Main rate curve — regime-coloured segments or single accent line
  if (data.color_regimes && ridV.some(v => v != null && isFinite(v))) {
    segmentsByRegime(xv, zv, ridV, cdV, {
      lineWidth: 2.5, hovertemplate: N_HOVER, hoverlabel: hl,
    }).forEach(t => traces.push(t));
  } else {
    traces.push({
      type: 'scatter', x: xv, y: zv,
      customdata: cdV,
      mode: 'lines',
      line: {color: accent, width: 2.5},
      showlegend: false,
      hovertemplate: N_HOVER,
      hoverlabel: hl,
    });
  }

  // Optimal N_exp marker (amber diamond) — placed on the CURRENT curve at
  // argmin|N_sweep - N_opt|. Matches components/figures.py:L374-L392 (upstream).
  if (data.N_opt != null && isFinite(data.N_opt) && xv.length > 0) {
    let optIdx = 0, optDist = Infinity;
    for (let k = 0; k < xv.length; k++) {
      const d = Math.abs(xv[k] - data.N_opt);
      if (d < optDist) { optDist = d; optIdx = k; }
    }
    const R_at_N_opt = zv[optIdx];
    if (isFinite(R_at_N_opt) && R_at_N_opt > 0) {
      const cd = cdV[optIdx] || [tCadRow, null, null, null];
      traces.push({
        type: 'scatter', x: [data.N_opt], y: [R_at_N_opt],
        mode: 'markers',
        marker: {size: 12, color: AMBER, symbol: 'diamond', line: {width: 1.5, color: 'white'}},
        name: 'Opt. N<sub>exp</sub>',
        customdata: [cd],
        hovertemplate:
          'N<sub>exp,opt</sub> = %{x:.4g}<br>' +
          't<sub>cad</sub> = %{customdata[0]:.4g} hr<br>' +
          't<sub>exp</sub> = %{customdata[1]:.3g} s<br>' +
          'q<sub>med</sub> = %{customdata[2]:.3g}<br>' +
          'D<sub>med</sub> = %{customdata[3]:.3g} Gpc<br>' +
          'R<sub>det</sub> = %{y:.4g} yr ⁻¹' +
          '<extra>Opt. N<sub>exp</sub></extra>',
        hoverlabel: hl,
      });
    }
  }

  // Reference lines: ZTF horizontal (R_ztf) + ZTF vertical (N_ztf)
  const shapes = [];
  const annotations = [{
    text: 'N<sub>exp</sub> slice  |  t<sub>cad</sub> = ' +
          (tCadRow != null && isFinite(tCadRow) ? _fmtTcad(tCadRow * 3600) : '?'),
    xref: 'paper', yref: 'paper', x: 0.01, y: 1.0,
    showarrow: false, xanchor: 'left', yanchor: 'top',
    font: {size: 12, color: annotCol()},
  }];
  // Dashed `Opt. t_cad` callout when the slice is not near the optimum.
  if (data.t_cad_opt_h != null && isFinite(data.t_cad_opt_h) && tCadRow != null
      && Math.abs(tCadRow - data.t_cad_opt_h) / (data.t_cad_opt_h + 1e-30) > 0.02) {
    annotations.push({
      text: 'Opt. t<sub>cad</sub> = ' + _fmtTcad(data.t_cad_opt_h * 3600),
      xref: 'paper', yref: 'paper', x: 0.99, y: 0.98,
      showarrow: false, xanchor: 'right', yanchor: 'top',
      font: {size: 10, color: AMBER},
    });
  }
  if (data.R_ztf != null && isFinite(data.R_ztf)) {
    shapes.push({
      type: 'line', xref: 'paper', x0: 0, x1: 1,
      yref: 'y', y0: data.R_ztf, y1: data.R_ztf,
      line: {color: CORAL, width: 1.5, dash: 'dot'},
    });
    annotations.push({
      text: 'R<sub>ZTF</sub> = ' + (+data.R_ztf).toPrecision(2).replace(/\.?0+$/, '') + ' yr ⁻¹',
      xref: 'paper', yref: 'y', x: 0.98, y: data.R_ztf,
      xanchor: 'right', yanchor: 'bottom', showarrow: false,
      font: {size: 11, color: CORAL},
    });
  }
  if (data.N_ztf != null && isFinite(data.N_ztf)) {
    shapes.push({
      type: 'line', yref: 'paper', y0: 0, y1: 1,
      xref: 'x', x0: data.N_ztf, x1: data.N_ztf,
      line: {color: CORAL, width: 1.2, dash: 'dot'},
    });
    annotations.push({
      text: 'N<sub>ZTF</sub>=' + Math.round(data.N_ztf),
      xref: 'x', yref: 'paper', x: data.N_ztf, y: 0.04,
      xanchor: 'center', yanchor: 'bottom', showarrow: false,
      font: {size: 10, color: CORAL},
    });
  }

  const layout = {
    // Dash's build_nslice_figure intentionally omits uirevision so the 2D view
    // resets on every update (matches Plotly.py default behaviour).
    template: dark ? 'plotly_dark' : 'plotly_white',
    paper_bgcolor: plotBg(), plot_bgcolor: plotBg(),
    font: {family: "'JetBrains Mono','Cascadia Code',monospace", color: fontCol(), size: 12},
    margin: {l: 64, r: 24, b: 48, t: 40},
    xaxis: {title: 'N<sub>exp</sub>', type: 'log', showgrid: true, gridcolor: gridColLight()},
    yaxis: {title: 'R<sub>det</sub> [yr ⁻¹]', type: 'log', showgrid: true, gridcolor: gridColLight()},
    showlegend: true,
    legend: {x: 0.01, y: 0.98, xanchor: 'left', yanchor: 'top', bgcolor: 'rgba(0,0,0,0)', font: {size: 11}},
    annotations: annotations,
    shapes: shapes,
    hoverlabel: {bgcolor: hoverBg(), font: {color: hoverFontCol()}, bordercolor: 'rgba(0,0,0,0)'},
  };

  Plotly.react('plot-nslice', traces, layout, {responsive: true});
}

// ── T-slice render (R vs t_cad at optimal N_exp) ───────────────────────────
// Ports components/figures.py::build_tslice_figure. Builds two data regions:
// (1) a continuous sub-night sweep (t_cad < gap_lo_h) taken from the 2-D grid
//     column at N_opt, and (2) a discrete-day region (integer-day cadences)
//     extracted from the shared `dayLine` payload. In optical mode an amber
//     gap rectangle spans [gap_lo_h, gap_hi_h]. In regime-colour mode both
//     regions use `segmentsByRegime`, and discrete days get per-day markers
//     coloured by regime (matches Dash's overlay markers in figures.py).
function renderTSlice(data) {
  const opticalOn  = data.gap_lo_h != null && isFinite(data.gap_lo_h);
  const gapLo      = data.gap_lo_h;
  const gapHi      = data.gap_hi_h;
  // Slice position (N_exp) comes from the tslice-nfix-slider payload, with
  // optimum fallback for the initial render.
  const N_fix      = (data.N_fix != null && isFinite(data.N_fix))
    ? data.N_fix
    : (data.N_opt != null && isFinite(data.N_opt) ? data.N_opt : null);
  const N_opt_col  = N_fix;

  // Chunk 7: consume dedicated high-res t-slice sweeps (600 cont / 500 disc
  // for optical, or 1500 cont + 0 disc for non-optical) computed by the
  // bridge. Mirrors callbacks/surface.py:471-513.
  const tContArr = data.t_cont_h_flat         || [];
  const rContArr = data.t_cont_R_flat         || [];
  const tecContArr = data.t_cont_t_exp_flat   || [];
  const qmContArr  = data.t_cont_q_med_flat   || [];
  const dmContArr  = data.t_cont_D_med_Gpc_flat|| [];
  const ridContArr = data.t_cont_regime_flat  || [];

  // ── Continuous region ────────────────────────────────────────────────────
  const tCont = [], rCont = [], ridCont = [], cdCont = [];
  for (let i = 0; i < tContArr.length; i++) {
    const y = tContArr[i], z = rContArr[i];
    if (y == null || !isFinite(y) || z == null || !isFinite(z) || !(z > 0)) continue;
    tCont.push(y);
    rCont.push(z);
    ridCont.push(ridContArr.length ? ridContArr[i] : null);
    cdCont.push([
      N_opt_col,
      tecContArr.length ? tecContArr[i] : null,
      qmContArr.length  ? qmContArr[i]  : null,
      dmContArr.length  ? dmContArr[i]  : null,
    ]);
  }

  // ── Discrete-day region (optical only; empty otherwise) ──────────────────
  const tDiscArr   = data.t_disc_h_flat         || [];
  const rDiscArr   = data.t_disc_R_flat         || [];
  const tecDiscArr = data.t_disc_t_exp_flat     || [];
  const qmDiscArr  = data.t_disc_q_med_flat     || [];
  const dmDiscArr  = data.t_disc_D_med_Gpc_flat || [];
  const ridDiscArr = data.t_disc_regime_flat    || [];

  const tDisc = [], rDisc = [], ridDisc = [], cdDisc = [];
  for (let i = 0; i < tDiscArr.length; i++) {
    const y = tDiscArr[i], z = rDiscArr[i];
    if (y == null || !isFinite(y) || z == null || !isFinite(z) || !(z > 0)) continue;
    tDisc.push(y);
    rDisc.push(z);
    ridDisc.push(ridDiscArr.length ? ridDiscArr[i] : null);
    cdDisc.push([
      N_opt_col,
      tecDiscArr.length ? tecDiscArr[i] : null,
      qmDiscArr.length  ? qmDiscArr[i]  : null,
      dmDiscArr.length  ? dmDiscArr[i]  : null,
    ]);
  }

  const hasCont = tCont.length > 0;
  const hasDisc = tDisc.length > 0;

  const dark = darkMode();
  const accent = dark ? '#6d9eff' : '#3b6fff';
  const gapCol = dark ? 'rgba(255,200,100,0.07)' : 'rgba(200,140,0,0.07)';
  const hl = {bgcolor: hoverBg(), font: {color: hoverFontCol()}, bordercolor: 'rgba(0,0,0,0)'};
  const colorOn = !!data.color_regimes;

  const traces = [];
  regimeLegendTraces2D(colorOn).forEach(t => traces.push(t));

  if (!hasCont && !hasDisc) {
    Plotly.react('plot-tslice', traces, {
      template: dark ? 'plotly_dark' : 'plotly_white',
      paper_bgcolor: plotBg(), plot_bgcolor: plotBg(),
      annotations: [{text: 'No valid data in t-slice', xref: 'paper', yref: 'paper',
                     x: 0.5, y: 0.5, showarrow: false,
                     font: {size: 14, color: annotCol()}}],
    }, {responsive: true});
    return;
  }

  // ── Continuous region traces ─────────────────────────────────────────────
  if (hasCont) {
    const ridFinite = ridCont.some(v => v != null && isFinite(v));
    if (colorOn && ridFinite) {
      segmentsByRegime(tCont, rCont, ridCont, cdCont, {
        lineWidth: 2.5, hovertemplate: T_HOVER, hoverlabel: hl,
      }).forEach(t => traces.push(t));
    } else {
      traces.push({
        type: 'scatter', x: tCont, y: rCont,
        customdata: cdCont,
        mode: 'lines',
        line: {color: accent, width: 2.5},
        showlegend: false,
        hovertemplate: T_HOVER,
        hoverlabel: hl,
      });
    }
  }

  // ── Discrete-day region: connecting lines + per-day markers ──────────────
  if (hasDisc) {
    const ridFinite = ridDisc.some(v => v != null && isFinite(v));
    if (colorOn && ridFinite) {
      // Regime-coloured connecting segments (thinner + slightly faded)
      segmentsByRegime(tDisc, rDisc, ridDisc, cdDisc, {
        lineWidth: 1.5, opacity: 0.9,
        hovertemplate: T_HOVER, hoverlabel: hl,
      }).forEach(t => traces.push(t));
      // Per-day markers coloured by regime — matches Dash's overlay markers
      // inside components/figures.py::build_tslice_figure L796-808.
      for (let k = 0; k < tDisc.length; k++) {
        const rv = ridDisc[k];
        const col = (rv != null && isFinite(rv) && rv >= 1 && rv <= 7)
          ? REGIME_HEX[(rv | 0) - 1] : '#888';
        traces.push({
          type: 'scatter',
          x: [tDisc[k]], y: [rDisc[k]],
          mode: 'markers',
          marker: {size: 6, color: col, line: {width: 1, color: 'white'}},
          showlegend: false,
          customdata: [cdDisc[k]],
          hovertemplate: T_HOVER,
          hoverlabel: hl,
        });
      }
    } else {
      traces.push({
        type: 'scatter', x: tDisc, y: rDisc,
        customdata: cdDisc,
        mode: 'markers+lines',
        marker: {size: 5, color: accent, line: {width: 1, color: 'white'}},
        line: {color: hexToRgba('#6d9eff', 0.4), width: 1.5},
        showlegend: false,
        hovertemplate: T_HOVER,
        hoverlabel: hl,
      });
    }
  }

  // ── Optimal t_cad marker — find R on the CURRENT curve at t_cad_opt_h ───
  // When the slice N_exp is off-optimum, the marker's R value differs from
  // data.R_opt (which was taken at both optima). Mirrors Dash upstream.
  const tOptH = data.t_cad_opt_h;
  if (tOptH != null && isFinite(tOptH)) {
    // Prefer whichever region contains t_cad_opt_h.
    let tSrc = null, rSrc = null, cdSrc = null;
    if (hasDisc) {
      let bestI = 0, bestD = Infinity;
      for (let k = 0; k < tDisc.length; k++) {
        const d = Math.abs(tDisc[k] - tOptH);
        if (d < bestD) { bestD = d; bestI = k; }
      }
      // Accept discrete match only when close (half-day tolerance).
      if (bestD < 12.0) { tSrc = tDisc[bestI]; rSrc = rDisc[bestI]; cdSrc = cdDisc[bestI]; }
    }
    if (rSrc == null && hasCont) {
      let bestI = 0, bestD = Infinity;
      for (let k = 0; k < tCont.length; k++) {
        const d = Math.abs(tCont[k] - tOptH);
        if (d < bestD) { bestD = d; bestI = k; }
      }
      tSrc = tCont[bestI]; rSrc = rCont[bestI]; cdSrc = cdCont[bestI];
    }
    if (rSrc != null && isFinite(rSrc) && rSrc > 0) {
      traces.push({
        type: 'scatter', x: [tSrc], y: [rSrc],
        mode: 'markers',
        marker: {size: 12, color: AMBER, symbol: 'diamond', line: {width: 1.5, color: 'white'}},
        name: 'Opt. t<sub>cad</sub>',
        customdata: [cdSrc || [N_opt_col, null, null, null]],
        hovertemplate:
          'N<sub>exp</sub> = %{customdata[0]:.4g}<br>' +
          't<sub>cad,opt</sub> = %{x:.4g} hr<br>' +
          't<sub>exp</sub> = %{customdata[1]:.3g} s<br>' +
          'q<sub>med</sub> = %{customdata[2]:.3g}<br>' +
          'D<sub>med</sub> = %{customdata[3]:.3g} Gpc<br>' +
          'R<sub>det</sub> = %{y:.4g} yr ⁻¹' +
          '<extra>Opt. t<sub>cad</sub></extra>',
        hoverlabel: hl,
      });
    }
  }

  // ── Shapes: gap rectangle + ZTF H-line + ZTF V-line + opt V-line ─────────
  const shapes = [];
  const annotations = [{
    text: 't<sub>cad</sub> slice  |  N<sub>exp</sub> = ' +
          (N_fix != null && isFinite(N_fix) ? _fmtNexp(N_fix) + ' fields' : '?'),
    xref: 'paper', yref: 'paper', x: 0.01, y: 1.0,
    showarrow: false, xanchor: 'left', yanchor: 'top',
    font: {size: 12, color: annotCol()},
  }];
  // Dashed `Opt. N_exp` callout when the slice is not near the optimum.
  if (data.N_opt != null && isFinite(data.N_opt) && N_fix != null
      && Math.abs(N_fix - data.N_opt) / (data.N_opt + 1e-30) > 0.02) {
    annotations.push({
      text: 'Opt. N<sub>exp</sub> = ' + _fmtNexp(data.N_opt),
      xref: 'paper', yref: 'paper', x: 0.99, y: 0.98,
      showarrow: false, xanchor: 'right', yanchor: 'top',
      font: {size: 10, color: AMBER},
    });
  }

  if (opticalOn && gapLo != null && gapHi != null && gapLo < gapHi) {
    shapes.push({
      type: 'rect', xref: 'x', yref: 'paper',
      x0: gapLo, x1: gapHi, y0: 0, y1: 1,
      fillcolor: gapCol, line: {width: 0}, layer: 'below',
    });
    annotations.push({
      text: 'gap',
      xref: 'x', yref: 'paper',
      x: Math.sqrt(gapLo * gapHi), y: 0.99,
      xanchor: 'center', yanchor: 'top', showarrow: false,
      font: {size: 10, color: AMBER},
    });
  }

  if (data.R_ztf != null && isFinite(data.R_ztf)) {
    shapes.push({
      type: 'line', xref: 'paper', x0: 0, x1: 1,
      yref: 'y', y0: data.R_ztf, y1: data.R_ztf,
      line: {color: CORAL, width: 1.5, dash: 'dot'},
    });
    annotations.push({
      text: 'R<sub>ZTF</sub> = ' + (+data.R_ztf).toPrecision(2).replace(/\.?0+$/, '') + ' yr ⁻¹',
      xref: 'paper', yref: 'y', x: 0.98, y: data.R_ztf,
      xanchor: 'right', yanchor: 'bottom', showarrow: false,
      font: {size: 11, color: CORAL},
    });
  }

  const tZtfH = data.t_cad_ztf_h;
  if (tZtfH != null && isFinite(tZtfH)) {
    shapes.push({
      type: 'line', yref: 'paper', y0: 0, y1: 1,
      xref: 'x', x0: tZtfH, x1: tZtfH,
      line: {color: CORAL, width: 1.2, dash: 'dot'},
    });
  }

  if (tOptH != null && isFinite(tOptH)) {
    shapes.push({
      type: 'line', yref: 'paper', y0: 0, y1: 1,
      xref: 'x', x0: tOptH, x1: tOptH,
      line: {color: AMBER, width: 1.8, dash: 'dash'},
    });
  }

  const layout = {
    // Dash's build_tslice_figure intentionally omits uirevision so the 2D view
    // resets on every update (matches Plotly.py default behaviour).
    template: dark ? 'plotly_dark' : 'plotly_white',
    paper_bgcolor: plotBg(), plot_bgcolor: plotBg(),
    font: {family: "'JetBrains Mono','Cascadia Code',monospace", color: fontCol(), size: 12},
    margin: {l: 64, r: 24, b: 48, t: 40},
    xaxis: {
      title: 't<sub>cad</sub>', type: 'log',
      tickvals: TCAD_TICKVALS_H, ticktext: TCAD_TICKTEXT,
      showgrid: true, gridcolor: gridColLight(),
    },
    yaxis: {title: 'R<sub>det</sub> [yr ⁻¹]', type: 'log', showgrid: true, gridcolor: gridColLight()},
    showlegend: true,
    legend: {x: 0.01, y: 0.98, xanchor: 'left', yanchor: 'top', bgcolor: 'rgba(0,0,0,0)', font: {size: 11}},
    annotations: annotations,
    shapes: shapes,
    hoverlabel: {bgcolor: hoverBg(), font: {color: hoverFontCol()}, bordercolor: 'rgba(0,0,0,0)'},
  };

  Plotly.react('plot-tslice', traces, layout, {responsive: true});
}

// ── Re-render all visible plots (theme change) ────────────────────────────
function rerenderAll(data) {
  const params = readParams();
  render3DSurface(data, params);
  if (_currentTab === 'nslice') renderNSlice(data);
  if (_currentTab === 'tslice') renderTSlice(data);
}

// ── Metrics bar update ─────────────────────────────────────────────────────
function updateMetricsBar(data) {
  document.getElementById('m-R-opt').textContent    = fmtR(data.R_opt) + (data.R_opt != null ? ' /yr' : '');
  document.getElementById('m-tcad-opt').textContent  = fmtT(data.t_cad_opt_s);
  document.getElementById('m-N-opt').textContent     = fmtN(data.N_opt);
  document.getElementById('m-texp-opt').textContent  = fmtT(data.t_exp_opt_s);
  document.getElementById('m-R-ztf').textContent     = fmtR(data.R_ztf) + (data.R_ztf != null ? ' /yr' : '');
  document.getElementById('m-tcad-ztf').textContent  = fmtT(data.t_cad_ztf_s);
  document.getElementById('m-N-ztf').textContent     = fmtN(data.N_ztf);
  document.getElementById('m-texp-ztf').textContent  = fmtT(data.t_exp_ztf_s);

  const gainEl = document.getElementById('m-gain');
  if (data.R_opt != null && data.R_ztf != null && isFinite(data.R_opt) && isFinite(data.R_ztf) && data.R_ztf > 0) {
    const gain = data.R_opt / data.R_ztf;
    gainEl.textContent = '×' + gain.toFixed(2);
    gainEl.className = 'metric-value metric-gain ' + (gain >= 1 ? 'positive' : 'negative');
  } else {
    gainEl.textContent = '—';
    gainEl.className = 'metric-value metric-gain';
  }
}

// ── Derived display update ─────────────────────────────────────────────────
function updateDerivedDisplays(data) {
  if (!data) return;
  const fmt = x => x >= 1e6 ? (x/1e6).toFixed(1)+'M' : x >= 1e3 ? (x/1e3).toFixed(1)+'k' : x.toFixed(1);
  document.getElementById('grb-ntotal-display').innerHTML  = 'R<sub>int</sub> = ' + fmt(data.R_int_yr) + ' yr⁻¹';
  document.getElementById('grb-ntoward-display').innerHTML = 'f<sub>b</sub>R<sub>int</sub> = ' + fmt(data.R_toward_day) + ' day⁻¹';
}

// ── CSV export ─────────────────────────────────────────────────────────────
document.getElementById('export-btn').addEventListener('click', () => {
  if (!_lastData || !_lastData.X_flat) { alert('No data yet — run a computation first.'); return; }
  const d = _lastData;
  const [ny, nx] = d.shape;
  const n = ny * nx;
  const rows = ['N_exp,t_cad_s,t_cad_h,log10_R_det,regime_id'];
  for (let i = 0; i < n; i++) {
    const N = d.X_flat[i], t = d.Y_flat[i], z = d.Z_log_flat ? d.Z_log_flat[i] : null, r = d.regime_flat ? d.regime_flat[i] : null;
    if (N == null || t == null) continue;
    // d.Y_flat is in hours (t_cad_h); t_cad_s = t_cad_h * 3600 to match Dash column order
    rows.push([
      N.toExponential(4),
      (t * 3600).toFixed(6),
      t.toFixed(6),
      z != null ? z.toFixed(4) : '',
      r != null ? Math.round(r) : '',
    ].join(','));
  }
  const blob = new Blob([rows.join('\n')], {type: 'text/csv'});
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = 'grb_detection_surface.csv'; a.click();
  URL.revokeObjectURL(url);
});

// ── Pyodide initialization ─────────────────────────────────────────────────
async function initPyodide() {
  try {
    setStatus('Loading Pyodide runtime…', true);
    pyodide = await loadPyodide({
      indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.27.0/full/',
    });
    window.pyodide = pyodide;

    setStatus('Installing NumPy…', true);
    await pyodide.loadPackage(['numpy']);

    setStatus('Unpacking physics engine…', true);
    const b64 = document.getElementById('physics-zip').textContent.trim();
    const bin = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
    await pyodide.unpackArchive(bin, 'zip');

    setStatus('Importing bridge module…', true);
    await pyodide.runPythonAsync('import standalone_bridge');

    // Warm-up call primes the LRU cache with default parameters
    setStatus('Warming up model cache…', true);
    await pyodide.runPythonAsync(`
import standalone_bridge as _b
_b.compute_all({
    'i_det':10,'A_log':-4.68,'f_live':0.2,'t_overhead_s':0.0,
    'omega_exp_deg2':47.0,'omega_srv_deg2':27500.0,'t_night_h':10.0,
    'p':2.5,'nu_log10':14.7,'E_kiso_log10':53.0,'n0_log10':0.0,
    'epsilon_e_log10':-1.0,'epsilon_B_log10':-2.0,'theta_j_rad':0.1,
    'gamma0_log10':2.5,'D_euc_gpc':5.28,'rho_grb_log10':2.415,
    'optical_survey':False,'color_regimes':False,
    'full_integral':False,'off_axis':False,'toh_approx':False,'nx':60,'ny':80,
})
print('Bridge ready')
`);

    const _sbMod = pyodide.globals.get('standalone_bridge');
    pyComputeAll    = _sbMod.compute_all;
    pyComputeNslice = _sbMod.compute_nslice;
    pyComputeTslice = _sbMod.compute_tslice;
    setStatus('Ready — rendering initial surface…', true);
    updateGrbCounts();
    updateNexpMaxDisplay();
    updateSubnightLimitDisplay();
    updateNsliceTfixDisplay();
    updateTsliceNfixDisplay();
    runUpdate();

  } catch (e) {
    setStatus('Initialization failed: ' + e.message);
    console.error(e);
  }
}

initPyodide();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Build function
# ---------------------------------------------------------------------------

def build() -> None:
    bridge = ROOT / "standalone_bridge.py"
    if not bridge.exists():
        raise FileNotFoundError(f"standalone_bridge.py not found at {bridge}")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for py in sorted((ROOT / "grb_detect").rglob("*.py")):
            if "__pycache__" not in py.parts:
                zf.write(py, py.relative_to(ROOT))
        zf.write(bridge, "standalone_bridge.py")
    zip_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    html = HTML_TEMPLATE.replace("%%PHYSICS_ZIP_B64%%", zip_b64)
    for key, sid, smin, smax, step, default, marks in _SLIDERS:
        html = html.replace(f"%%SLIDER_{key}%%", _cs_slider(sid, smin, smax, step, default, marks))
    out = ROOT / "grb_detection_rate.html"
    out.write_text(html, encoding="utf-8")
    size_kb = out.stat().st_size // 1024
    print(f"Written {out}  ({size_kb} KB)")

    # Quick sanity: verify zip contents
    zf_check = zipfile.ZipFile(io.BytesIO(base64.b64decode(zip_b64)))
    names = sorted(zf_check.namelist())
    print(f"Zip contains {len(names)} files:")
    for n in names:
        print(f"  {n}")


if __name__ == "__main__":
    build()
