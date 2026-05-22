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
               default: float, marks: list, slider_id: str | None = None,
               discrete_values: list | None = None) -> str:
    """Generate a .cs-wrap custom slider block with correct thumb/fill alignment.

    When `discrete_values` is provided, the slider is restricted to exactly
    those values — drag positions and keyboard arrows map onto the nearest
    listed value, and no in-between values are reachable. Used for the slice-
    position sliders so typical values (1 day, 2 day, 1 week, …) are the only
    selectable points.
    """
    if slider_id is None:
        slider_id = f"{sid}_slider"

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
        ticks.append(
            f'<div class="cs-tick" data-pct="{p:.4f}" style="--tick-pct:{p:.4f}"></div>')

    dflt_pct = pct(default)
    discrete_attr = ""
    if discrete_values:
        discrete_attr = (
            ' data-discrete-values="'
            + ",".join(f"{v:g}" for v in discrete_values)
            + '"'
        )
    return (
        f'<div class="cs-wrap" data-id="{sid}"{discrete_attr} style="--val-pct:{dflt_pct:.4f}">'
        f'<div class="cs-track-area">'
        f'<div class="cs-track-bg"></div>'
        f'<div class="cs-track-fill"></div>'
        + ''.join(ticks)
        + f'<div class="cs-thumb" tabindex="0" role="slider"'
        f' aria-valuemin="{fmt(smin)}" aria-valuemax="{fmt(smax)}"'
        f' aria-valuenow="{fmt(default)}"></div>'
        f'</div>'
        f'<div class="cs-marks">{"".join(mark_spans)}</div>'
        f'<input type="range" id="{slider_id}" class="cs-hidden" tabindex="-1"'
        f' min="{fmt(smin)}" max="{fmt(smax)}" step="{fmt(step)}"'
        f' value="{fmt(default)}">'
        f'</div>'
    )

_SLIDERS = [
    # (KEY, sid, min, max, step, default, marks, slider_id_override?)
    ('I',           'i',           2,      100,     1,      10,      [(2,'2'),(10,'10'),(30,'30'),(100,'100')]),
    ('FLIVE',       'flive',       0.01,   1,       0.01,   0.2,     [(0.01,'0.01'),(0.2,'0.2'),(0.5,'0.5'),(1,'1')]),
    ('ALOG',        'Alog',        -12,    -2,      0.01,   -4.68,   [(-12,'-12'),(-8,'-8'),(-4.68,'-4.68'),(-2,'-2')]),
    ('OMEGAEXP',    'omegaexp',    1,      200,     1,      47,      [(1,'1'),(47,'47'),(100,'100'),(200,'200')]),
    ('TOH',         'toh',         0,      30,      0.5,    0,       [(0,'0'),(15,'15'),(30,'30')]),
    ('OMEGA_SRV',   'omega_srv',   100,    41253,   100,    27500,   [(100,'100'),(10000,'10k'),(27500,'27.5k'),(41253,'41k')]),
    ('QMIN',        'qmin',        0,      14.5,    0.01,   0,       [(0,'0'),(1,'1'),(2,'2'),(5,'5'),(10,'10'),(14.5,'14.5')]),
    ('DMIN',        'Dmin',        0,      12,      0.01,   0,       [(0,'0'),(1,'1'),(5.28,'5.28'),(8,'8'),(12,'12')]),
    ('TNIGHT',      'tnight',      4,      24,      0.25,   10,      [(4,'4'),(8,'8'),(10,'10'),(14,'14'),(18,'18'),(24,'24')]),
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
    # Slice-position sliders (under N-slice / t-slice plots). Hyphen IDs preserved
    # so existing JS selectors (e.g. 'nslice-tfix-slider') keep working.
    # Discrete sliders: only the listed log10 values are reachable — no in-between
    # values. Step on the hidden <input type="range"> is kept fine (0.001) for
    # browser compatibility; the custom JS layer enforces discrete positions.
    ('NSLICE_TFIX', 'nslice-tfix', 2,      8,       0.001,  4.937,
        [(2,'100 s'),(3.556,'1 hr'),(4.937,'1 day'),(5.238,'2 day'),(5.781,'1 wk'),(7.499,'1 yr')],
        'nslice-tfix-slider',
        # log10(seconds) — reachable values (densely populated, includes all
        # original anchors: 100 s, 10 min, 30 min, 1 hr, 3 hr, 6 hr, 12 hr,
        # 1 day, 2 day, 3 day, 1 wk, 2 wk, 1 mo, 3 mo, 1 yr).
        [2.000,   # 100 s
         2.301,   # 200 s
         2.477,   # 5 min
         2.778,   # 10 min
         2.954,   # 15 min
         3.079,   # 20 min
         3.255,   # 30 min
         3.431,   # 45 min
         3.556,   # 1 hr
         3.732,   # 1.5 hr
         3.857,   # 2 hr
         4.034,   # 3 hr
         4.158,   # 4 hr
         4.335,   # 6 hr
         4.459,   # 8 hr
         4.636,   # 12 hr
         4.812,   # 18 hr
         4.937,   # 1 day
         5.113,   # 1.5 day
         5.238,   # 2 day
         5.414,   # 3 day
         5.539,   # 4 day
         5.636,   # 5 day
         5.781,   # 1 wk
         5.937,   # 10 day
         6.082,   # 2 wk
         6.259,   # 3 wk
         6.414,   # 1 mo (30 day)
         6.560,   # 6 wk
         6.715,   # 2 mo
         6.892,   # 3 mo
         7.017,   # 4 mo
         7.193,   # 6 mo
         7.368,   # 9 mo
         7.499]), # 1 yr
    ('TSLICE_NFIX', 'tslice-nfix', 0,      4,       0.001,  2.0,
        [(0,'1'),(1,'10'),(2,'100'),(3,'1k'),(4,'10k')],
        'tslice-nfix-slider',
        # log10(N_exp) — reachable values (densely populated, includes all
        # original anchors: 1, 2, 5, 10, 20, 50, 100, 200, 500, 1k, 2k, 5k, 10k).
        [0.000,   # 1
         0.301,   # 2
         0.477,   # 3
         0.699,   # 5
         0.845,   # 7
         1.000,   # 10
         1.176,   # 15
         1.301,   # 20
         1.477,   # 30
         1.699,   # 50
         1.845,   # 70
         2.000,   # 100
         2.176,   # 150
         2.301,   # 200
         2.477,   # 300
         2.699,   # 500
         2.845,   # 700
         3.000,   # 1k
         3.176,   # 1.5k
         3.301,   # 2k
         3.477,   # 3k
         3.699,   # 5k
         3.845,   # 7k
         4.000]), # 10k
]

# ---------------------------------------------------------------------------
# HTML template + web assets live under web/. build() loads them at build time.
#   - web/template.html : outer scaffold with %%STYLES%%, %%APP_JS%%,
#                         %%SLIDER_*%%, %%PHYSICS_ZIP_B64%% placeholders
#   - web/styles.css    : extracted CSS
#   - web/app.js        : extracted application JavaScript
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

    web = ROOT / "web"
    html = (web / "template.html").read_text(encoding="utf-8")
    html = html.replace("%%STYLES%%", (web / "styles.css").read_text(encoding="utf-8"))
    html = html.replace("%%APP_JS%%", (web / "app.js").read_text(encoding="utf-8"))
    html = html.replace("%%PHYSICS_ZIP_B64%%", zip_b64)
    for entry in _SLIDERS:
        key, sid, smin, smax, step, default, marks = entry[:7]
        slider_id = entry[7] if len(entry) > 7 else None
        discrete_values = entry[8] if len(entry) > 8 else None
        html = html.replace(f"%%SLIDER_{key}%%",
                            _cs_slider(sid, smin, smax, step, default, marks,
                                       slider_id, discrete_values))
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
