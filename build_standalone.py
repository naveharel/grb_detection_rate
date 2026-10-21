"""Build script for the standalone GRB Detection Rate HTML app.

Zips grb_detect/ + standalone_bridge.py, base64-encodes the zip,
injects it into the HTML template, and writes grb_detection_rate.html.

Usage:
    python build_standalone.py
"""
from __future__ import annotations

import base64
import io
import math
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
        return str(v)

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
            + ",".join(str(v) for v in discrete_values)
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
        f' min="{fmt(smin)}" max="{fmt(smax)}" step="{"any" if discrete_values else fmt(step)}"'
        f' value="{fmt(default)}">'
        f'</div>'
    )

_CADENCE_SECONDS = (
    [100, 200, 300, 600, 900, 1200, 1800, 2700, 3600, 5400, 7200,
     10800, 14400, 21600, 28800, 43200, 64800]
    + [days * 86400 for days in (1, 1.5, 2, 3, 4, 5, 7, 10, 14, 21,
                                 30, 42, 60, 90, 120, 180, 270, 365)]
)
_CADENCE_LOGS = [math.log10(t) for t in _CADENCE_SECONDS]
_CADENCE_MARKS = [(math.log10(t), label) for t, label in
                  [(100, '100 s'), (3600, '1 hr'), (86400, '1 day'),
                   (172800, '2 day'), (604800, '1 wk'), (365 * 86400, '1 yr')]]
_FIELD_LOGS = [math.log10(n) for n in
               (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200,
                300, 500, 700, 1000, 1500, 2000, 3000, 5000, 7000, 10000)]

_SLIDERS = [
    # (KEY, sid, min, max, step, default, marks, slider_id_override?)
    ('I',           'i',           2,      30,      1,      2,       [(2,'2'),(10,'10'),(20,'20'),(30,'30')]),
    ('FLIVE',       'flive',       0.01,   1,       0.001,  0.166,   [(0.01,'0.01'),(0.166,'0.166'),(0.5,'0.5'),(1,'1')]),
    ('ALOG',        'Alog',        -12,    -2,      0.01,   -4.68,   [(-12,'-12'),(-8,'-8'),(-4.68,'-4.68'),(-2,'-2')]),
    ('OMEGAEXP',    'omegaexp',    1,      200,     1,      47,      [(1,'1'),(47,'47'),(100,'100'),(200,'200')]),
    ('TOH',         'toh',         0,      30,      0.5,    0,       [(0,'0'),(15,'15'),(30,'30')]),
    ('OMEGA_SRV',   'omega_srv',   100,    41253,   100,    41253,   [(100,'100'),(10000,'10k'),(15000,'15k'),(30000,'30k'),(41253,'41k')]),
    ('QMIN',        'qmin',        0,      14.5,    0.01,   0,       [(0,'0'),(1,'1'),(2,'2'),(5,'5'),(10,'10'),(14.5,'14.5')]),
    ('DMIN',        'Dmin',        0,      12,      0.01,   0,       [(0,'0'),(1,'1'),(4.55,'4.55'),(8,'8'),(12,'12')]),
    ('SFADE',       'sfade',       0,      2,       0.05,   0,       [(0,'0'),(0.3,'0.3'),(0.5,'0.5'),(1,'1'),(2,'2')]),
    ('SRISE',       'srise',       0,      2,       0.05,   0,       [(0,'0'),(0.3,'0.3'),(0.5,'0.5'),(1,'1'),(2,'2')]),
    ('TNIGHT',      'tnight',      1,      24,      0.25,   10,      [(1,'1'),(4,'4'),(6,'6'),(8,'8'),(10,'10'),(14,'14'),(18,'18'),(24,'24')]),
    ('P',           'p',           2.01,   3,       0.01,   2.2,     [(2.01,'2'),(2.2,'2.2'),(3,'3')]),
    ('NU_LOG',      'nu_log',      14.3,   15.1,    0.01,   14.7,    [(14.3,'14.3'),(14.7,'14.7'),(15.1,'15.1')]),
    ('EKISO_LOG',   'Ekiso_log',   51,     55,      0.1,    53,      [(51,'51'),(52,'52'),(53,'53'),(54,'54'),(55,'55')]),
    ('N0_LOG',      'n0_log',      -3,     2,       0.1,    0,       [(-3,'-3'),(-2,'-2'),(-1,'-1'),(0,'0'),(1,'1'),(2,'2')]),
    ('GAMMA0_LOG',  'gamma0_log',  2,      3.5,     0.05,   2.5,     [(2,'2'),(2.5,'2.5'),(3,'3'),(3.5,'3.5')]),
    ('THETAJ',      'thetaj',      0.01,   0.5,     0.01,   0.1,     [(0.01,'0.01'),(0.1,'0.1'),(0.3,'0.3'),(0.5,'0.5')]),
    ('EPSE',        'epse',        -2,     -0.3,    0.05,   -1,      [(-2,'-2'),(-1,'-1'),(-0.3,'-0.3')]),
    ('EPSB',        'epsB',        -5,     -1,      0.05,   -4,      [(-5,'-5'),(-4,'-4'),(-2,'-2'),(-1,'-1')]),
    ('DEUC',        'deuc',        1,      12,      0.01,   8.66,    [(1,'1'),(4,'4'),(8.66,'8.66'),(12,'12')]),
    ('RHO_GRB_LOG', 'rho_grb_log', 10,     500,     1,      138,     [(10,'10'),(138,'138'),(250,'250'),(400,'400'),(500,'500')]),
    # All-positive-L intensity; L = L_nu(t_dec), L_ref = 1e32 erg/s/Hz.
    ('LF_ALPHA',    'lf_alpha',    -2.49,  -1.01,   0.001,  -2,      [(-2.49,'-2.49'),(-2,'-2'),(-1.5,'-1.5'),(-1.01,'-1.01')]),
    ('LF_NORM_LOG', 'lf_norm_log', -3,     3,       0.05,   0,       [(-3,'-3'),(-1,'-1'),(0,'0'),(1,'1'),(3,'3')]),
    # TEMP-FDEC-OVERRIDE — begin (delete this row on removal)
    ('FDEC_LOG',    'fdec_log',    -6,     6,       0.01,   -2.05,   [(-6,'-6'),(-3,'-3'),(-2.05,'-2.05'),(3,'3'),(6,'6')]),
    # TEMP-FDEC-OVERRIDE — end
    # Exact physical anchors; logarithms are only slider coordinates. The hidden
    # range uses step="any" while the custom control enforces these positions.
    ('NSLICE_TFIX', 'nslice-tfix', 2, 8, 0.001, math.log10(86400),
        _CADENCE_MARKS, 'nslice-tfix-slider', _CADENCE_LOGS),
    ('TSLICE_NFIX', 'tslice-nfix', 0, 4, 0.001, 2.0,
        [(0,'1'),(1,'10'),(2,'100'),(3,'1k'),(4,'10k')],
        'tslice-nfix-slider', _FIELD_LOGS),
    ('QDVIEW_NFIX', 'qdview-nfix', 0, 4, 0.001, 2.0,
        [(0,'1'),(1,'10'),(2,'100'),(3,'1k'),(4,'10k')],
        'qdview-nfix-slider', _FIELD_LOGS),
    ('QDVIEW_TFIX', 'qdview-tfix', 2, 8, 0.001, math.log10(86400),
        _CADENCE_MARKS, 'qdview-tfix-slider', _CADENCE_LOGS),
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
