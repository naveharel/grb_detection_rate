"""Browser-level regression checks: drive the real, built HTML app.

Unlike the other tests/ files, these open the actual `grb_detection_rate.html`
in a real (headless) browser and interact with the real DOM controls — typing
into the visible numeric inputs next to each slider, clicking the real
checkboxes, selecting from the real preset <select> — the same way a person
clicking around the page would, and inspect the page's own state (the status
banner, the live Plotly figure) rather than calling any Python function
directly. This is deliberately the "closest to a real user" layer: the two
historical bugs guarded against here (a red-text error appearing when
enabling "Exact rate mode", and a marker point not lining up with the rest of
the graph) were both found by a person clicking around the live app, not by
a unit test.

Requires Playwright + a Chromium browser to be installed:

    pip install playwright
    playwright install chromium

Skips (not fails) if either is missing, so the regular fast `pytest tests/`
run is unaffected.

Run with::

    .venv/Scripts/python -m pytest tests/test_browser_e2e.py -v
"""
from __future__ import annotations

import math
import pathlib
import subprocess
import sys

import pytest

playwright_sync_api = pytest.importorskip("playwright.sync_api")
sync_playwright = playwright_sync_api.sync_playwright

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
HTML_PATH = REPO_ROOT / "grb_detection_rate.html"

# Boot-sequence substrings written to #metric-status while Pyodide/numpy/the
# physics engine load (see web/app.js initPyodide()) — wait until none of
# these are the current status text before interacting with the page.
_BOOT_SUBSTRINGS = (
    "Loading Pyodide", "Installing NumPy", "Unpacking physics engine",
    "Importing bridge module", "Warming up model cache",
    "Ready — rendering initial surface", "Computing",
)
_BOOT_TIMEOUT_MS = 120_000  # Pyodide + numpy download can be slow on CI
_COMPUTE_TIMEOUT_MS = 30_000  # exact-mode payloads are slower than the dominant-term default


@pytest.fixture(scope="module")
def built_html() -> pathlib.Path:
    """(Re)build the standalone HTML from current source so the test always
    exercises what `python build_standalone.py` actually produces."""
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "build_standalone.py")],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )
    assert HTML_PATH.exists(), "build_standalone.py did not produce the HTML file"
    return HTML_PATH


@pytest.fixture(scope="module")
def browser():
    try:
        with sync_playwright() as pw:
            try:
                b = pw.chromium.launch(headless=True)
            except Exception as e:  # browser binary not installed
                pytest.skip(f"Chromium not available for Playwright: {e}")
            yield b
            b.close()
    except Exception as e:
        pytest.skip(f"Playwright could not start: {e}")


@pytest.fixture
def page(browser, built_html):
    pg = browser.new_page()
    pg.goto(built_html.as_uri())
    _wait_for_ready(pg)
    yield pg
    pg.close()


# --------------------------------------------------------------------------- #
# Real-user-interaction helpers                                              #
# --------------------------------------------------------------------------- #

def _status_text(page) -> str:
    return page.locator("#metric-status").inner_text()


def _wait_for_ready(page) -> None:
    """Wait out the Pyodide boot sequence, then the initial compute."""
    page.wait_for_function(
        "() => { const t = document.getElementById('metric-status').innerText; "
        "return t === '' || (!t.includes('Loading') && !t.includes('Installing') "
        "&& !t.includes('Unpacking') && !t.includes('Importing') "
        "&& !t.includes('Warming') && !t.includes('Computing')); }",
        timeout=_BOOT_TIMEOUT_MS,
    )


def wait_for_compute(page) -> None:
    # runUpdate() is debounced 300ms (web/app.js triggerUpdate()) — the status
    # banner can still read '' (leftover from the previous render) for up to
    # that long before "Computing…" actually appears, so wait out the
    # debounce window first or a too-early check races ahead of the update.
    page.wait_for_timeout(400)
    page.wait_for_function(
        "() => !document.getElementById('metric-status').innerText.includes('Computing')",
        timeout=_COMPUTE_TIMEOUT_MS,
    )


def status_is_error(page) -> bool:
    el = page.locator("#metric-status")
    cls = el.get_attribute("class") or ""
    return "error" in cls.split()


def open_accordion(page, title: str) -> None:
    """Expand a collapsed sidebar <details> section (Constraints, Settings,
    ...) by clicking its <summary>, the way a real user would before they
    can see or touch the controls inside it. No-op if already open."""
    details = page.locator(f'details.acc-item:has(summary:text-is("{title}"))')
    if not details.evaluate("el => el.open"):
        details.locator("summary").click()


def set_number_input(page, slider_id: str, value: float) -> None:
    """Type into the real numeric input paired with a slider, then fire the
    same 'change' event a real blur/Enter would (mirrors syncFromInput)."""
    inp = page.locator(f"#{slider_id}_input")
    inp.fill(str(value))
    inp.dispatch_event("change")
    wait_for_compute(page)


def toggle_switch(page, switch_id: str, checked: bool | None = None) -> None:
    """Click the visible toggle-track (the checkbox itself is visually
    hidden by the custom switch styling — a real user clicks the track, which
    forwards the click to the <input> via the wrapping <label for=...>)."""
    open_accordion(page, "Settings")
    box = page.locator(f"#{switch_id}")
    is_checked = box.is_checked()
    if checked is None or checked != is_checked:
        page.locator(f'label[for="{switch_id}"] .toggle-track').click()
    wait_for_compute(page)


def select_preset(page, value: str) -> None:
    page.locator("#preset-select").select_option(value)
    wait_for_compute(page)


def read_traces(page) -> list[dict]:
    return page.evaluate("() => document.getElementById('plot-3d').data")


def preset_options(page) -> list[str]:
    vals = page.evaluate(
        "() => Array.from(document.getElementById('preset-select').options)"
        ".map(o => o.value).filter(v => v !== 'none')"
    )
    return vals


# --------------------------------------------------------------------------- #
# 1. App boots clean                                                         #
# --------------------------------------------------------------------------- #


def test_app_loads_without_error(page):
    assert not status_is_error(page), f"error on load: {_status_text(page)}"
    traces = read_traces(page)
    assert any(t.get("type") == "surface" for t in traces), "no surface trace drawn"


# --------------------------------------------------------------------------- #
# 2. Toggling exact mode must never show a red error banner                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("preset", ["none", "ztf_public", "ztf_hc", "rubin"])
def test_toggling_exact_mode_never_shows_red_error(page, preset):
    if preset != "none":
        assert preset in preset_options(page), f"preset {preset!r} not found on this build"
        select_preset(page, preset)
        assert not status_is_error(page), (
            f"error after selecting preset {preset}: {_status_text(page)}")

    toggle_switch(page, "full-integral-switch", checked=True)
    assert not status_is_error(page), (
        f"red error banner after enabling exact mode "
        f"(preset={preset}): {_status_text(page)}")

    # The three sub-refinements, toggled the way a curious user would.
    for sub_id in ("win-tp-switch", "rise-rs-switch", "fade-rs-switch"):
        toggle_switch(page, sub_id, checked=True)
        assert not status_is_error(page), (
            f"red error banner after enabling {sub_id} "
            f"(preset={preset}): {_status_text(page)}")

    toggle_switch(page, "full-integral-switch", checked=False)
    assert not status_is_error(page), (
        f"red error banner after disabling exact mode again "
        f"(preset={preset}): {_status_text(page)}")


# Extreme filter-slider combinations (q_min/D_min/s_fade/s_rise near their UI
# maxima) with exact mode + all sub-toggles on are covered natively and
# quickly by tests/test_exact_mode_no_crash.py instead — running that same
# sweep through a real browser is prohibitively slow (minutes per case under
# Pyodide's WASM numpy, worse the more Pyodide instances this file's other
# tests have already spun up in the same browser process) for the marginal
# extra coverage over the toggle-only check above.


# --------------------------------------------------------------------------- #
# 3. Markers must sit on the rendered surface                               #
# --------------------------------------------------------------------------- #


def _bilinear_interp(x_grid, y_grid, z_grid, x, y):
    """z_grid[row][col] with y_grid[row][0] varying by row, x_grid[0][col]
    varying by col (Plotly's row-major surface convention)."""
    xs = x_grid[0]
    ys = [row[0] for row in y_grid]
    if x < min(xs) or x > max(xs) or y < min(ys) or y > max(ys):
        return None
    ix = max(0, min(len(xs) - 2, next(i for i in range(len(xs) - 1) if xs[i] <= x <= xs[i + 1])))
    iy = max(0, min(len(ys) - 2, next(i for i in range(len(ys) - 1) if ys[i] <= y <= ys[i + 1])))
    x0, x1 = xs[ix], xs[ix + 1]
    y0, y1 = ys[iy], ys[iy + 1]
    z00, z01 = z_grid[iy][ix], z_grid[iy][ix + 1]
    z10, z11 = z_grid[iy + 1][ix], z_grid[iy + 1][ix + 1]
    if None in (z00, z01, z10, z11) or x1 == x0 or y1 == y0:
        return None
    tx = (x - x0) / (x1 - x0)
    ty = (y - y0) / (y1 - y0)
    z0 = z00 + tx * (z01 - z00)
    z1 = z10 + tx * (z11 - z10)
    return z0 + ty * (z1 - z0)


def _linear_interp_on_dayline(day_lines, x, y):
    """Optical mode's day-cadence markers sit on a per-day 1D overlay line
    (addDay3DLines in web/app.js), not the continuous surface grid — find the
    line at this marker's y (constant per line) and interpolate its x -> z."""
    for line in day_lines:
        ys = line["y"]
        if not ys or abs(ys[0] - y) > 1e-3:
            continue
        xs, zs = line["x"], line["z"]
        if x < min(xs) or x > max(xs):
            continue
        for i in range(len(xs) - 1):
            x0, x1 = xs[i], xs[i + 1]
            if x0 <= x <= x1 or x1 <= x <= x0:
                if x1 == x0:
                    return zs[i]
                t = (x - x0) / (x1 - x0)
                return zs[i] + t * (zs[i + 1] - zs[i])
    return None


@pytest.mark.parametrize("preset", ["none", "ztf_public", "ztf_hc", "rubin"])
def test_markers_sit_on_rendered_surface(page, preset):
    if preset != "none":
        select_preset(page, preset)
    traces = read_traces(page)
    surface = next((t for t in traces if t.get("type") == "surface"), None)
    assert surface is not None, "no surface trace drawn"
    day_lines = [t for t in traces if t.get("type") == "scatter3d" and t.get("mode") == "lines"]

    markers = [t for t in traces if t.get("type") == "scatter3d" and t.get("mode") == "markers+text"]
    assert markers, "no marker traces drawn"

    checked = 0
    for m in markers:
        x, y, z = m["x"][0], m["y"][0], m["z"][0]
        if x is None or y is None or z is None:
            continue
        interp = _bilinear_interp(surface["x"], surface["y"], surface["z"], x, y)
        source = "surface"
        if interp is None:
            # Discrete-day-cadence marker (y >= 24h): the continuous surface
            # grid excludes these rows in favour of the day-line overlay
            # (web/app.js render3DSurface) — check against that line instead.
            interp = _linear_interp_on_dayline(day_lines, x, y)
            source = "day-line"
        if interp is None:
            continue
        assert math.isclose(z, interp, rel_tol=0.05, abs_tol=0.05), (
            f"[{preset}] marker {m.get('name')!r} at (x={x}, y={y}) has "
            f"z={z} but the rendered {source} there is {interp} — "
            f"marker does not sit on the graph"
        )
        checked += 1
    assert checked >= 1, f"[{preset}] no marker fell on either the surface or a day-line to check"


# --------------------------------------------------------------------------- #
# 4. Marker follows the N_v / dt_v sliders in the live browser              #
# --------------------------------------------------------------------------- #


def test_ztf_marker_follows_nv_dtv_sliders(page):
    toggle_switch(page, "optical-switch", checked=True)

    def ztf_z():
        traces = read_traces(page)
        m = next(t for t in traces if t.get("type") == "scatter3d" and "ZTF public" in (t.get("name") or ""))
        return m["z"][0]

    set_number_input(page, "nv", 1)
    z1 = ztf_z()
    set_number_input(page, "nv", 3)
    set_number_input(page, "dtv", 1.0)
    z2 = ztf_z()
    assert z1 is not None and z2 is not None, "ZTF public marker not drawn at either schedule"
    assert not math.isclose(z1, z2, rel_tol=1e-6), (
        "ZTF marker z did not respond to N_v/dt_v — evaluated on a model "
        "other than the displayed one?"
    )
