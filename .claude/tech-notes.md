# Technical Notes

Cross-cutting implementation constraints and patterns for the Dash app.

## App architecture (canonical)

The frontend is split into packages so `grb_detect/` (physics) is never touched:

```
plot_3d.py             thin entry: Dash init, register callbacks, run server
components/
  layout.py            full page layout (navbar, metrics strip, sidebar, view panels)
  sidebar.py           DBC Accordion with parameter sliders and switches
  figures.py           ALL Plotly figure builders (3D surface, N-slice, t-slice, metrics bar)
callbacks/
  surface.py           main update callback (computes surface + slices + metrics) + CSV export
  sync.py              bidirectional slider ↔ input sync (clientside + server)
  ui.py                theme, sidebar collapse, view-mode switching, presets
assets/style.css       CSS design system: tokens, dark/light, component overrides
grb_detect/            physics engine — never modify
```

## Plotly 6 + Dash 4 numpy serialisation (CRITICAL)

**Problem:** Plotly 6 (Python) serialises `numpy` arrays as binary objects:
```json
{"dtype": "f8", "bdata": "E6v2/P8...", "shape": "260, 220"}
```
The Plotly.js bundled inside Dash 4.0 does **not** support this binary format. Any trace that receives a raw numpy array (`go.Surface`, `go.Scatter`, `go.Scatter3d`, marker `color=`, etc.) renders as blank with no error.

**Rule:** Always convert numpy arrays to plain Python lists before passing to Plotly trace constructors.

**Pattern:** `_a2l()` helper in `components/figures.py`:
```python
def _a2l(arr):
    """numpy → list for Plotly JSON serialisation (Plotly 6 / Dash 4 compat)."""
    return arr.tolist() if hasattr(arr, "tolist") else arr
```

Apply to every array argument: `x=_a2l(xs)`, `y=_a2l(ys)`, `z=_a2l(zs)`, `surfacecolor=_a2l(sc)`, `color=_a2l(c)`.

Note: Python `list(1d_numpy)` also works for 1-D arrays; `arr.tolist()` handles any dimensionality.

## Dash / DBC version constraints

| Package | Pinned | Reason |
|---------|--------|--------|
| `dash` | `==3.4.0` | Dash 4.0.0 ships renderer `v4_0_0`; DBC 2.0.4 requires `v3_4_0`. Symptom: page loads but sends **zero** POST callbacks. |
| `dash-bootstrap-components` | `>=2.0` | DBC 2.0+ brings `dbc.Switch` (bool value), `dbc.Accordion always_open=True`. |

Do **not** upgrade Dash until DBC releases a compatible version.

## Diagnosing blank Dash plots

Work through this checklist in order:

1. **Test the callback server-side.** Use `app.server.test_client()` to POST to `/_dash-update-component` and inspect the HTTP response. A 200 with the correct trace types means the problem is in the browser.

2. **Check for bdata.** Inspect the response JSON:
   ```python
   for trace in fig['data']:
       for k, v in trace.items():
           if isinstance(v, dict) and 'bdata' in v:
               print(f"BDATA: trace.{k}")  # → apply _a2l() fix
   ```

3. **Check CSS height chain.** Each ancestor in the flex column must have `min-height: 0` (not just the graph):
   `#app-root (100vh) → .app-body (flex:1) → .app-main (flex:1) → .view-panel (flex:1) → dcc.Graph (flex:1, minHeight:0)`

4. **Check `prevent_initial_call`.** The main surface callback must have `prevent_initial_call=False` (the default). Verify via `/_dash-dependencies`.

5. **Check component ID mismatches.** Every `Input(id=...)` and `Output(id=...)` must exactly match an `id=` in the layout. With `suppress_callback_exceptions=True` mismatches fail silently.

## Clientside callback patterns

**DOM theme application** (fires on every theme-store change, outputs a dummy store):
```javascript
function(theme) {
    document.documentElement.setAttribute("data-theme", theme || "dark");
    return window.dash_clientside.no_update;
}
```

**View-mode switching** (outputs panel `style` dicts, instant display toggling):
```javascript
function(mode) {
    var show = {display: "flex", flex: "1", minHeight: "0"};
    var hide = {display: "none"};
    return [
        mode === "3d"     ? show : hide,
        mode === "nslice" ? show : hide,
        mode === "tslice" ? show : hide,
    ];
}
```

`window.dash_clientside.no_update` is valid in Dash 3.x and 4.x.

## Python environment

Virtual environment: `.venv/Scripts/python` (system Python lacks numpy/dash).
Always run via `.venv\Scripts\python.exe plot_3d.py`, not system `python`.

Debug mode: set `GRB_DEBUG=1` environment variable — shows runtime info and status bar.
