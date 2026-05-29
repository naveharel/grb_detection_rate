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
// of the string 'Plasma' — guarantees Plotly.js renders identically to Plotly.py.
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

// Presets — each one only touches: i, f_live, A_log, omega_exp, t_oh, omega_srv, optical.
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
let pyComputeQdview = null;
let _computing = false;
let _pendingUpdate = false;
let _debounceTimer = null;
let _sliceDebounceTimer = null;   // independent debounce for slice-position drags
let _sliceComputing = { nslice: false, tslice: false, qdview: false };
let _sliceQueued    = { nslice: false, tslice: false, qdview: false };
let _lastData = null;
let _currentTab = '3d';
let _currentTheme = 'dark';
// Unified cumulative/differential mode for R(q) and R(D). One toggle in the
// shared big-sliders strip drives both panels; defaults to differential since
// the cumulative view is essentially the sidebar q_min/D_min filter visualised.
let _qdviewMode = 'diff';

// ── Status indicator (top-right of metrics strip) ─────────────────────────
function setStatus(msg, spinning = false, isError = false) {
  const el = document.getElementById('metric-status');
  if (!el) return;
  if (!msg && !spinning) { el.innerHTML = ''; el.classList.remove('error'); return; }
  el.innerHTML = (spinning ? '<span class="spinner"></span>' : '') + (msg || '');
  el.classList.toggle('error', !!isError);
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
    // Shared qdview ctrl strip only shows on the R(q) / R(D) tabs.
    const qdCtrl = document.getElementById('qdview-ctrl');
    if (qdCtrl) qdCtrl.classList.toggle('hidden', !(tab === 'qview' || tab === 'dview'));
    if (_lastData) {
      if (tab === 'nslice') renderNSlice(_lastData);
      else if (tab === 'tslice') renderTSlice(_lastData);
      else if (tab === 'qview') renderQView(_lastData);
      else if (tab === 'dview') renderDView(_lastData);
    }
  });
});

// ── Slider ↔ input sync ────────────────────────────────────────────────────
const SLIDER_IDS = [
  'i','flive','Alog','omegaexp','toh','omega_srv','qmin','Dmin','smin','tnight',
  'p','nu_log','Ekiso_log','n0_log','gamma0_log','thetaj','epse','epsB','deuc','rho_grb_log'
];

// Effective minimum: respects optional `data-min-floor` on the .cs-wrap, which
// is used to impose a dynamic lower bound on a slider while keeping its
// structural `min` (and the baked-in tick positions) unchanged.
function _effMin(sl) {
  const slMin = parseFloat(sl.min);
  const wrap = sl.closest('.cs-wrap');
  if (!wrap) return slMin;
  const floor = parseFloat(wrap.dataset.minFloor);
  return isFinite(floor) && floor > slMin ? floor : slMin;
}

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
  if (id === 'Alog') updateMagDisplay();
  if (id === 'flive') _updateTnightFloor();
  if (id === 'flive' || id === 'tnight') _updateTnightFloorNote();
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
  // Clamp to [effective min, slider.max]. _effMin honours an optional
  // dynamic floor set via data-min-floor on the .cs-wrap.
  const mn = _effMin(sl), mx = parseFloat(sl.max);
  const clamped = Math.min(Math.max(v, mn), mx);
  inp.value = clamped;
  sl.value = clamped;
  updateSliderVisual(sl);
  if (id === 'deuc' || id === 'thetaj' || id === 'rho_grb_log') updateGrbCounts();
  if (id === 'Alog') updateMagDisplay();
  if (id === 'flive') _updateTnightFloor();
  if (id === 'flive' || id === 'tnight') _updateTnightFloorNote();
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
    // Generic visual refresh on every input event — needed for sliders not in
    // SLIDER_IDS (e.g. slice-position sliders); idempotent for sidebar sliders.
    sl.addEventListener('input', () => updateSliderVisual(sl));

    // Discrete-value slider: when `data-discrete-values` is present, the slider
    // is restricted to exactly those log10 values. Drag positions map to the
    // nearest listed value; keyboard arrows move between adjacent values; no
    // continuous in-between values are reachable. Sidebar sliders (no attr)
    // keep their existing continuous step-based behaviour.
    const discreteValues = wrap.dataset.discreteValues
      ? wrap.dataset.discreteValues.split(',')
          .map(parseFloat).filter(v => isFinite(v))
          .sort((a, b) => a - b)
      : null;

    function nearestDiscrete(v) {
      let best = discreteValues[0], bestD = Math.abs(v - best);
      for (let i = 1; i < discreteValues.length; i++) {
        const d = Math.abs(v - discreteValues[i]);
        if (d < bestD) { bestD = d; best = discreteValues[i]; }
      }
      return best;
    }

    function valFromX(clientX) {
      const rect = area.getBoundingClientRect();
      const p = Math.min(1, Math.max(0, (clientX - rect.left - CS_R) / (rect.width - 2 * CS_R)));
      const slMin = parseFloat(sl.min), mx = parseFloat(sl.max), st = parseFloat(sl.step) || 1;
      // Compute the value from the click position using the structural min so
      // tick alignment stays consistent; clamp to the effective (dynamic) min
      // at the end so sliders with a data-min-floor cannot be dragged below it.
      let v = slMin + p * (mx - slMin);
      if (discreteValues && discreteValues.length) {
        v = nearestDiscrete(v);
      } else if (st > 0) {
        v = Math.round((v - slMin) / st) * st + slMin;
      }
      const floor = _effMin(sl);
      return Math.min(mx, Math.max(floor, v));
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
      const mn = _effMin(sl), mx = parseFloat(sl.max), st = parseFloat(sl.step) || 1;
      let v = parseFloat(sl.value);
      if (discreteValues && discreteValues.length) {
        // Discrete slider: arrows step between adjacent listed values.
        let idx = discreteValues.indexOf(v);
        if (idx < 0) idx = discreteValues.indexOf(nearestDiscrete(v));
        if      (e.key === 'ArrowRight' || e.key === 'ArrowUp')   idx = Math.min(discreteValues.length - 1, idx + 1);
        else if (e.key === 'ArrowLeft'  || e.key === 'ArrowDown') idx = Math.max(0, idx - 1);
        else if (e.key === 'Home') idx = 0;
        else if (e.key === 'End')  idx = discreteValues.length - 1;
        else return;
        e.preventDefault();
        applyVal(discreteValues[idx]);
        return;
      }
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
  _updateTnightFloorNote();
  _checkPresetDrift();
  triggerUpdate();
});

// Other toggles
['toh-approx-switch','regime-color-switch','full-integral-switch','s-mode-switch'].forEach(id => {
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
  const fn =
    which === 'nslice' ? pyComputeNslice :
    which === 'tslice' ? pyComputeTslice :
    which === 'qdview' ? pyComputeQdview : null;
  if (!fn) return;
  if (_sliceComputing[which]) { _sliceQueued[which] = true; return; }
  _sliceComputing[which] = true;
  _sliceQueued[which] = false;
  setStatus('Computing…', true);
  try {
    const params = readParams();
    const pyParams = pyodide.toPy(params);
    let pyResult;
    if (which === 'nslice') {
      const t_cad_fix_s = Math.pow(10, parseFloat(document.getElementById('nslice-tfix-slider').value));
      pyResult = fn(pyParams, t_cad_fix_s);
    } else if (which === 'tslice') {
      const N_fix = Math.pow(10, parseFloat(document.getElementById('tslice-nfix-slider').value));
      pyResult = fn(pyParams, N_fix);
    } else {  // qdview: shared (N_exp, t_cad) drive both R(q) and R(D)
      const N_fix       = Math.pow(10, parseFloat(document.getElementById('qdview-nfix-slider').value));
      const t_cad_fix_s = Math.pow(10, parseFloat(document.getElementById('qdview-tfix-slider').value));
      pyResult = fn(pyParams, N_fix, t_cad_fix_s);
    }
    const payload = pyResult.toJs({ dict_converter: Object.fromEntries });
    pyResult.destroy();
    pyParams.destroy();
    if (payload.error) {
      setStatus('Slice error: ' + String(payload.error).slice(0, 200), false, true);
    } else {
      // Merge slice payload into _lastData so renderers use fresh values.
      Object.assign(_lastData, payload);
      if (which === 'nslice' && _currentTab === 'nslice') renderNSlice(_lastData);
      if (which === 'tslice' && _currentTab === 'tslice') renderTSlice(_lastData);
      if (which === 'qdview' && _currentTab === 'qview')  renderQView(_lastData);
      if (which === 'qdview' && _currentTab === 'dview')  renderDView(_lastData);
      setStatus('');
    }
  } catch (e) {
    setStatus('Slice compute failed: ' + e.message, false, true);
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
document.getElementById('qdview-nfix-slider').addEventListener('input', function() {
  updateQdviewNfixDisplay();
  triggerSliceUpdate('qdview');
});
document.getElementById('qdview-tfix-slider').addEventListener('input', function() {
  updateQdviewTfixDisplay();
  triggerSliceUpdate('qdview');
});

// Cumulative/Differential unified toggle (in the shared big-sliders strip).
// Affects whichever of R(q) / R(D) is currently visible; the other re-renders
// next time it's shown.  Bridge always returns both arrays in _lastData, so
// the toggle is render-only.
document.querySelectorAll('.mode-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const mode = btn.dataset.mode;
    if (!mode || mode === _qdviewMode) return;
    _qdviewMode = mode;
    document.querySelectorAll('.mode-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.mode === mode);
    });
    if (_lastData) {
      if (_currentTab === 'qview') renderQView(_lastData);
      else if (_currentTab === 'dview') renderDView(_lastData);
    }
  });
});

// ── Preset loader ──────────────────────────────────────────────────────────
// Only 7 controls are touched.
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
  _updateTnightFloor();
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
    qmin:            v('qmin_slider'),
    Dmin_cm:         v('Dmin_slider') * 3.085677581491367e27,  // GPC_TO_CM
    s_min:           v('smin_slider'),
    s_mode:          b('s-mode-switch') ? 'continuous' : 'discrete',
    toh_approx:      b('toh-approx-switch'),
    nslice_tfix_log: v('nslice-tfix-slider'),
    tslice_nfix_log: v('tslice-nfix-slider'),
    qdview_nfix_log: v('qdview-nfix-slider'),
    qdview_tfix_log: v('qdview-tfix-slider'),
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
function updateQdviewNfixDisplay() {
  const sl = document.getElementById('qdview-nfix-slider');
  const el = document.getElementById('qdview-nfix-value');
  if (sl && el) el.textContent = _fmtNexp(Math.pow(10, parseFloat(sl.value)));
}
function updateQdviewTfixDisplay() {
  const sl = document.getElementById('qdview-tfix-slider');
  const el = document.getElementById('qdview-tfix-value');
  if (sl && el) el.textContent = _fmtTcad(Math.pow(10, parseFloat(sl.value)));
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

// Dynamic t_night lower bound: enforces f_night ≥ f_live so that the
// continuous (sub-night) f_live_eff = f_live / f_night stays ≤ 1.
// Sets data-min-floor on the t_night .cs-wrap (read by _effMin) and clamps
// the current t_night value if it falls below the new floor.
function _updateTnightFloor() {
  const tn = document.getElementById('tnight_slider');
  if (!tn) return;
  const wrap = tn.closest('.cs-wrap');
  if (!wrap) return;
  const flive = parseFloat(document.getElementById('flive_slider').value);
  const structuralMin = parseFloat(tn.min);
  const max_h = parseFloat(tn.max);
  const step = parseFloat(tn.step) || 0.25;
  const rawFloor = Math.min(max_h, Math.max(structuralMin, isFinite(flive) ? 24 * flive : structuralMin));
  // Snap up to the nearest step multiple so the floor lands on the slider grid.
  const floor = Math.min(max_h, Math.ceil(rawFloor / step - 1e-9) * step);
  wrap.dataset.minFloor = String(floor);
  // Dim ticks/marks below the new floor (purely cosmetic — clamp is what
  // actually prevents reaching them).
  const pctFloor = (floor - structuralMin) / (max_h - structuralMin);
  wrap.querySelectorAll('.cs-tick').forEach(t => {
    const tp = parseFloat(t.dataset.pct);
    t.classList.toggle('below-floor', isFinite(tp) && tp < pctFloor - 1e-6);
  });
  wrap.querySelectorAll('.cs-mark').forEach(m => {
    const mp = parseFloat(m.style.getPropertyValue('--mpct'));
    m.classList.toggle('below-floor', isFinite(mp) && mp < pctFloor - 1e-6);
  });
  // Clamp current value up to the new floor if necessary.
  const cur = parseFloat(tn.value);
  if (isFinite(cur) && cur < floor - 1e-9) {
    tn.value = floor;
    const inp = document.getElementById('tnight_input');
    if (inp) inp.value = floor;
    updateSliderVisual(tn);
    updateSubnightLimitDisplay();
  }
  _updateTnightFloorNote();
}

function _updateTnightFloorNote() {
  const el = document.getElementById('tnight-floor-note');
  if (!el) return;
  const optical = document.getElementById('optical-switch').checked;
  const tn = document.getElementById('tnight_slider');
  if (!optical || !tn) { el.innerHTML = ''; return; }
  const wrap = tn.closest('.cs-wrap');
  const floor = wrap ? parseFloat(wrap.dataset.minFloor) : NaN;
  const cur = parseFloat(tn.value);
  if (!isFinite(floor) || !isFinite(cur)) { el.innerHTML = ''; return; }
  if (Math.abs(cur - floor) <= 1e-6) {
    el.innerHTML = '<span class="derived-info">f<sub>night</sub> must be bigger than f<sub>live</sub></span>';
  } else {
    el.innerHTML = '';
  }
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

// ── AB magnitude display (instant, no Python needed) ─────────────────────
function updateMagDisplay() {
  const A_log = parseFloat(document.getElementById('Alog_slider').value);
  const m_AB  = -2.5 * A_log + 2.5 * Math.log10(3631.0);
  document.getElementById('mag-ab-display').innerHTML =
    'm<sub>AB</sub> = ' + m_AB.toFixed(2) + ' mag';
}
updateMagDisplay();
_updateTnightFloor();

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
    setStatus('Error: ' + e.message, false, true);
    _computing = false;
    return;
  }

  if (data.error) {
    setStatus('Python error: ' + String(data.error).slice(0, 300), false, true);
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
  else if (_currentTab === 'qview')  renderQView(data);
  else if (_currentTab === 'dview')  renderDView(data);

  setStatus('');

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

// ── Standardized hovertemplates ────────────────────────────────────────────
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

// R(q) cumulative: x=q, y=R(q≥x); customdata=[N_exp, t_cad_h, fraction_of_total]
const Q_HOVER_CUM =
  'q ≥ %{x:.3g}<br>' +
  'N<sub>exp</sub> = %{customdata[0]:.4g}, t<sub>cad</sub> = %{customdata[1]:.4g} hr<br>' +
  'R<sub>det</sub>(q≥x) = %{y:.4g} yr ⁻¹<br>' +
  'Fraction of total = %{customdata[2]:.1%}' +
  '<extra></extra>';
// R(q) differential: x=q, y=dR/dq; customdata=[N_exp, t_cad_h]
const Q_HOVER_DIFF =
  'q = %{x:.3g}<br>' +
  'N<sub>exp</sub> = %{customdata[0]:.4g}, t<sub>cad</sub> = %{customdata[1]:.4g} hr<br>' +
  'dR<sub>det</sub>/dq = %{y:.4g} yr ⁻¹' +
  '<extra></extra>';
// R(D) cumulative: x=D [Gpc], y=R(D≥x); customdata=[N_exp, t_cad_h, fraction_of_total]
const D_HOVER_CUM =
  'D ≥ %{x:.3g} Gpc<br>' +
  'N<sub>exp</sub> = %{customdata[0]:.4g}, t<sub>cad</sub> = %{customdata[1]:.4g} hr<br>' +
  'R<sub>det</sub>(D≥x) = %{y:.4g} yr ⁻¹<br>' +
  'Fraction of total = %{customdata[2]:.1%}' +
  '<extra></extra>';
// R(D) differential: x=D [Gpc], y=dR/dD; customdata=[N_exp, t_cad_h]
const D_HOVER_DIFF =
  'D = %{x:.3g} Gpc<br>' +
  'N<sub>exp</sub> = %{customdata[0]:.4g}, t<sub>cad</sub> = %{customdata[1]:.4g} hr<br>' +
  'dR<sub>det</sub>/dD = %{y:.4g} yr ⁻¹ Gpc⁻¹' +
  '<extra></extra>';

// ── Regime-segmentation helper for 2D slice line traces ────────────────────
// Walks x/y/regime arrays in lockstep, emitting one Plotly scatter trace per
// contiguous run of equal regime IDs; optionally bridges to the first point of
// the next segment so neighbouring segments visually connect. `customdata` is
// a parallel array of per-point hover payloads (length == xArr.length).
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

// Only emit the optional hover lines when the scalar is finite.
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
// Days with no valid (unmasked) points are skipped. In regime-colour mode
// each consecutive run of equal regime IDs becomes its own segment; in
// height-colour mode a single light uniform line is drawn per day.
function addDay3DLines(traces, shared, params, zmaxLog) {
  const dl = shared && shared.dayLine;
  if (!dl || dl.n_days === 0) return;
  const regimeMode = !!params.color_regimes;
  const heightCmax = (zmaxLog != null && isFinite(zmaxLog)) ? zmaxLog : 1.0;

  // Height-colour mode: collect marker traces separately so they are appended
  // AFTER every per-day line trace. WebGL depth-sort ties break in favour of
  // later draw calls, so deferring the markers keeps dots painted above the
  // faint connecting lines regardless of camera angle.
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

  // Discrete day-cadence overlay lines (optical only); added BEFORE the main
  // surface so the surface z-order wins on overlapping regions.
  if (params.optical_survey) addDay3DLines(traces, shared, params, data.zmax_log10);

  // Optical mode: exclude discrete-day rows (y ≥ 24 h) from the surface — they
  // are rendered as 1D scatter3d lines by addDay3DLines instead. The surface
  // grid keeps only rows with y_s < DAY_S.
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

  // Include the surface max in Rmax so the zaxis adapts to actual data even
  // when R_opt / R_ztf are NaN (e.g. filters are restrictive but some surface
  // points still pass).
  const RmaxCands = [0.11];
  if (isFinite(data.R_opt)) RmaxCands.push(data.R_opt);
  if (isFinite(data.R_ztf)) RmaxCands.push(data.R_ztf);
  if (isFinite(data.R_surface_max) && data.R_surface_max > 0) RmaxCands.push(data.R_surface_max);
  const Rmax = Math.max(...RmaxCands);

  // 3D scene grid colour (slightly stronger than the 2D grids).
  const grid3d = dark ? 'rgba(255,255,255,0.14)' : 'rgba(0,0,0,0.12)';

  // Empty-state detection: a "real" surface trace is type==='surface'.
  // Phantom legend traces (scatter3d with x=[null]) and 3D markers don't count.
  const hasSurface = traces.some(t => t.type === 'surface');
  const annotText = dark ? '#8ba0c0' : '#4b6080';
  const layoutAnnotations = hasSurface ? [] : [{
    text: 'No detections above 0.01/yr — adjust filters',
    xref: 'paper', yref: 'paper', x: 0.5, y: 0.5,
    showarrow: false, xanchor: 'center', yanchor: 'middle',
    font: {size: 14, color: annotText},
  }];

  // Toggle uirevision based on data presence — forces a camera/zoom reset
  // across the empty ↔ populated transition so a stale view doesn't persist.
  const uirev = hasSurface ? 'keep-view-v1-data' : 'keep-view-v1-empty';

  const layout = {
    uirevision: uirev,
    annotations: layoutAnnotations,
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
// Slice position is driven by the nslice-tfix-slider (payload t_cad_fix_h), so
// the render no longer early-returns when the optimizer fails. The amber-
// diamond marker tracks the optimizer's N_opt on the CURRENT curve (R value
// read at argmin|N_sweep - N_opt|); when the slice is not near the optimum
// t_cad, an annotation calls that out.
function renderNSlice(data) {
  // Slice position comes from the bridge payload (falls back to optimum on
  // first render if the slice-position slider has not yet been consulted).
  const tCadRow = (data.t_cad_fix_h != null && isFinite(data.t_cad_fix_h))
    ? data.t_cad_fix_h
    : (data.t_cad_opt_h != null && isFinite(data.t_cad_opt_h) ? data.t_cad_opt_h : null);

  // Consume dedicated high-res N-sweep (800 points, logspaced) computed by
  // the bridge.
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
  // argmin|N_sweep - N_opt|.
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

  // Explicit axis ranges — Plotly.js's autorange + fixed tickvals can
  // otherwise widen the view far beyond the data extent. Match data range
  // with a small log-padding margin.
  const xPad = 0.05, yPad = 0.1;
  let xRange = null, yRange = null;
  if (xv.length > 0) {
    let xLo = Infinity, xHi = -Infinity;
    for (const x of xv) { if (x > 0) { if (x < xLo) xLo = x; if (x > xHi) xHi = x; } }
    if (isFinite(xLo) && isFinite(xHi) && xLo > 0 && xHi > 0) {
      xRange = [Math.log10(xLo) - xPad, Math.log10(xHi) + xPad];
    }
  }
  if (zv.length > 0) {
    let yLo = Infinity, yHi = -Infinity;
    for (const z of zv) { if (z > 0) { if (z < yLo) yLo = z; if (z > yHi) yHi = z; } }
    if (isFinite(yLo) && isFinite(yHi) && yLo > 0 && yHi > 0) {
      yRange = [Math.log10(yLo) - yPad, Math.log10(yHi) + yPad];
    }
  }

  const layout = {
    // No uirevision: the 2D view intentionally resets on every update
    // (matches Plotly.py default behaviour).
    template: dark ? 'plotly_dark' : 'plotly_white',
    paper_bgcolor: plotBg(), plot_bgcolor: plotBg(),
    font: {family: "'JetBrains Mono','Cascadia Code',monospace", color: fontCol(), size: 12},
    margin: {l: 64, r: 24, b: 48, t: 40},
    xaxis: Object.assign(
      {title: 'N<sub>exp</sub>', type: 'log', showgrid: true, gridcolor: gridColLight()},
      xRange ? {range: xRange, autorange: false} : {autorange: true},
    ),
    yaxis: Object.assign(
      {title: 'R<sub>det</sub> [yr ⁻¹]', type: 'log', showgrid: true, gridcolor: gridColLight()},
      yRange ? {range: yRange, autorange: false} : {autorange: true},
    ),
    showlegend: true,
    legend: {x: 0.01, y: 0.98, xanchor: 'left', yanchor: 'top', bgcolor: 'rgba(0,0,0,0)', font: {size: 11}},
    annotations: annotations,
    shapes: shapes,
    hoverlabel: {bgcolor: hoverBg(), font: {color: hoverFontCol()}, bordercolor: 'rgba(0,0,0,0)'},
  };

  Plotly.react('plot-nslice', traces, layout, {responsive: true});
}

// ── T-slice render (R vs t_cad at optimal N_exp) ───────────────────────────
// Builds two data regions: (1) a continuous sub-night sweep (t_cad < gap_lo_h)
// taken from the 2-D grid column at N_opt, and (2) a discrete-day region
// (integer-day cadences) extracted from the shared `dayLine` payload. In
// optical mode an amber gap rectangle spans [gap_lo_h, gap_hi_h]. In regime-
// colour mode both regions use `segmentsByRegime`, and discrete days get
// per-day markers coloured by regime.
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

  // Consume dedicated high-res t-slice sweeps (600 cont / 500 disc for
  // optical, or 1500 cont + 0 disc for non-optical) computed by the bridge.
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
      // Per-day markers coloured by regime.
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
  // data.R_opt (which was taken at both optima).
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

  // Explicit axis ranges — combine cont + disc regions; otherwise Plotly.js's
  // autorange + fixed TCAD_TICKVALS_H widens the axis to ~[1 sec, 1 yr] even
  // when data only spans an hour or two.
  const xPadT = 0.05, yPadT = 0.1;
  const tAll = hasCont ? (hasDisc ? tCont.concat(tDisc) : tCont) : (hasDisc ? tDisc : []);
  const rAll = hasCont ? (hasDisc ? rCont.concat(rDisc) : rCont) : (hasDisc ? rDisc : []);
  let xRangeT = null, yRangeT = null;
  if (tAll.length > 0) {
    let tLo = Infinity, tHi = -Infinity;
    for (const t of tAll) { if (t > 0) { if (t < tLo) tLo = t; if (t > tHi) tHi = t; } }
    if (isFinite(tLo) && isFinite(tHi) && tLo > 0 && tHi > 0) {
      xRangeT = [Math.log10(tLo) - xPadT, Math.log10(tHi) + xPadT];
    }
  }
  if (rAll.length > 0) {
    let rLo = Infinity, rHi = -Infinity;
    for (const r of rAll) { if (r > 0) { if (r < rLo) rLo = r; if (r > rHi) rHi = r; } }
    if (isFinite(rLo) && isFinite(rHi) && rLo > 0 && rHi > 0) {
      yRangeT = [Math.log10(rLo) - yPadT, Math.log10(rHi) + yPadT];
    }
  }

  const layout = {
    // No uirevision: the 2D view intentionally resets on every update
    // (matches Plotly.py default behaviour).
    template: dark ? 'plotly_dark' : 'plotly_white',
    paper_bgcolor: plotBg(), plot_bgcolor: plotBg(),
    font: {family: "'JetBrains Mono','Cascadia Code',monospace", color: fontCol(), size: 12},
    margin: {l: 64, r: 24, b: 48, t: 40},
    xaxis: Object.assign(
      {title: 't<sub>cad</sub>', type: 'log',
       tickvals: TCAD_TICKVALS_H, ticktext: TCAD_TICKTEXT,
       showgrid: true, gridcolor: gridColLight()},
      xRangeT ? {range: xRangeT, autorange: false} : {autorange: true},
    ),
    yaxis: Object.assign(
      {title: 'R<sub>det</sub> [yr ⁻¹]', type: 'log', showgrid: true, gridcolor: gridColLight()},
      yRangeT ? {range: yRangeT, autorange: false} : {autorange: true},
    ),
    showlegend: true,
    legend: {x: 0.01, y: 0.98, xanchor: 'left', yanchor: 'top', bgcolor: 'rgba(0,0,0,0)', font: {size: 11}},
    annotations: annotations,
    shapes: shapes,
    hoverlabel: {bgcolor: hoverBg(), font: {color: hoverFontCol()}, bordercolor: 'rgba(0,0,0,0)'},
  };

  Plotly.react('plot-tslice', traces, layout, {responsive: true});
}

// ── R(q) and R(D) views — shared (N_exp, t_cad) drive both panels ─────────
//
// The bridge payload always carries both cumulative and differential arrays;
// the per-panel toggle (_qviewMode / _dviewMode) flips which one we draw with
// zero recompute cost. Both views are at a fixed strategy, so the regime is
// constant along the curve — no segmentsByRegime here.

function _renderQDPlot(opts) {
  // Shared rendering for R(q) and R(D). opts:
  //   plotId, axisLabel ('q' or 'D [Gpc]'), xGrid, yCum, yDiff, mode ('cum'|'diff'),
  //   nFix, tCadH, totalRate, guides [{x,label,style}], accentMarker {x,label} or null,
  //   panelTitle ('R(q)' or 'R(D)'), hoverCum, hoverDiff, yLabelCum, yLabelDiff, xRange.
  const dark = darkMode();
  const accent = dark ? '#6d9eff' : '#3b6fff';
  const hl = {bgcolor: hoverBg(), font: {color: hoverFontCol()}, bordercolor: 'rgba(0,0,0,0)'};

  const xArr = opts.xGrid || [];
  const yArr = (opts.mode === 'diff') ? (opts.yDiff || []) : (opts.yCum || []);
  const n = Math.min(xArr.length, yArr.length);

  // Drop NaN/None and non-positive y (log axis) — Plotly is forgiving but
  // explicit filtering keeps the trace cleaner.
  const xv = [], yv = [], cdV = [];
  const totalR = (opts.totalRate != null && isFinite(opts.totalRate) && opts.totalRate > 0)
    ? opts.totalRate : null;
  for (let i = 0; i < n; i++) {
    const x = xArr[i], y = yArr[i];
    if (x == null || y == null) continue;
    if (!isFinite(x) || !isFinite(y) || !(y > 0)) continue;
    xv.push(x); yv.push(y);
    const frac = (opts.mode === 'cum' && totalR != null) ? y / totalR : null;
    cdV.push([opts.nFix, opts.tCadH, frac]);
  }

  const traces = [];

  if (xv.length === 0) {
    Plotly.react(opts.plotId, traces, {
      template: dark ? 'plotly_dark' : 'plotly_white',
      paper_bgcolor: plotBg(), plot_bgcolor: plotBg(),
      annotations: [{
        text: 'No valid data — strategy may be t<sub>OH</sub>-invalid',
        xref: 'paper', yref: 'paper', x: 0.5, y: 0.5, showarrow: false,
        font: {size: 13, color: annotCol()},
      }],
    }, {responsive: true});
    return;
  }

  traces.push({
    type: 'scatter', x: xv, y: yv,
    customdata: cdV,
    mode: 'lines',
    line: {color: accent, width: 2.5},
    showlegend: false,
    hovertemplate: (opts.mode === 'diff') ? opts.hoverDiff : opts.hoverCum,
    hoverlabel: hl,
  });

  // Vertical guides for physical reference scales (dashed gray).
  const shapes = [];
  const annotations = [];
  (opts.guides || []).forEach((g, idx) => {
    if (g.x == null || !isFinite(g.x)) return;
    shapes.push({
      type: 'line', yref: 'paper', y0: 0, y1: 1,
      xref: 'x', x0: g.x, x1: g.x,
      line: {color: annotCol(), width: 1.2, dash: 'dash'},
    });
    annotations.push({
      text: g.label, xref: 'x', yref: 'paper',
      x: g.x, y: 0.95 - 0.05 * idx,
      xanchor: 'left', yanchor: 'top', showarrow: false,
      font: {size: 10, color: annotCol()},
    });
  });

  // Accent marker for the sidebar's own filter value (only if > 0).
  if (opts.accentMarker && opts.accentMarker.x != null
      && isFinite(opts.accentMarker.x) && opts.accentMarker.x > 0) {
    shapes.push({
      type: 'line', yref: 'paper', y0: 0, y1: 1,
      xref: 'x', x0: opts.accentMarker.x, x1: opts.accentMarker.x,
      line: {color: accent, width: 1.6, dash: 'solid'},
    });
    annotations.push({
      text: opts.accentMarker.label, xref: 'x', yref: 'paper',
      x: opts.accentMarker.x, y: 0.04,
      xanchor: 'left', yanchor: 'bottom', showarrow: false,
      font: {size: 10, color: accent},
    });
  }

  // Strategy + mode annotation (top-left).
  const tCadStr = (opts.tCadH != null && isFinite(opts.tCadH))
    ? _fmtTcad(opts.tCadH * 3600) : '?';
  const nStr = (opts.nFix != null && isFinite(opts.nFix)) ? _fmtNexp(opts.nFix) : '?';
  // opts.panelTitle is "R(q)" or "R(D)" — charAt(2) extracts the inner symbol.
  const innerSym = opts.panelTitle.charAt(2);
  const modeLabel = (opts.mode === 'diff')
    ? ('Differential dR<sub>det</sub>/d' + innerSym)
    : 'Cumulative';
  annotations.push({
    text: opts.panelTitle + ' slice  |  N<sub>exp</sub> = ' + nStr +
          ', t<sub>cad</sub> = ' + tCadStr,
    xref: 'paper', yref: 'paper', x: 0.01, y: 1.0,
    showarrow: false, xanchor: 'left', yanchor: 'top',
    font: {size: 12, color: annotCol()},
  });
  annotations.push({
    text: modeLabel,
    xref: 'paper', yref: 'paper', x: 0.99, y: 1.0,
    showarrow: false, xanchor: 'right', yanchor: 'top',
    font: {size: 11, color: annotCol()},
  });

  // y-axis range from data.
  let yRange = null;
  let yLo = Infinity, yHi = -Infinity;
  for (const y of yv) { if (y > 0) { if (y < yLo) yLo = y; if (y > yHi) yHi = y; } }
  if (isFinite(yLo) && isFinite(yHi) && yLo > 0 && yHi > 0 && yHi > yLo * 1.0001) {
    yRange = [Math.log10(yLo) - 0.1, Math.log10(yHi) + 0.1];
  }

  const layout = {
    template: dark ? 'plotly_dark' : 'plotly_white',
    paper_bgcolor: plotBg(), plot_bgcolor: plotBg(),
    font: {family: "'JetBrains Mono','Cascadia Code',monospace", color: fontCol(), size: 12},
    margin: {l: 64, r: 24, b: 48, t: 40},
    xaxis: Object.assign(
      {title: opts.axisLabel, type: 'log', showgrid: true, gridcolor: gridColLight()},
      // Plotly log-axis ranges are in log10 of the data; convert here.
      (opts.xRange && opts.xRange[0] > 0 && opts.xRange[1] > 0)
        ? {range: [Math.log10(opts.xRange[0]), Math.log10(opts.xRange[1])], autorange: false}
        : {autorange: true},
    ),
    yaxis: Object.assign(
      {title: (opts.mode === 'diff') ? opts.yLabelDiff : opts.yLabelCum,
       type: 'log', showgrid: true, gridcolor: gridColLight()},
      yRange ? {range: yRange, autorange: false} : {autorange: true},
    ),
    showlegend: false,
    annotations: annotations,
    shapes: shapes,
    hoverlabel: {bgcolor: hoverBg(), font: {color: hoverFontCol()}, bordercolor: 'rgba(0,0,0,0)'},
  };

  Plotly.react(opts.plotId, traces, layout, {responsive: true});
}

function renderQView(data) {
  const qNr  = data.qdview_q_nr;
  const qDec = data.qdview_q_dec;
  const qJ   = data.qdview_q_j;
  const tCadH = data.qdview_t_cad_fix_h;
  const nFix  = data.qdview_N_fix;

  const guides = [];
  if (qDec != null && isFinite(qDec)) guides.push({x: qDec, label: 'q<sub>dec</sub>'});
  if (qJ   != null && isFinite(qJ))   guides.push({x: qJ,   label: 'q<sub>j</sub>'});
  if (qNr  != null && isFinite(qNr))  guides.push({x: qNr,  label: 'q<sub>nr</sub>'});

  const qMinSidebar = data.qdview_qmin_sidebar;
  const accentMarker = (qMinSidebar != null && qMinSidebar > 0)
    ? {x: qMinSidebar, label: 'q<sub>min</sub>'} : null;

  _renderQDPlot({
    plotId: 'plot-qview',
    axisLabel: 'q',
    xGrid: data.qdview_q_grid_flat || [],
    yCum:  data.qdview_Rq_cum_flat  || [],
    yDiff: data.qdview_Rq_diff_flat || [],
    mode: _qdviewMode,
    nFix: nFix,
    tCadH: tCadH,
    totalRate: data.qdview_total_rate_q,
    guides: guides,
    accentMarker: accentMarker,
    panelTitle: 'R(q)',
    hoverCum: Q_HOVER_CUM,
    hoverDiff: Q_HOVER_DIFF,
    yLabelCum:  'R<sub>det</sub>(q ≥ q<sub>min</sub>) [yr ⁻¹]',
    yLabelDiff: 'dR<sub>det</sub>/dq [yr ⁻¹]',
    // Log x-axis: lower bound matches the bridge view-grid (q_nr/200).
    xRange: (qNr != null && isFinite(qNr) && qNr > 0)
      ? [qNr / 200.0, qNr * 1.02] : null,
  });
}

function renderDView(data) {
  const dEucGpc = data.qdview_D_Euc_Gpc;
  const tCadH = data.qdview_t_cad_fix_h;
  const nFix  = data.qdview_N_fix;

  const guides = [];
  if (dEucGpc != null && isFinite(dEucGpc)) {
    guides.push({x: dEucGpc, label: 'D<sub>Euc</sub>'});
  }

  const dMinSidebar = data.qdview_Dmin_Gpc_sidebar;
  const accentMarker = (dMinSidebar != null && dMinSidebar > 0)
    ? {x: dMinSidebar, label: 'D<sub>min</sub>'} : null;

  _renderQDPlot({
    plotId: 'plot-dview',
    axisLabel: 'D [Gpc]',
    xGrid: data.qdview_D_grid_Gpc_flat || [],
    yCum:  data.qdview_RD_cum_flat     || [],
    yDiff: data.qdview_RD_diff_flat    || [],
    mode: _qdviewMode,
    nFix: nFix,
    tCadH: tCadH,
    totalRate: data.qdview_total_rate_D,
    guides: guides,
    accentMarker: accentMarker,
    panelTitle: 'R(D)',
    hoverCum: D_HOVER_CUM,
    hoverDiff: D_HOVER_DIFF,
    yLabelCum:  'R<sub>det</sub>(D ≥ D<sub>min</sub>) [yr ⁻¹]',
    yLabelDiff: 'dR<sub>det</sub>/dD [yr ⁻¹ Gpc⁻¹]',
    // Log x-axis: lower bound matches the bridge view-grid (D_Euc/1000).
    xRange: (dEucGpc != null && isFinite(dEucGpc) && dEucGpc > 0)
      ? [dEucGpc / 1000.0, dEucGpc * 1.02] : null,
  });
}

// ── Re-render all visible plots (theme change) ────────────────────────────
function rerenderAll(data) {
  const params = readParams();
  render3DSurface(data, params);
  if (_currentTab === 'nslice') renderNSlice(data);
  if (_currentTab === 'tslice') renderTSlice(data);
  if (_currentTab === 'qview')  renderQView(data);
  if (_currentTab === 'dview')  renderDView(data);
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

  if (data.t_dec_s != null && isFinite(data.t_dec_s)) {
    const t = data.t_dec_s;
    let val, unit;
    if      (t < 1)    { val = t * 1000; unit = 'ms';  }
    else if (t < 60)   { val = t;        unit = 's';   }
    else if (t < 3600) { val = t / 60;   unit = 'min'; }
    else               { val = t / 3600; unit = 'hr';  }
    document.getElementById('tdec-display').innerHTML =
      't<sub>dec</sub> = ' + val.toPrecision(3) + ' ' + unit;
  }

  if (data.F_nu_tdec_Jy != null && isFinite(data.F_nu_tdec_Jy)) {
    const Fj = data.F_nu_tdec_Jy;
    let val, unit;
    if      (Fj >= 1)    { val = Fj;       unit = 'Jy';  }
    else if (Fj >= 1e-3) { val = Fj * 1e3; unit = 'mJy'; }
    else if (Fj >= 1e-6) { val = Fj * 1e6; unit = 'μJy'; }
    else                 { val = Fj * 1e9; unit = 'nJy'; }
    document.getElementById('fnu-tdec-display').innerHTML =
      'F<sub>ν</sub>(t<sub>dec</sub>,D<sub>Euc</sub>) = ' + val.toPrecision(3) + ' ' + unit;
  }
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
    // d.Y_flat is in hours (t_cad_h); emit t_cad_s = t_cad_h * 3600 alongside.
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
    'full_integral':False,'qmin':0.0,'Dmin_cm':0.0,'s_min':0.0,'s_mode':'discrete','toh_approx':False,'nx':60,'ny':80,
})
print('Bridge ready')
`);

    const _sbMod = pyodide.globals.get('standalone_bridge');
    pyComputeAll    = _sbMod.compute_all;
    pyComputeNslice = _sbMod.compute_nslice;
    pyComputeTslice = _sbMod.compute_tslice;
    pyComputeQdview = _sbMod.compute_qdview;
    setStatus('Ready — rendering initial surface…', true);
    updateGrbCounts();
    updateNexpMaxDisplay();
    updateSubnightLimitDisplay();
    updateNsliceTfixDisplay();
    updateTsliceNfixDisplay();
    updateQdviewNfixDisplay();
    updateQdviewTfixDisplay();
    runUpdate();

  } catch (e) {
    setStatus('Initialization failed: ' + e.message, false, true);
    console.error(e);
  }
}

initPyodide();