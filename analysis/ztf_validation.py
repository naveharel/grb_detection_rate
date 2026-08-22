"""ZTF validation study: predicted vs observed serendipitous afterglow rates.

Drives the engine (read-only, via the same bridge helpers the app uses) at the
two real ZTF observing modes — now expressed as two-timescale night schedules
(docs/implementation_reference.tex, Sec. "Two-Timescale Night Schedule") — and
produces:

  1. the calibration echo for the recalibrated defaults (eps_B, D_Euc),
  2. the schedule vs pre-schedule encoding comparison (the headline change:
     the public mode's intra-night pair, and Mode B's i = 2 decoupled from
     its 6 visits/night),
  3. a *waterfall* per mode — old physics -> recalibrated -> detection-window
     settings -> identification cuts — against the observed rates,
  4. the anatomy of the old apparent match (thesis Sec 3.3, k = 10),
  5. q- and D-distributions of the predicted detected population,
  6. one-at-a-time sensitivities (now including the schedule knobs).

Observational targets (Ho et al. 2022; Andreoni et al. 2021):
  Mode A — ZTF-II public all-sky: ~15,000 deg2 every 2 nights, 2 visits/night
           a few hours apart, m_lim ~ 20.5, pipeline needs i ~ 2
           -> ~2 events / yr.
  Mode B — high-cadence partnership/ZUDS: ~2,500 deg2, 6 visits/night nightly,
           pipeline needs i ~ 2                    -> ~2-3 events / yr.
  Benchmark — Ho et al. Sec 4.1 strict intranight MC: 19,190 field-nights in
           2 yr (~1,240 deg2/night effective) -> lambda = 1.04 per 2 yr.
  Distributions — q_med <~ 1 (>= 80% on-axis-consistent), z = 0.88-2.9,
           median D_C ~ 3.7-3.9 Gpc, detected nuL_nu(1 d) ~ 1-4e44 erg/s.

Run with (from the repo root)::

    .venv/Scripts/python analysis/ztf_validation.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# Windows consoles default to cp1252 — keep the unicode in the tables safe.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grb_detect.constants import DAY_S  # noqa: E402
from grb_detect.params import GPC_TO_CM  # noqa: E402
import standalone_bridge as bridge  # noqa: E402


# ── Non-engine factor: effective coverage ────────────────────────────────────
# Weather, moon, Galactic-plane cut, reference-image and image-quality losses.
# Anchor: Ho et al. (2022) Sec 4.1 count only 19,190 strict-quality field-nights
# in 2 yr (~1,240 deg2/night usable) against a nominal HC footprint of
# 2,500 deg2/night — and their strict criteria also fold in the depth margin.
# For the loose accounting we take weather (~0.7) x moon/quality (~0.8) x
# |b| > 10 deg (~0.8) x reference availability (~0.8) ~ 0.35 as a single value.
# NOTE: f_eff (the night-window usable fraction of the schedule model) budgets
# the telescope TIME; eps_cov budgets which field-nights yield usable
# DISCOVERY data — they are different losses and both apply.
EPS_COVERAGE = 0.35

# ── Base physics params (shared by every configuration) ─────────────────────
PHYS_NEW = dict(p=2.2, nu_log10=14.7, E_kiso_log10=53.0, n0_log10=0.0,
                epsilon_e_log10=-1.0, epsilon_B_log10=-4.0, theta_j_rad=0.1,
                gamma0_log10=2.5, D_euc_gpc=4.55, rho_grb_log10=math.log10(260.0))
PHYS_OLD = dict(PHYS_NEW, epsilon_B_log10=-2.0, D_euc_gpc=5.28)

BASE_PARAMS = dict(
    t_overhead_s=15.0, omega_exp_deg2=47.0, t_night_h=10.0,
    A_log=-4.68, optical_survey=True, color_regimes=False, full_integral=True,
    qmin=0.0, Dmin_cm=0.0, s_fade=0.0, s_rise=0.0, s_mode="discrete",
    toh_approx=False, win_iminus1=False, win_tp=False,
)

T_NIGHT_S = BASE_PARAMS["t_night_h"] * 3600.0
F_NIGHT = T_NIGHT_S / DAY_S

# ── The two ZTF observing modes (night schedules; web/app.js PRESETS) ────────
# Both modes take ~319 exposures/night on the same telescope, so both give
# f_eff = 319 * 45 s / 36000 s ~= 0.40 — matching ZTF's ~40% public time share.
MODES = {
    "A (public 2/night every 2 nights)": dict(
        i_det=2, f_eff=0.40, N_v=2, dt_v_h=2.0, omega_srv_deg2=15000.0,
        N_exp=15000.0 / 47.0, t_cad_s=2.0 * DAY_S,
        observed_per_yr=2.0,
    ),
    "B (high-cad 6/night nightly)": dict(
        i_det=2, f_eff=0.40, N_v=6, dt_v_h=1.5, omega_srv_deg2=2500.0,
        N_exp=2500.0 / 47.0, t_cad_s=1.0 * DAY_S,
        observed_per_yr=2.5,
    ),
}

# Pre-schedule encodings of the same modes (the 2026-08-14 study): Mode A as a
# single visit every 2 nights (f_live = 0.08 -> f_eff = 0.192), Mode B as the
# degenerate intra-night schedule with i conflated to the 6 visits
# (f_live = 0.17 -> f_eff = 0.408; visit spacing = 0.98 * t_night / 6).
MODES_OLD_ENCODING = {
    "A (public 2/night every 2 nights)": dict(
        i_det=2, f_eff=0.08 / F_NIGHT, N_v=1, dt_v_h=2.0,
        omega_srv_deg2=15000.0, N_exp=15000.0 / 47.0, t_cad_s=2.0 * DAY_S,
        observed_per_yr=2.0,
    ),
    "B (high-cad 6/night nightly)": dict(
        i_det=6, f_eff=0.17 / F_NIGHT, N_v=6, dt_v_h=0.98 * 10.0 / 6.0,
        omega_srv_deg2=2500.0, N_exp=2500.0 / 47.0, t_cad_s=1.0 * DAY_S,
        observed_per_yr=2.5,
    ),
}


def eval_config(mode: dict, physics: dict, **over) -> dict:
    """Evaluate (R, t_exp, q_med, D_med) at a mode's operating point."""
    params = dict(BASE_PARAMS)
    params.update(physics)
    params.update(omega_srv_deg2=mode["omega_srv_deg2"], i_det=mode["i_det"])
    for key in ("f_eff", "f_live", "N_v", "dt_v_h"):
        if key in mode:
            params[key] = mode[key]
    params.update(over)
    state = bridge._build_models(params)
    R, t_exp, q_med, D_med = bridge._eval_point(
        mode["N_exp"], mode["t_cad_s"], state["i_det"], state["model"],
        state["toh_approx"], state["t_overhead_s"],
        full_integral=state["full_on"], q_min=state["q_min"],
        D_min_cm=state["D_min_cm"], s_fade=state["s_fade"],
        s_rise=state["s_rise"], s_mode=state["s_mode"],
    )
    return dict(R=R, t_exp=t_exp, q_med=q_med, D_med_Gpc=D_med, state=state,
                params=params)


def dists_at_config(mode: dict, physics: dict, **over) -> dict:
    """q/D distribution stats of the detected population at a config."""
    res = eval_config(mode, physics, **over)
    state = res["state"]
    model = state["model"]
    kw = dict(q_min=state["q_min"], D_min_cm=state["D_min_cm"],
              s_fade=state["s_fade"], s_rise=state["s_rise"],
              s_mode=state["s_mode"])
    q, dRdq = model.dR_dq_full_integral(
        state["i_det"], mode["N_exp"], mode["t_cad_s"], N_q=800, **kw)
    D, dRdD = model.dR_dD_full_integral(
        state["i_det"], mode["N_exp"], mode["t_cad_s"], N_q=500, N_D=400, **kw)
    cq = np.cumsum(0.5 * (dRdq[1:] + dRdq[:-1]) * np.diff(q))
    cD = np.cumsum(0.5 * (dRdD[1:] + dRdD[:-1]) * np.diff(D))
    q_dec = float(model.derived.q_dec)
    frac_onaxis = float(np.interp(q_dec, q[1:], cq / cq[-1]))
    # "On-axis-consistent" in the observational sense: viewed within roughly
    # the jet edge + beaming-cone smoothing (light curves indistinguishable
    # from on-axis) — a looser boundary than the strict q < q_dec.
    frac_q15 = float(np.interp(1.5, q[1:], cq / cq[-1]))
    q_med = float(np.interp(0.5, cq / cq[-1], q[1:]))
    D_med = float(np.interp(0.5, cD / cD[-1], D[1:])) / GPC_TO_CM
    D_90 = float(np.interp(0.9, cD / cD[-1], D[1:])) / GPC_TO_CM
    return dict(q_med=q_med, frac_onaxis=frac_onaxis, frac_q15=frac_q15,
                D_med_Gpc=D_med, D_90_Gpc=D_90, R=res["R"])


def fmt(x, n=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:.{n}g}"


def print_calibration_echo():
    print("=" * 78)
    print("CALIBRATION ECHO (see model_gap_report.md for motivation)")
    print("=" * 78)
    from grb_detect.params import AfterglowPhysicalParams, MicrophysicsParams
    from grb_detect.pls import PLSG
    from grb_detect.afterglow_ism import t_dec_s

    for tag, phys_kw in (("old (eps_B=1e-2, D=5.28)", PHYS_OLD),
                         ("new (eps_B=10^-4, D=4.55)", PHYS_NEW)):
        phys = AfterglowPhysicalParams(
            E_kiso_erg=10 ** phys_kw["E_kiso_log10"],
            D_euc_cm=phys_kw["D_euc_gpc"] * GPC_TO_CM)
        micro = MicrophysicsParams(epsilon_B=10 ** phys_kw["epsilon_B_log10"])
        td = t_dec_s(phys, micro)
        F_dec = PLSG().F_dec_Jy(phys, micro, td)
        F_1d = F_dec * (DAY_S / td) ** (-3.0 * (phys.p - 1.0) / 4.0)
        nuLnu = phys.nu_hz * 4 * math.pi * phys.D_euc_cm ** 2 * F_1d * 1e-23
        rho = 260.0
        R_int = (4 / 3) * math.pi * rho * phys_kw["D_euc_gpc"] ** 3
        onaxis = R_int * phys.theta_j_rad ** 2 / 2
        print(f"  {tag:<30} nuLnu(1d) = {nuLnu:.2e} erg/s   "
              f"F_dec = {F_dec * 1e3:.0f} mJy   on-axis all-sky = {onaxis:.0f}/yr")
    print("  targets: median detected nuLnu(1d) = 1.9e44 erg/s (Ho+22 Tab.5); "
          "on-axis all-sky = 511/yr (Ho+22 S4.1)")
    print()


def print_schedule_comparison():
    print("=" * 78)
    print("SCHEDULE vs PRE-SCHEDULE ENCODING (new physics defaults, final "
          "config: windows + id cuts)")
    print("=" * 78)
    over = dict(win_tp=True, win_iminus1=True, s_fade=0.3, s_rise=0.5)
    print(f"   {'mode / encoding':<44}{'R_raw':>9}{'x eps_cov':>11}"
          f"{'q_med':>8}{'D_med':>8}{'D_90':>8}")
    for name in MODES:
        for tag, mode in (("pre-schedule", MODES_OLD_ENCODING[name]),
                          ("schedule", MODES[name])):
            d = dists_at_config(mode, PHYS_NEW, **over)
            label = (f"{name.split(' ')[0]} {tag} (i={mode['i_det']}, "
                     f"N_v={mode['N_v']}, f_eff={mode['f_eff']:.2f})")
            print(f"   {label:<44}{fmt(d['R']):>9}"
                  f"{fmt(d['R'] * EPS_COVERAGE if d['R'] else None):>11}"
                  f"{fmt(d['q_med']):>8}{fmt(d['D_med_Gpc']):>8}"
                  f"{fmt(d['D_90_Gpc']):>8}")
        print(f"   {'':<44}{'':>9}{'':>11}   observed ≈ "
              f"{MODES[name]['observed_per_yr']}/yr; obs D_C med ≈ "
              f"{3.67 if name.startswith('A') else 3.88} Gpc")
    print()


def print_waterfall():
    print("=" * 78)
    print("WATERFALL: predicted R_det [/yr] per ZTF mode "
          "(full-integral; eps_cov applied only in the last rows)")
    print("=" * 78)
    cuts = dict(s_fade=0.3, s_rise=0.5)
    steps = [
        ("old physics, conservative window", PHYS_OLD, {}),
        ("recalibrated defaults",          PHYS_NEW, {}),
        ("+ window from peak (t_on≈t_p)",  PHYS_NEW, dict(win_tp=True)),
        ("+ id cuts s_fade=.3 s_rise=.5",  PHYS_NEW, dict(win_tp=True, **cuts)),
        ("+ optimistic wait [T_req=S_c]",  PHYS_NEW, dict(win_tp=True,
                                                          win_iminus1=True, **cuts)),
    ]
    for name, mode in MODES.items():
        print(f"\n-- Mode {name}  (observed ≈ {mode['observed_per_yr']}/yr) --")
        print(f"   {'step':<34}{'R_raw':>10}{'x eps_cov':>12}{'q_med':>8}{'D_med':>8}")
        R_bracket = {}
        for label, phys, over in steps:
            r = eval_config(mode, phys, **over)
            Rc = r["R"] * EPS_COVERAGE if r["R"] is not None else None
            print(f"   {label:<34}{fmt(r['R']):>10}{fmt(Rc):>12}"
                  f"{fmt(r['q_med']):>8}{fmt(r['D_med_Gpc']):>8}")
            if label.startswith("+ id cuts"):
                R_bracket["cons"] = r["R"]
            if label.startswith("+ optimistic"):
                R_bracket["opt"] = r["R"]
        # The conservative/optimistic wait conventions bracket the schedule
        # phase ramp P_i(T) exactly (tex, Sec. sched_dominant); in exact mode
        # win_tp integrates the ramp itself, so the pair here brackets only
        # the residual wait convention of the cut deadlines. Quote the
        # geometric mean as the point estimate.
        if all(k in R_bracket for k in ("cons", "opt")):
            lo, hi = sorted((R_bracket["cons"], R_bracket["opt"]))
            gm = math.sqrt(lo * hi)
            print(f"   {'wait bracket [cons, opt] (w/ cuts)':<34}"
                  f"{f'[{lo:.3g}, {hi:.3g}]':>10}   geo-mean = {gm:.3g}"
                  f"   x eps_cov = [{lo * EPS_COVERAGE:.2g}, "
                  f"{hi * EPS_COVERAGE:.2g}]")
    print()


def print_thesis_coincidence():
    print("=" * 78)
    print("ANATOMY OF THE OLD APPARENT MATCH (thesis §3.3: k=10, 3,000 deg²)")
    print("=" * 78)
    # Thesis configuration: F_lim = 3e-5 Jy, t_cad = 1 d, i = 10, 3,000 deg².
    # Map F_lim to f_live: F_lim = A (t_exp/30 s)^-1/2 with t_OH = 0.
    # (Non-optical mode: f_eff is the wall-clock live fraction, no schedule.)
    t_exp = 30.0 * (10 ** -4.68 / 3e-5) ** 2
    N_exp = 3000.0 / 47.0
    f_live = N_exp * t_exp / DAY_S
    mode = dict(i_det=10, f_eff=f_live, omega_srv_deg2=3000.0,
                N_exp=N_exp, t_cad_s=1.0 * DAY_S, observed_per_yr=6.5)
    r10 = eval_config(mode, PHYS_OLD, t_overhead_s=0.0, optical_survey=False)
    mode2 = dict(mode, i_det=2)
    r2 = eval_config(mode2, PHYS_OLD, t_overhead_s=0.0, optical_survey=False)
    print(f"  old defaults, i=10, 3,000 deg², 1-day cadence:  "
          f"R = {fmt(r10['R'])}/yr  (thesis quoted ≈ 7.1/yr; observed ≈ 6.5/yr)")
    print(f"  same config at the *real* i = 2:                "
          f"R = {fmt(r2['R'])}/yr  — the 'match' needed k=10 + a small footprint")
    print()


def print_distributions():
    print("=" * 78)
    print("DETECTED-POPULATION DISTRIBUTIONS (final config: new defaults + "
          "windows + id cuts)")
    print("=" * 78)
    over = dict(win_tp=True, win_iminus1=True, s_fade=0.3, s_rise=0.5)
    for name, mode in MODES.items():
        d = dists_at_config(mode, PHYS_NEW, **over)
        print(f"  Mode {name}:")
        print(f"    q_med = {d['q_med']:.2f}   strictly on-axis (q < q_dec) = "
              f"{d['frac_onaxis'] * 100:.0f}%   on-axis-consistent (q < 1.5) = "
              f"{d['frac_q15'] * 100:.0f}%")
        print(f"    D_med = {d['D_med_Gpc']:.2f} Gpc   D_90 = "
              f"{d['D_90_Gpc']:.2f} Gpc   (D_Euc = 4.55 Gpc)")
    print("  targets: q_med <~ 1.5, on-axis fraction >= 60%; observed group "
          "medians D_C ≈ 3.67 (low-cad) / 3.88 (high-cad) Gpc, up to 6.26; "
          "the hard D_Euc wall is a tolerated limitation.")
    print()


def print_benchmark():
    print("=" * 78)
    print("HO ET AL. §4.1 STRICT BENCHMARK (their MC: λ = 1.04 per 2 yr)")
    print("=" * 78)
    # Strict criteria: >= 2 r-band detections in one night out of 3 r visits
    # -> schedule (t_cad = 1 d, N_v = 3, dt_v ~ t_night/3), i = 2; first
    # detection 0.5 mag above the m~20.5 limit (m < 20 -> raise A by 0.2 dex);
    # previous-night constraint ~ rise cut.
    # Coverage: 19,190 field-nights / 730 nights x 47 deg² ~ 1,240 deg²
    # effective — weather/moon/|b| cuts are already inside their audit, so no
    # extra eps_cov here. f_eff converts the old wall-clock budget:
    # f_live = 0.17 x (omega_eff / 2500) -> f_eff = f_live / f_night.
    omega_eff = 19190.0 / 730.0 * 47.0
    mode = dict(i_det=2, f_eff=0.17 * (omega_eff / 2500.0) / F_NIGHT,
                N_v=3, dt_v_h=BASE_PARAMS["t_night_h"] / 3.0,
                omega_srv_deg2=omega_eff, N_exp=omega_eff / 47.0,
                t_cad_s=1.0 * DAY_S, observed_per_yr=0.52)
    r = eval_config(mode, PHYS_NEW, win_tp=True, win_iminus1=True,
                    s_fade=0.3, s_rise=0.5, A_log=-4.48)
    print(f"  model (i=2, N_v=3 nightly, {omega_eff:.0f} deg² effective, "
          f"m_id=20): R = {fmt(r['R'])}/yr   vs Ho MC expectation 0.52/yr")
    print()


def print_sensitivity():
    print("=" * 78)
    print("SENSITIVITY (Mode A final config; one-at-a-time)")
    print("=" * 78)
    mode = MODES["A (public 2/night every 2 nights)"]
    over0 = dict(win_tp=True, win_iminus1=True, s_fade=0.3, s_rise=0.5)
    base = eval_config(mode, PHYS_NEW, **over0)["R"]
    print(f"  baseline: R_raw = {fmt(base)}/yr  (x eps_cov {EPS_COVERAGE} = "
          f"{fmt(base * EPS_COVERAGE)}/yr)")
    for label, phys, m, over in [
        ("eps_B −0.5 dex (10^-4.5)", dict(PHYS_NEW, epsilon_B_log10=-4.5), mode, over0),
        ("eps_B +0.5 dex (10^-3.5)", dict(PHYS_NEW, epsilon_B_log10=-3.5), mode, over0),
        ("s_rise = 1.0", PHYS_NEW, mode, dict(over0, s_rise=1.0)),
        ("s_rise = 0.3", PHYS_NEW, mode, dict(over0, s_rise=0.3)),
        ("no id cuts", PHYS_NEW, mode, dict(over0, s_fade=0.0, s_rise=0.0)),
        ("i = 3", PHYS_NEW, dict(mode, i_det=3), over0),
        ("dt_v = 0.5 h", PHYS_NEW, dict(mode, dt_v_h=0.5), over0),
        ("dt_v = 5 h", PHYS_NEW, dict(mode, dt_v_h=5.0), over0),
        ("N_v = 1 (same budget)", PHYS_NEW, dict(mode, N_v=1), over0),
        ("conservative wait", PHYS_NEW, mode, dict(over0, win_iminus1=False)),
    ]:
        r = eval_config(m, phys, **over)
        ratio = (r["R"] / base) if (r["R"] and base) else float("nan")
        print(f"  {label:<26} R_raw = {fmt(r['R']):>8}/yr   (x{ratio:.2f})")
    print(f"  eps_cov in [0.2, 0.5] scales every row linearly.")
    print()


def main():
    print_calibration_echo()
    print_schedule_comparison()
    print_waterfall()
    print_thesis_coincidence()
    print_distributions()
    print_benchmark()
    print_sensitivity()


if __name__ == "__main__":
    main()
