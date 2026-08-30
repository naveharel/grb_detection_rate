"""Simple-machinery mode-model check (2026-08-29, assessment only).

Question under test: can the two ZTF observing modes be represented with the
ORIGINAL simple machinery — fix i = 2, evaluate the legacy continuous model at
the intra-night spacing dt_v, weight by the nightly landing window, and divide
by the multi-day period n — instead of the schedule-cadence channel mixture?

The branch's schedule model is the referee: its dominant (gap-averaged)
mixture decomposes exactly into per-channel contributions
(grb_detect/schedule.py `channels()`), so the check quantifies what the
simple recipe keeps and what it drops.

Channel anatomy at i = 2 (N_v visits at spacing dt_v, visit-nights every
n days, overnight gap G = n·DAY − (N_v−1)·dt_v):

  A_G   (m=1,      gap G,    span dt_v) — burst lands in the overnight gap;
        caught by the NEXT night's first pair.  Its short-wait slice
        (u ≤ dt_v: burst lands just before the first visit) is the "fresh
        pair" detection; long waits are gap survivors.
  A_dlt (m=N_v−2,  gap dt_v, span dt_v) — burst lands in an interior
        intra-night gap; caught by the next two same-night visits (fresh).
  B     (m=1,      gap dt_v, span G)    — burst lands in the LAST intra-night
        gap; confirmation wraps the overnight gap (stale).

The simple intranight recipe ≡ (fresh terms), with the legacy rectangle at
T = i·dt_v / (i−1)·dt_v bracketing the gap average; its landing weight is
exactly (N_v−1)·dt_v / (n·DAY).  The overnight-only alternative ≡ legacy model
at t_cad = n·DAY (the pre-schedule public encoding).  Model-selection rule
(user decision 2026-08-29): the proposed simple model is ONE dominant term
per mode, never a mixture — this script measures which.

Mode schedules (= the app presets, web/app.js PRESETS):
  public — N_v=2, dt_v=2 h, every 2 nights (n=2), 15,000 deg²
  HC     — N_v=6, dt_v=1 h, nightly   (n=1),  2,500 deg²
(Note: the app preset dt_v = 1 h for HC, not ztf_validation.py's 1.5 h.)

Run with (from the repo root)::

    .venv/Scripts/python analysis/simple_mode_check.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ztf_validation as zv  # noqa: E402  (also bootstraps stdout utf-8)
from grb_detect.constants import DAY_S  # noqa: E402
from grb_detect.detection_rate import _GL5_NODES, _GL5_WEIGHTS  # noqa: E402

F_NIGHT = zv.F_NIGHT
T_OH = zv.BASE_PARAMS["t_overhead_s"]

# ── The two ZTF modes at the app-preset schedules ────────────────────────────
MODES = {
    "public (2/night @2h, every 2 nights)": dict(
        i_det=2, f_eff=0.40, N_v=2, dt_v_h=2.0, omega_srv_deg2=15000.0,
        N_exp=15000.0 / 47.0, t_cad_s=2.0 * DAY_S, n_days=2,
        observed_per_yr=2.0,
    ),
    "HC (6/night @1h, nightly)": dict(
        i_det=2, f_eff=0.40, N_v=6, dt_v_h=1.0, omega_srv_deg2=2500.0,
        N_exp=2500.0 / 47.0, t_cad_s=1.0 * DAY_S, n_days=1,
        observed_per_yr=2.5,
    ),
}

CHANNEL_NAMES = ("A_G  (overnight gap)", "A_dlt (interior gaps)",
                 "B    (last-gap wrap)")


def unified_texp(mode: dict) -> float:
    """The schedule model's budget: t_exp = f_eff·f_night·t_cad/(N_exp·N_v) − t_OH."""
    return (mode["f_eff"] * F_NIGHT * mode["t_cad_s"]
            / (mode["N_exp"] * mode["N_v"]) - T_OH)


def legacy_eval(mode: dict, t_cad_s: float, f_live: float, **over) -> dict:
    """Legacy continuous model (non-optical path) at an arbitrary (t_cad, f_live)."""
    m = dict(i_det=mode["i_det"], omega_srv_deg2=mode["omega_srv_deg2"],
             N_exp=mode["N_exp"], t_cad_s=t_cad_s, f_live=f_live)
    return zv.eval_config(m, zv.PHYS_NEW, optical_survey=False,
                          full_integral=False, **over)


def _rect(model, mode, T, S, g, cuts=None) -> float:
    """One hard-rectangle evaluation R_rect(T) of the channel machinery [linear /yr]."""
    z = model.rate_log10(
        mode["i_det"], np.array([mode["N_exp"]]), np.array([mode["t_cad_s"]]),
        _channel=(np.array([T]), np.array([S]), np.array([g])),
        **(cuts or {}))
    z = float(z[0])
    return 10.0 ** z if math.isfinite(z) else 0.0


def _gap_avg(model, mode, S, u_lo, u_hi, g, cuts=None) -> float:
    """(1/(u_hi−u_lo)) ∫ R_rect(S+u) du over [u_lo, u_hi] by GL5 (engine's rule)."""
    if u_hi <= u_lo:
        return 0.0
    acc = 0.0
    for x_n, om_n in zip(_GL5_NODES, _GL5_WEIGHTS):
        acc += om_n * _rect(model, mode, S + u_lo + x_n * (u_hi - u_lo), S, g,
                            cuts)
    return acc


def channel_regime(model, mode, T, S, g) -> str:
    """Regime label of the rectangle at window T (classification only)."""
    _, comps = model.rate_log10(
        mode["i_det"], np.array([mode["N_exp"]]), np.array([mode["t_cad_s"]]),
        return_components=True, _channel=(np.array([T]), np.array([S]), np.array([g])))
    for k in ("A1", "A2", "A3", "A4", "A5", "A6", "A7"):
        if bool(comps[k][0]):
            kind = "flux-lim" if k in ("A1", "A2", "A3") else (
                "cad-lim" if k in ("A4", "A5", "A6") else "doubly-lim")
            return f"{k}/{kind}"
    return "A0/invalid"


def decompose(model, mode, cuts=None) -> dict:
    """Engine-exact per-channel contributions to the dominant-mode mixture,
    plus the fresh (pair) / stale (gap-survivor) physics split.

    The fresh/stale split subdivides the A_G wait integral at u = dt_v, so it
    uses a composite (finer) GL5 rule; fresh+stale differs from the engine's
    single-GL5 channel value only by the quadrature error (documented ≲2%).
    """
    i_det = mode["i_det"]
    t_arr = np.array([mode["t_cad_s"]])
    dt_v = mode["dt_v_h"] * 3600.0
    rows, total = [], 0.0
    fresh, stale = 0.0, 0.0
    for name, (m_c, g_c, S_c) in zip(
            CHANNEL_NAMES, model.schedule.channels(i_det, t_arr)):
        if m_c == 0:
            rows.append(dict(name=name, w=0.0, R=0.0, reg_lo="—", reg_hi="—"))
            continue
        g = float(np.asarray(g_c).ravel()[0])
        S = float(np.asarray(S_c).ravel()[0])
        w_c = m_c * g / mode["t_cad_s"]
        R_c = w_c * _gap_avg(model, mode, S, 0.0, g, g, cuts)
        rows.append(dict(
            name=name, w=w_c, R=R_c,
            reg_lo=channel_regime(model, mode, S, S, g),
            reg_hi=channel_regime(model, mode, S + g, S, g)))
        total += R_c
        if name.startswith("A_dlt"):
            fresh += R_c
        elif name.startswith("B"):
            stale += R_c
        else:  # A_G: split the wait integral at u = dt_v (fresh pair vs survivor)
            u_split = min(dt_v, g)
            fresh += (m_c / mode["t_cad_s"]) * u_split * _gap_avg(
                model, mode, S, 0.0, u_split, g, cuts)
            stale += (m_c / mode["t_cad_s"]) * (g - u_split) * _gap_avg(
                model, mode, S, u_split, g, g, cuts)
    return dict(rows=rows, total=total, fresh=fresh, stale=stale)


def fmt(x, n=3):
    return zv.fmt(x, n)


def main() -> None:
    checks: list[tuple[str, bool]] = []
    print("=" * 78)
    print("SIMPLE-MACHINERY MODE MODEL vs SCHEDULE MODEL (assessment 2026-08-29)")
    print("conventions: new physics defaults, cuts off, win_iminus1/win_tp off")
    print("=" * 78)

    verdicts = {}
    for name, mode in MODES.items():
        n = mode["n_days"]
        dt_v = mode["dt_v_h"] * 3600.0
        N_v = mode["N_v"]
        t_exp_uni = unified_texp(mode)
        train_h = (N_v - 1) * mode["dt_v_h"]
        land_corr = (N_v - 1) * dt_v / mode["t_cad_s"]
        land_lit = F_NIGHT / n

        print(f"\n### Mode {name} — observed ≈ {mode['observed_per_yr']}/yr")
        print(f"  schedule: N_v={N_v}, dt_v={mode['dt_v_h']} h, n={n} night(s), "
              f"train span {train_h:g} h of a {zv.BASE_PARAMS['t_night_h']:g} h night")
        print(f"  unified budget t_exp = {t_exp_uni:.1f} s "
              f"(f_eff={mode['f_eff']}, N_exp={mode['N_exp']:.0f})")
        checks.append((f"{name}: unified t_exp ≈ 30 s",
                       abs(t_exp_uni - 30.0) < 1.0))

        # ── Reference: the branch schedule model.  With win_tp off the
        # gap-averaged mixture IS the exact phase average of the hard
        # detection criterion (up to GL5 ≲2%), so it is the referee; the
        # full-integral rows keep hard per-channel rectangles at the
        # endpoint conventions and bracket it.
        ref_dom = zv.eval_config(mode, zv.PHYS_NEW, full_integral=False)
        ref_cons = zv.eval_config(mode, zv.PHYS_NEW, full_integral=True)
        ref_opt = zv.eval_config(mode, zv.PHYS_NEW, full_integral=True,
                                 win_iminus1=True)
        model = ref_dom["state"]["model"]
        print(f"\n  REFERENCE (schedule model)"
              f"{'':<18}{'R [/yr]':>10}{'q_med':>8}{'D_med':>8}")
        for tag, r in (("phase-avg mixture (referee)", ref_dom),
                       ("full integral, cons. wait bracket", ref_cons),
                       ("full integral, opt. wait bracket", ref_opt)):
            print(f"    {tag:<40}{fmt(r['R']):>10}"
                  f"{fmt(r['q_med']):>8}{fmt(r['D_med_Gpc']):>8}")
        checks.append((f"{name}: model t_exp = unified t_exp",
                       abs(ref_dom["t_exp"] - t_exp_uni) < 1e-6))

        # ── Channel decomposition of the dominant-mode rate ─────────────────
        dec = decompose(model, mode)
        R_mix = dec["total"]
        checks.append((f"{name}: Σ channels = mixture rate",
                       abs(R_mix - ref_dom["R"]) < 1e-9 * max(ref_dom["R"], 1e-30)))
        checks.append((f"{name}: Σ channel weights = 1",
                       abs(sum(r["w"] for r in dec["rows"]) - 1.0) < 1e-12))
        print(f"\n  CHANNEL DECOMPOSITION (dominant mode)"
              f"{'':<4}{'weight':>8}{'R [/yr]':>10}{'share':>8}"
              f"   regime [best wait → worst]")
        for r in dec["rows"]:
            if r["w"] == 0.0:
                print(f"    {r['name']:<26}{'—':>8}{'—':>10}{'—':>8}   (empty)")
                continue
            print(f"    {r['name']:<26}{r['w']:>8.3f}{fmt(r['R']):>10}"
                  f"{r['R'] / R_mix:>8.1%}   {r['reg_lo']} → {r['reg_hi']}")
        frac_fresh = dec["fresh"] / (dec["fresh"] + dec["stale"])
        print(f"    physics split: fresh pair (wait ≤ dt_v) = "
              f"{fmt(dec['fresh'])}/yr ({frac_fresh:.1%})   "
              f"gap survivors = {fmt(dec['stale'])}/yr ({1 - frac_fresh:.1%})")
        checks.append((f"{name}: fresh+stale ≈ mixture (≲2% GL5 quadrature)",
                       abs(dec["fresh"] + dec["stale"] - R_mix)
                       < 0.04 * max(R_mix, 1e-30)))

        # Sensitivity: the same split under the app's identification cuts
        # (dominant-mode best-case boundaries; the rise cut's per-channel gap
        # g_c punishes the overnight gap survivors exponentially).
        cuts = dict(s_fade=0.3, s_rise=0.5)
        dec_c = decompose(model, mode, cuts=cuts)
        frac_fresh_c = dec_c["fresh"] / (dec_c["fresh"] + dec_c["stale"])
        print(f"    with id cuts (s_fade=0.3, s_rise=0.5): total = "
              f"{fmt(dec_c['total'])}/yr, fresh = {fmt(dec_c['fresh'])}/yr "
              f"({frac_fresh_c:.1%}), survivors = {fmt(dec_c['stale'])}/yr "
              f"({1 - frac_fresh_c:.1%})")

        # ── Candidate simple models ──────────────────────────────────────────
        print(f"\n  CANDIDATE SIMPLE MODELS (legacy machinery only)"
              f"{'':<6}{'t_exp':>7}{'R [/yr]':>10}{'/ref':>7}")

        # (a) literal recipe: legacy at dt_v, naive budget, × f_night ÷ n
        lit = legacy_eval(mode, dt_v, mode["f_eff"])
        R_lit = (lit["R"] * land_lit) if math.isfinite(lit["R"]) else math.nan
        t_lit = lit["t_exp"]
        note = ("INVALID (t_exp ≤ 0)" if not math.isfinite(t_lit)
                else f"shallow ({t_lit:.0f} s vs {t_exp_uni:.0f} s)"
                if t_lit < 0.8 * t_exp_uni else "")
        print(f"    literal: legacy(dt_v) × f_night/n = ×{land_lit:.3f}"
              f"{'':<3}{fmt(t_lit, 2):>7}{fmt(R_lit):>10}"
              f"{fmt(R_lit / R_mix, 2) if math.isfinite(R_lit) else '—':>7}"
              f"   {note}")

        # (b) corrected intranight recipe: forced unified budget,
        #     landing = (N_v−1)·dt_v/(n·DAY); win conventions bracket the wait
        f_forced_dt = (t_exp_uni + T_OH) * mode["N_exp"] / dt_v
        rows_b = {}
        for wtag, wflag in (("cons. wait (T=i·dt_v)", False),
                            ("opt. wait (T=(i−1)·dt_v)", True)):
            r = legacy_eval(mode, dt_v, f_forced_dt, win_iminus1=wflag)
            R_b = r["R"] * land_corr
            rows_b[wflag] = R_b
            print(f"    intranight: legacy(dt_v) × {land_corr:.4f}, "
                  f"{wtag:<24}{fmt(r['t_exp'], 2):>7}{fmt(R_b):>10}"
                  f"{fmt(R_b / R_mix, 2):>7}")
        lo_b, hi_b = sorted(rows_b.values())
        ok_bracket = lo_b <= dec["fresh"] * (1 + 1e-6) and \
            dec["fresh"] <= hi_b * (1 + 1e-6)
        checks.append((f"{name}: intranight recipe brackets the fresh term",
                       ok_bracket))
        print(f"      → brackets the fresh-pair term {fmt(dec['fresh'])}/yr: "
              f"[{fmt(lo_b)}, {fmt(hi_b)}]  geo-mean {fmt(math.sqrt(lo_b * hi_b))}")

        # (c) overnight-only: legacy at t_cad = n·DAY, unified budget
        f_forced_day = (t_exp_uni + T_OH) * mode["N_exp"] / mode["t_cad_s"]
        rows_c = {}
        for wtag, wflag in (("cons. wait (T=i·n·DAY)", False),
                            ("opt. wait (T=(i−1)·n·DAY)", True)):
            r = legacy_eval(mode, mode["t_cad_s"], f_forced_day,
                            win_iminus1=wflag)
            rows_c[wflag] = r["R"]
            print(f"    overnight-only: legacy(n·DAY), "
                  f"{wtag:<28}{fmt(r['t_exp'], 2):>7}{fmt(r['R']):>10}"
                  f"{fmt(r['R'] / R_mix, 2):>7}")
        if max(rows_c.values()) < dec["stale"]:
            print(f"      → neither endpoint reaches the gap-survivor term "
                  f"{fmt(dec['stale'])}/yr: the wait average spans the "
                  f"flux→cadence-limited transition inside the gap, which no "
                  f"single rectangle represents")

        dom_term = "intranight" if frac_fresh >= 0.5 else "overnight"
        verdicts[name] = (dom_term, frac_fresh, frac_fresh_c)

    # ── Context: pre-schedule encodings (main branch) ────────────────────────
    print("\n" + "=" * 78)
    print("CONTEXT: pre-schedule encodings (main branch), dominant mode, "
          "same conventions")
    print("=" * 78)
    for name, mode in zv.MODES_OLD_ENCODING.items():
        r = zv.eval_config(mode, zv.PHYS_NEW, full_integral=False)
        print(f"  {name.split(' ')[0]} pre-schedule (i={mode['i_det']}, "
              f"N_v={mode['N_v']}, f_eff={mode['f_eff']:.2f}): "
              f"R = {fmt(r['R'])}/yr  (× eps_cov {zv.EPS_COVERAGE} = "
              f"{fmt(r['R'] * zv.EPS_COVERAGE)}/yr)")
    print(f"  observed: public ≈ 2/yr, HC ≈ 2.5/yr "
          f"(raw model rates carry the known ×5–7 normalization excess; "
          f"eps_cov = {zv.EPS_COVERAGE} applies to every raw rate above)")

    # ── Sanity: schedule model at N_v = 1 equals the legacy model ────────────
    mode_a = next(iter(MODES.values()))
    m1 = dict(mode_a, N_v=1)
    r_sched1 = zv.eval_config(m1, zv.PHYS_NEW, full_integral=False)
    f_eq = m1["f_eff"] * F_NIGHT
    r_leg = legacy_eval(m1, m1["t_cad_s"], f_eq)
    checks.append(("N_v=1 schedule model = legacy model (parity spot check)",
                   abs(r_sched1["R"] - r_leg["R"])
                   < 1e-9 * max(r_leg["R"], 1e-30)))

    # ── Verdict per the single-dominant-term rule ────────────────────────────
    print("\n" + "=" * 78)
    print("VERDICT (single-dominant-term rule, user decision 2026-08-29)")
    print("=" * 78)
    for name, (term, frac, frac_c) in verdicts.items():
        print(f"  {name}: fresh-pair share = {frac:.0%} (cuts off) / "
              f"{frac_c:.0%} (id cuts on) → simple model = {term} term "
              f"(hard-threshold criterion)")

    print("\nSANITY CHECKS")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not all(ok for _, ok in checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
