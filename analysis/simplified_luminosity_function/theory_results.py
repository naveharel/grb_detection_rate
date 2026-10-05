"""Reproduce controlled theory tables and verification outcomes using NumPy only.

Run: .venv/Scripts/python.exe -B -m analysis.simplified_luminosity_function.theory_results
"""
from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import unittest

import numpy as np

from . import theory as th
from .test_theory import TheoryTests

OUT = Path(__file__).parent/"results"
SLOPES = [-2.4, -2, -1.75, -1.25, -1.05]


def case(alpha, prescription, cadence_s=None, i_det=2, late_time=True, n=128):
    s = th.DEFAULT_SHAPE
    kw = dict(prescription=prescription, cadence_s=cadence_s, i_det=i_det,
              late_time=late_time, n=n)
    J = th.angular_moment(alpha, **kw)
    qmed = th.angular_median(alpha, **kw)
    lmed = th.luminosity_median(alpha, **kw)
    reference = (1.0 if prescription == "continuous" else
                 1/float(s.lightcurve(max(s.t_dec_s,i_det*cadence_s),late_time)))
    cdf = lambda q: th.angular_cdf(q,alpha,**kw)
    lcdf = lambda L: th.luminosity_cdf(L,alpha,**kw)
    qvalues = sorted(set(list(np.geomspace(1e-4,s.q_nr,4000))+
                         th.q_bounds(s,cadence_s,i_det,prescription,late_time=late_time)))
    qvalues = np.array([q for q in qvalues if q>0])
    log_density = qvalues*th.moment_density(qvalues,alpha,cadence_s,i_det,prescription,late_time)
    row = dict(alpha=alpha, prescription=prescription, cadence_days=cadence_s/th.DAY_S if cadence_s else None,
               i_det=i_det, late_time_IV=late_time, J=J,
               rate_A1_D528_F30microJy=th.rate(1,alpha,1.63e28/th.GPC_CM,3e-5,J),
               D_med_over_DE=th.distance_median(alpha),q_median=qmed,
               q_log_density_mode_sampled=float(qvalues[np.argmax(log_density)]),
               L_median_over_Ldec=lmed,log10_L_median_over_Ldec=math.log10(lmed),
               L_reference_over_Ldec=reference,
               L_median_over_reference=lmed/reference,
               fraction_L_below_reference=lcdf(reference),
               fraction_L_within_decade_reference=lcdf(10*reference)-lcdf(.1*reference),
               fraction_q_lt_1=cdf(1),fraction_q_lt_1p5=cdf(1.5),
               fraction_q_gt_2=1-cdf(2),fraction_q_le_qdec=cdf(s.q_dec))
    if prescription in ("continuous","legacy"):
        row.update(th.corner_ratios(alpha,cadence_s,i_det,prescription,late_time))
    return row


def write_csv(name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open("w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    s = th.DEFAULT_SHAPE
    cases = []
    convergence = []
    for alpha in SLOPES:
        configs = [("continuous",None,2)]
        configs += [(p,2*th.DAY_S,i) for p in ("legacy","peak_step","random_phase") for i in (2,10)]
        configs += [(p,th.DAY_S,300) for p in ("legacy","peak_step","random_phase")]
        for p,cad,i in configs:
            row = case(alpha,p,cad,i)
            high = case(alpha,p,cad,i,n=256)
            checks = dict(alpha=alpha,prescription=p,cadence_days=cad/th.DAY_S if cad else None,
                          i_det=i, J_relative_change=abs(row["J"]/high["J"]-1),
                          L_median_relative_change=abs(row["L_median_over_Ldec"]/high["L_median_over_Ldec"]-1),
                          q_median_absolute_change=abs(row["q_median"]-high["q_median"]))
            checks["passes"] = (checks["J_relative_change"]<1e-7 and
                checks["L_median_relative_change"]<.01 and checks["q_median_absolute_change"]<.01)
            convergence.append(checks)
            if p != "continuous":
                legacy = th.angular_moment(alpha,cad,i,"legacy")
                row["rate_over_legacy"] = row["J"]/legacy
                no_IV = th.angular_moment(alpha,cad,i,p,late_time=False)
                row["rate_IV_over_extrapolated_III"] = row["J"]/no_IV
            cases.append(row)
    tails = []
    for alpha in SLOPES:
        for decades in [4,8,12]:
            low,high = 10.0**(-decades),10.0**(decades+10)
            tail = th.omitted_tail_fractions(low,high,alpha)
            retained = th.finite_rate_ratio(low,high,alpha)
            tails.append(dict(alpha=alpha,Llow_over_Ldec=low,Lhigh_over_Ldec=high,
                              retained_rate_fraction=retained,**tail,
                              accounting_absolute_error=abs(1-retained-tail["total"])))
    boundaries=[]
    for alpha in [-3,-2.5,-1,-.5]:
        for span in [4,8,12]:
            low,high=10.0**(-span),10.0**(span+10)
            value=th.gauss_integral(lambda q:q*th.finite_luminosity_weight(
                low,high,s.peak_shape(q),alpha),[0,s.q_dec,2,s.q_nr],128)
            boundaries.append(dict(alpha=alpha,Llow_over_Ldec=low,Lhigh_over_Ldec=high,
                dimensionless_finite_detection_integral=value))
    # Conditional concentration is universal even though global L is an angular mixture.
    conditional=[]
    for alpha in SLOPES:
        b,k=-alpha-1,alpha+2.5
        med=(3/(4*b))**(1/k) if b>=.75 else (4*k/3)**(1/b)
        conditional.append(dict(alpha=alpha,median_L_over_local_wall=med,
            fraction_below_local_wall=2*b/3,
            fraction_within_decade_local_wall=float(th.conditional_luminosity_cdf(10,alpha)-
                th.conditional_luminosity_cdf(.1,alpha))))
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(TheoryTests))
    verification=dict(test_groups_run=result.testsRun, failures=len(result.failures),
                      errors=len(result.errors),passed=result.wasSuccessful(),
                      output=stream.getvalue(),resolution_checks=convergence,
                      all_resolution_checks_pass=all(c["passes"] for c in convergence),
                      maximum_rate_resolution_error=max(c["J_relative_change"] for c in convergence),
                      maximum_L_median_resolution_error=max(c["L_median_relative_change"] for c in convergence),
                      maximum_q_median_resolution_error=max(c["q_median_absolute_change"] for c in convergence))
    verification["extreme_quantiles"]=[dict(alpha=-1.001,prescription=p,
        log10_L_median_over_Ldec=th.log10_luminosity_median(-1.001,
            prescription=p,cadence_s=2*th.DAY_S,i_det=2),
        ordinary_float_result="positive_infinity",
        interpretation="Finite mathematical median exceeds floating-point range; use logarithm")
        for p in ("continuous","random_phase")]
    verification["single_visit_limit"]=dict(alpha=-2,
        one_day_phase_over_continuous=th.phase_averaged_moment(-2,th.DAY_S,1)/th.angular_moment(-2),
        cadence_1e_minus_4_s_phase_over_continuous=th.phase_averaged_moment(-2,1e-4,1)/th.angular_moment(-2),
        independent_two_dimensional_check_passed=result.wasSuccessful())
    payload=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        model=asdict(s),derived=dict(t_j_days=s.t_j_s/th.DAY_S,t_nr_days=s.t_nr_s/th.DAY_S,
                                   q_dec=s.q_dec,q_nr=s.q_nr),
        conventions=dict(L_ref_erg_s_Hz=th.L_REF,A_units="Gpc^-3 yr^-1 per natural-log L",
            luminosity="L_nu(t_dec)",flux_limit_Jy=3e-5,D_euc_cm=1.63e28,
            note="Improper intrinsic intensity; only the selected population is normalized.",
            Newtonian_slope="(21-15*p)/10",angular_cap="sqrt(2)/theta_j",
            rate_reference="Full sky, A=1; no survey sky fraction or identification cuts",
            random_phase="Peak-retained periodic visits; same uniform phase sets count threshold",
            L_reference="L_dec continuous; L_i=L_dec/g(i*t_cad) for cadence cases"),
        cases=cases,conditional_concentration=conditional,truncation=tails,
        boundary_divergence=boundaries,verification=verification)
    (OUT/"theory.json").write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    write_csv("theory_cases.csv",cases)
    write_csv("theory_truncation.csv",tails)
    write_csv("theory_boundaries.csv",boundaries)
    write_csv("theory_resolution.csv",convergence)
    print(json.dumps(dict(cases=len(cases),verification_passed=verification["passed"],
        all_resolution_checks_pass=verification["all_resolution_checks_pass"],
        output=str(OUT/"theory.json"))))
    if not verification["passed"] or not verification["all_resolution_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
