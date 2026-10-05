"""Generate all numerical LaTeX excerpts from the saved, reproducible results."""
from pathlib import Path
import json
import math

OUT = Path(__file__).resolve().parent / "results"


def sci(x, precision=3):
    if x == 0:
        return "$0$"
    e = math.floor(math.log10(abs(x)))
    return rf"${x/10**e:.{precision-1}f}\times10^{{{e}}}$"


def table(caption, headings, rows, cols=None):
    cols = cols or "l" + "r"*(len(headings)-1)
    text = [r"\begin{table}[H]\centering\small", r"\renewcommand{\arraystretch}{1.15}",
            r"\caption{"+caption+"}", r"\begin{tabular}{@{}"+cols+"@{}}", r"\toprule",
            " & ".join(headings)+r" \\\midrule"]
    text += [" & ".join(str(x) for x in row)+r" \\" for row in rows]
    text += [r"\bottomrule\end{tabular}\end{table}", ""]
    return "\n".join(text)


def write(name, text):
    (OUT / name).write_text(text+"\n", encoding="utf-8")


def main():
    th = json.loads((OUT / "theory.json").read_text())
    summary = json.loads((OUT / "ztf_summary.json").read_text())
    highlights = json.loads((OUT / "ztf_highlights.json").read_text())
    verify = json.loads((OUT / "ztf_verification.json").read_text())
    best = highlights["best_shared"]["result"]
    common_d = best["windows"]["conservative"]["modes"]["public"]["D_med_Gpc"]
    write("headline.tex", rf"""
For the retained ZTF benchmark, one shared pair, $\alpha={best['alpha']:.3f}$ and
$\mathcal A={best['A']:.3f}\,\mathrm{{Gpc^{{-3}}\,yr^{{-1}}}}$, passes the
previous study's factor-three rate and $35\%$ distance tolerances for both
survey modes and both timing prescriptions. It predicts a common distance
median of ${common_d:.3f}$ Gpc. This is an approximate diagnostic match;
the model cannot reproduce both central distance proxies exactly. Only
$25.7$--$31.9\%$ of these selected events have $q<1$, so the result also
does not establish a predominantly geometrically on-axis detected population.
""")
    cases = th["cases"]
    cont = [c for c in cases if c["prescription"] == "continuous"]
    text = table("Full-time results for the thesis shape. Luminosity medians are global, after angular mixing.",
                 [r"$\alpha$", r"$J_\alpha$", r"$q_{\rm med}$", r"$L_{\rm med}/L_{\rm dec}$",
                  r"$D_{\rm med}/D_\Euc$", r"$P(q\le q_\dec)$"],
                 [[f"{c['alpha']:g}", f"{c['J']:.6g}", f"{c['q_median']:.4f}",
                   sci(c["L_median_over_Ldec"]), f"{c['D_med_over_DE']:.5f}",
                   f"{100*c['fraction_q_le_qdec']:.2f}\\%"] for c in cont])
    text += r"""
At $\alpha=-2$, the full-time core holds nearly $97\%$ of the selected
rate, so its angular approximation is accurate. However, a similarly dominant
angular core at $\alpha=-2.4$ does not place the luminosities near $L_{\rm dec}$:
their global median is only about $0.002L_{\rm dec}$. At $\alpha=-1.25$,
the global median exceeds $L_{\rm dec}$ by nearly six orders of magnitude.
The latter shift combines the bright conditional tail with larger angular wall
luminosities. Thus a conditional wall mode is not a universally representative
luminosity for the full population.
"""
    text += table("Local luminosity concentration and two distinct angular approximations, full-time selection.",
                  [r"$\alpha$", r"Local decade fraction", r"Global decade fraction", r"Core/full", r"Moving corner/full"],
                  [[f"{c['alpha']:g}", f"{100*d['fraction_within_decade_local_wall']:.2f}\\%",
                    f"{100*c['fraction_L_within_decade_reference']:.2f}\\%",
                    f"{c['angular_box_over_exact']:.5f}", f"{c['integrated_thesis_corner_over_exact']:.5f}"]
                   for c,d in zip(cont, th["conditional_concentration"])])
    text += r"The local fraction uses $[0.1L_{\rm wall}(q),10L_{\rm wall}(q)]$ at each angle; the global fraction uses the single interval $[0.1L_{\rm dec},10L_{\rm dec}]$."+"\n"
    timing = [c for c in cases if c["alpha"] == -2 and c["prescription"] != "continuous"]
    names = {"legacy": "Thesis step", "peak_step": "Peak step", "random_phase": "Random phase"}
    text += table(r"Timing tests at $\alpha=-2$. The luminosity reference is $L_i=L_{\rm dec}/\widetilde F_\nu(i t_{\rm cad})$ even when a different selection is evaluated. The last column isolates the Newtonian-tail effect.",
                  [r"$t_{\rm cad}$ [d], $i$", "Selection", r"$R/R_{\rm thesis}$", r"$q_{\rm med}$",
                   r"$L_{\rm med}/L_i$", r"$R_{\rm IV}/R_{\rm III\ ext.}$"],
                  [[f"{c['cadence_days']:g}, {c['i_det']}", names[c["prescription"]],
                    f"{c['rate_over_legacy']:.5f}", f"{c['q_median']:.4f}",
                    f"{c['L_median_over_reference']:.5f}", f"{c['rate_IV_over_extrapolated_III']:.6f}"]
                   for c in timing], "llrrrr")
    text += r"""
For two visits at two-day cadence, retaining the peak and averaging phase
raises the rate by $14.09\%$ relative to the thesis step. For ten visits at
the same cadence, it instead lowers it by $48.46\%$. These are different
selection approximations, not statistical uncertainty intervals. The long-window
stress case, 300 visits at one-day cadence, exposes an $11.86\%$ effect from
including phase IV rather than continuing phase III. The corresponding effect
for the two-visit random-phase case at two-day cadence is below $10^{-8}$.

The separate single-visit test gives a periodic-to-continuous rate ratio
of $0.00158827$ at one-day cadence and $\alpha=-2$. Reducing cadence to
$10^{-4}$ s gives $0.99999715$. The one-day result has independent phase
validation; short cadence tests the continuous limit.

Extreme quantiles are evaluated logarithmically. For example,
$\alpha=-1.001$ gives, for continuous selection,
\[\log_{10}(L_{\rm med}/L_{\rm dec})=309.5692.\]
The median is mathematically finite, but exceeds ordinary
floating-point range and has no plausible interpretation as a physical GRB
luminosity. This is a numerical and physical warning against using a
nearly nonconvergent unbounded intensity as a complete population model.
"""
    write("theory_tables.tex", text)

    windows = [("conservative", "Conservative"), ("optimistic", "Optimistic")]
    modes = [("public", "Public"), ("high_cadence", "High cadence")]
    rows=[];angle_rows=[];lum_rows=[]
    for w, wn in windows:
        for m, mn in modes:
            s = best["windows"][w]["modes"][m]
            target = 2 if m == "public" else 2.5
            rows.append([mn, wn, f"{s['raw_rate']:.3f}", f"{s['rate']:.3f}",
                         f"{s['rate']/target:.3f}", f"{s['q_med']:.4f}"])
            angle_rows.append([mn, wn, f"{s['theta_med_deg']:.3f}",
                               *[f"{100*s[k]:.2f}\\%" for k in ["frac_q_lt_1", "frac_q_lt_1p5", "frac_q_gt_2"]]])
            lum_rows.append([mn, wn, sci(s["L_med_spectral"]), sci(s["L_med_1day_erg_s"])])
    text=rf"""
\subsection{{Shared calibration and results}}
The scan contains {summary['scan']['count']:,} slopes from $-2.499$ to $-1.001$
at spacing 0.001. Its best shared compromise is
\begin{{equation}}
\boxed{{\alpha={best['alpha']:.3f},\qquad
\mathcal A={best['A']:.5f}\,\mathrm{{Gpc^{{-3}}\,yr^{{-1}}}}.}}
\end{{equation}}
The minimized joint score is {best['score']:.6f}, below the acceptance value one.
The common median distance is ${common_d:.6f}$ Gpc, respectively $21.71\%$
and $25.94\%$ below the public and high-cadence proxies. Rate discrepancies
remain as large as a factor $2.258$; passing the loose benchmark is not agreement
with its central values.
"""
    text+=table(r"Shared-LF rates. Raw rates include the optical-night convention; effective rates additionally include coverage 0.35. Rates are in $\mathrm{yr^{-1}}$.",
                ["Survey", "Window", "Raw", "Effective", "Rate/target", r"$q_{\rm med}$"],rows,"llrrrr")
    text+=table(r"Viewing-angle diagnostics of the same shared model. Physical angles are in degrees; geometrical on-axis means $q<1$.",
                ["Survey", "Window", r"$\theta_{\rm med}$", r"$q<1$", r"$q<1.5$", r"$q>2$"],angle_rows,"llrrrr")
    text+=table(r"Detected luminosity medians, with the two luminosity definitions kept explicit.",
                ["Survey", "Window", r"$L_\nu(t_{\rm dec})$ [erg s$^{-1}$ Hz$^{-1}$]",
                 r"$\nu L_\nu(1\,\mathrm{d})$ [erg s$^{-1}$]"],lum_rows,"llrr")
    text+=r"""
The public-mode luminosity median is substantially higher than the high-cadence
one even though their distance medians are identical: the longer time between
required public visits selects intrinsically brighter afterglows. These
luminosity medians are predictions, not additional calibration targets.
All four detected means diverge in the ideal model at the selected slope;
the quoted medians remain finite.

\subsection{Accepted slopes, endpoint alternatives, and coverage}
"""
    lo,hi=summary["shared_accepted_alpha_range"]
    text+=rf"At 0.001 resolution, shared calibrated slopes ${lo:.3f}\le\alpha\le{hi:.3f}$ pass both endpoints. Neighboring slopes $-1.913$ and $-1.351$ fail. This is a tolerance-defined feasible interval, not a confidence interval. The lower boundary follows the analytic distance constraint; the upper boundary comes from incompatible rate ratios. The $\alpha=-2$ model fails because its common median is only 2.275 Gpc, regardless of amplitude."+"\n"
    fitrows=[]
    for key,name in [("best_shared","Both endpoints"),("best_conservative","Conservative only"),("best_optimistic","Optimistic only")]:
        r=highlights[key]["result"]
        optscore=highlights[key].get("optimized_score",r["score"])
        fitrows.append([name,f"{r['alpha']:.3f}",f"{r['A']:.4f}",
                        f"{r['windows']['conservative']['modes']['public']['D_med_Gpc']:.4f}",f"{optscore:.5f}"])
    text+=table(r"Endpoint-specific fits use their own shared two-survey amplitude; they are separate alternatives, not a replacement of the four-case calibration. $\mathcal A$ has units $\mathrm{Gpc^{-3}\,yr^{-1}}$.",
                ["Calibration target",r"$\alpha$",r"$\mathcal A$",r"$D_{\rm med}$ [Gpc]","Target score"],fitrows)
    covrows=[]
    for cov, entry in highlights["best_shared"]["coverage"].items():
        covrows.append([cov,*[f"{entry[w]['rates'][m]:.3f}" for w,_ in windows for m,_ in modes],
                        "Both" if all(entry[w]["passes"] for w,_ in windows) else
                        "Conservative" if entry["conservative"]["passes"] else "Neither"])
    text+=table(r"Coverage sensitivity at the selected, fixed $\mathcal A$ and $\alpha$. C/O denote conservative/optimistic windows; rates are effective $\mathrm{yr^{-1}}$.",
                ["Coverage","Public C","HC C","Public O","HC O","Passing endpoints"],covrows,"rrrrrl")
    write("ztf_tables.tex", text)
    write("conclusion.tex",rf"""
For the existing two-survey benchmark the simplified model provides a useful
two-parameter alternative: $\alpha={best['alpha']:.3f}$ and
$\mathcal A={best['A']:.3f}\,\mathrm{{Gpc^{{-3}}\,yr^{{-1}}}}$ meet the
retained loose rate and distance criteria under both endpoint prescriptions.
Its independence of distance distribution from survey selection is an especially
clear, testable prediction. The fit retains a rate discrepancy of about a factor
2.26 and a distance deficit as large as 25.94 percent, and does not establish
agreement of the angular or luminosity distributions with observations.

The general simplification is therefore supported for this diagnostic use,
while a universal statement that contributions are confined near $q=1$ and
$L=L_{{\rm dec}}$ is not. The numerical examples and remote-tail tests identify
where that stronger interpretation fails.
""")
    tv=th["verification"]
    checks=verify["engine_checks"]
    if not checks:
        raise ValueError("Engine comparisons must finish before building final report tables")
    rows=[['Analytic theory test groups',str(tv['test_groups_run']),"All pass"],
          ['Theory resolution cases',str(len(tv['resolution_checks'])),"All pass"],
          ['ZTF resolution cases',str(len(verify['resolution_doubling'])),"All pass"],
          ['Finite-cutoff engine comparisons',str(len(checks)),"All pass"],
          ['Theory maximum rate resolution change',sci(tv['maximum_rate_resolution_error']),r"$<10^{-7}$"],
          ['Engine maximum relative rate difference',f"{100*max(c['errors']['rate'] for c in checks):.5f}\\%",r"$<1\%$"],
          ['Engine maximum relative distance-median difference',f"{100*max(c['errors']['D_median'] for c in checks):.5f}\\%",r"$<1\%$"],
          ['Engine maximum absolute angular-median difference',f"{max(c['errors']['q_median'] for c in checks):.6f}",r"$<0.01$"],
          ['Engine luminosity CDF error at analytic median',f"{max(c['errors']['luminosity_median_cdf'] for c in checks):.6f}",r"$<0.01$"]]
    text=table("Numerical validation summary. Analytic-controlled tolerances and engine-grid tolerances are distinguished.",
               ["Check","Result","Acceptance"],rows,"lrl")
    text+=r"""
The ZTF standalone resolution test doubles angular and phase quadrature from
96/32 to 192/64 nodes per segment. The independent engine comparisons double
the engine grids from 2000/1000 to 4000/2000 angular/distance points, compare
rates and normalized distributions at identical finite bounds, and explicitly
check the luminosity CDF at the independently computed median. These tests
use both $\alpha=-2$ and the selected slope, both survey modes, both endpoints,
and spectral bounds $[10^{26},10^{40}]$ and $[10^{24},10^{42}]$ in
$\mathrm{erg\,s^{-1}\,Hz^{-1}}$. All comparisons pass.
"""
    tailrows=[]
    for c in checks:
        if c['alpha']==best['alpha'] and c['window']=='conservative':
            tailrows.append(["Public" if c['mode']=='public' else 'High cadence',
                             f"{c['spectral_log10_bounds'][0]:g}, {c['spectral_log10_bounds'][1]:g}",
                             sci(c['analytic']['omitted_faint']),sci(c['analytic']['omitted_bright'])])
    text+=table(r"Omitted fractions of the infinite selected rate at the best shared slope, conservative endpoint. Bounds are $\log_{10}(L/[\mathrm{erg\,s^{-1}\,Hz^{-1}}])$; the differential amplitude is held fixed.",
                ["Survey","Lower, upper bound","Faint fraction","Bright fraction"],tailrows,"llrr")
    tailrows=[[f"{r['alpha']:g}",f"{math.log10(r['Llow_over_Ldec']):g}, {math.log10(r['Lhigh_over_Ldec']):g}",
               f"{r['retained_rate_fraction']:.6f}",sci(r['faint']),sci(r['bright'])]
              for r in th['truncation'] if r['Llow_over_Ldec']==1e-12]
    text+=table(r"Controlled continuous-selection cutoff sensitivity. Even bounds twelve decades below and twenty-two above $L_{\rm dec}$ need not capture the entire rate near a convergence boundary.",
                [r"$\alpha$",r"$\log_{10}(L_{\min,\max}/L_{\rm dec})$","Retained","Faint missing","Bright missing"],tailrows,"rlrrr")
    boundaryrows=[]
    for alpha in [-3,-2.5,-1,-.5]:
        r=[r for r in th['boundary_divergence'] if r['alpha']==alpha]
        boundaryrows.append([f"{alpha:g}",*[sci(v['dimensionless_finite_detection_integral']) for v in r]])
    text+=table(r"Divergence outside the admissible slope interval: dimensionless finite selected integrals at fixed amplitude with bounds $[10^{-n},10^{n+10}]L_{\rm dec}$. Boundary slopes grow logarithmically; the outside slopes grow as powers.",
                [r"$\alpha$",r"$n=4$",r"$n=8$",r"$n=12$"],boundaryrows)
    write("verification_tables.tex",text)
    print("Built five numerical LaTeX includes from saved JSON results.")


if __name__ == "__main__":
    main()
