# Approved app integration plan

Approved 2026-10-04. Replace the app LF setting with dR/dL = A/L_ref (L/L_ref)^alpha for every positive L, with L = L_nu(t_dec), L_ref = 1e32 erg/s/Hz, alpha = -2 and A = 1 Gpc^-3 yr^-1 by default. LF stays off initially. Require -2.5 < alpha < -1. The luminosity is defined at the selected observing band.

## Calculation

- Introduce a distinct scale-free LF configuration and lf_log10_A factory/bridge parameter, independent of the single-luminosity volumetric rate.
- Exact off uses analytic most-contributing rectangles integrated over all luminosities; exact on integrates luminosity and distance analytically and the remaining angle numerically.
- Preserve existing light curves, cadence/day-night dispatch, peak-window and joint rise/fade selection.
- All outputs, distributions and medians follow the selected calculation mode. Approximate cumulative queries intersect a fixed base rectangle selection. Use analytic marginals and log-distance median inversion.
- Reuse normalized-distribution calculations across equal cadence strategies. Retain finite-cutoff calculations only through an explicit Python reference path; update research callers.

## Interface

- Keep the LF Settings switch and dependent Parameters block. Replace bounds with alpha [-2.49, -1.01], step .001, and log10 A [-3, 3], step .05. Explain A per natural-log luminosity and display fixed L_ref.
- Gray and truly disable rho, nu, epsilon_e, epsilon_B and the F_dec override while LF is active; restore saved values and behavior on exit.
- Hide undefined intrinsic-rate/population-median and obsolete brightness-reference readouts. Keep Exact selectable.
- Normalize q/D plot percentages by actual totals and disclose appreciable mass below the displayed range.
- Update project/UI implementation notes with explicit notation, without historical LF comparisons in app copy. Preserve scientific reports and unrelated local edits.

## Acceptance

- Record baseline. Verify smooth full-integral quadrature agreement at 1e-7, rate scalings, inactive-control and reference-luminosity invariance.
- Cover convergence boundaries, representative slopes, jet/cadence transitions, rise/fade and peak windows, nonzero q/D floors and empty selections.
- Check CDF normalization/monotonicity, densities and selected-mode medians, including extremely small distances.
- Compare finite-cutoff reference calculations at fixed differential normalization with quantified omitted tails (within 1% once tails/numerical error resolved).
- Rebuild standalone HTML and verify the running app controls, restoration, plots, optimization, presets and LF-off parity in a browser.
