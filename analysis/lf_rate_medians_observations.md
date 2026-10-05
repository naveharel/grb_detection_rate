# Observational benchmarks for the LF rate-median study

Checked on 2026-09-17 against the current project comparison and the primary
papers linked below. These benchmarks are deliberately approximate. A passing
LF establishes compatibility with these diagnostic tolerances, not a fitted
intrinsic luminosity function or a selection-matched population inference.

## Rates and search selection

The retained targets, 2/year for public and 2.5/year for high cadence, come from
[ztf_validation.py](ztf_validation.py) and the observational-target table in
[model_gap_report.md](model_gap_report.md). They are rounded project proxies
based on discovery counts and approximate observing modes, not independently
estimated, exposure-corrected rates with published uncertainty intervals. The
0.35 coverage multiplier is also a project assumption; the study keeps it fixed
and separately reports 0.2 and 0.5 sensitivities.

Ho et al. describe two public-survey discoveries and four discoveries in
high-cadence programs. Public coverage was about 15,000 square degrees every two
nights; high-cadence coverage about 2,500 square degrees with six visits/night.
Crucially, each public event had one g and one r alert, while high-cadence visits
were divided among filters. Thus two modeled detections separated by two days,
or six same-band detections per night, do not reproduce the discovery selection.
The search used rapid rise, color/fading, and host information, evolved over time,
and favored on-axis events. Its separate completeness-oriented test used 19,190
field-nights over 2020-2021: the simulated expectation was 1.04 events in two
years, with three observed. That expectation must not be substituted for either
rounded target above. [Ho et al. (2022), sections 2.1-2.2 and 4.1](https://arxiv.org/html/2201.12366v2)

The analysis preserves the existing comparison's two/six-detection requirements
and 0.3/0.5 mag/day fading/rising cuts to isolate the effect of adding an LF.
It does not recalibrate survey selection. Accordingly, disagreement cannot be
attributed uniquely to the LF, and agreement does not validate those settings.

## Distance proxies and sample membership

The 3.67/3.88-Gpc targets are the project's **comoving-distance** medians, not
luminosity distances and not numbers directly tabulated as survey medians by Ho
et al. They come from the distance-distribution assessment in
[model_gap_report.md](model_gap_report.md), whose underlying grouping is:

| Proxy group | Events and redshifts retained by the project | N | Median comoving distance |
|---|---|---:|---:|
| Low cadence, used for public | AT2019pim 1.2592; AT2021buv 0.876; AT2021lfa 1.0624; AT2023sva 2.280 | 4 | 3.67136 Gpc |
| High cadence | AT2020blt 2.9; AT2020kym 1.256; AT2021any 2.5131; AT2021qbd 1.1345; AT2023lcr 1.0272 | 5 | 3.88309 Gpc |

Recomputing those retained redshifts with flat Lambda-CDM, H0=70 km/s/Mpc and
Omega_m=0.3 reproduces the targets. Explicitly,

\[
D_C(z)=\frac{c}{H_0}\int_0^z
\frac{dz'}{\sqrt{\Omega_m(1+z')^3+1-\Omega_m}}.
\]

This check uses 128-point Gauss-Legendre quadrature; the search continues to use
the approved rounded targets 3.67 and 3.88. The low-cadence group combines ToO,
TESS, two-day public, and wide nightly discoveries, while the high-cadence group
includes a later 2023 event. Some events have no measured redshift; another
redshift-known event has unspecified cadence and is absent from these two groups.
The samples are tiny, heterogeneous, and not complete distance-selected samples.
Treat a 35% interval as a diagnostic tolerance, not a statistical confidence
interval. Small later redshift revisions do not justify silently changing the
approved targets.

The model uses one Euclidean distance for both inverse-square flux dilution and
volume. Real cosmology separates luminosity and comoving distances and introduces
redshift-dependent time and frequency factors. The comoving comparison follows
the existing project's volume-calibrated convention; it cannot simultaneously
make the flux interpretation exact. The older report's phase-dependent
flux-equivalent distances are a useful reminder that the yardstick itself is
model-dependent.

## Angles: diagnostics, not an observed hard threshold

Use q=theta_obs/theta_j with the study's fixed theta_j=0.1 rad. Then median
theta_obs=0.1*q_median rad, or approximately 5.72958*q_median degrees. Geometrical
on-axis means q<1; q<1.5 is an additional descriptive fraction, not a measured
observational classifier. In particular, the model's q_dec slightly above one
must not be relabeled as the strict jet edge.

Li et al. model AT2023lcr and three earlier orphan candidates. Classical on-axis
GRBs remain viable for AT2023lcr, AT2020blt, and AT2021any, but some fits also allow
off-axis or low-Lorentz-factor alternatives. AT2021lfa admits both an on-axis
low-Lorentz-factor origin and structured-jet off-axis solutions. Representative
off-axis fits have viewing/core-angle ratios about 2 for a Gaussian jet and 1.3
for a power-law jet, with model-dependent shortcomings. These individual,
degenerate fits provide no selection-corrected population q-median threshold.
Their structured-jet core angles are also not automatically equivalent to this
model's top-hat edge. [Li et al., sections 5.3-5.5 and 6](https://arxiv.org/html/2411.07973v2)

Consequently report the full requested angle summaries alongside the rate and
distance test. A concentration near the jet edge may be qualitatively plausible;
it does not establish that the predicted on-axis fraction agrees quantitatively
with observations. An LF rescales brightness while leaving the jet geometry and
light-curve shape fixed, so it need not solve an independent angle discrepancy.

## What the Euclidean wall does and does not establish

The model excludes every event beyond 4.55 Gpc, regardless of LF upper cutoff.
The retained proxy samples include comoving distances of about 5.56, 5.84, and
6.26 Gpc. No LF in the unchanged engine can reproduce that distant tail. Bright
objects eventually gain no additional volume from increased luminosity, so a
best candidate at a bright-luminosity boundary must be interpreted with care.

There is a stronger **model-dependent median ceiling** than the hard source wall.
Here the intrinsic density is uniform in Euclidean volume, the distance floor is
zero, and selection survival is non-increasing with distance at each luminosity
and viewing angle. The existing
[`_weighted_D_volume` implementation](../grb_detect/detection_rate.py#L1182)
documents the declining rise survival and distance-independent fading survival;
flux/window horizon indicators also only remove more distant sources. Averaging
over the fixed LF and orientation distribution preserves that monotonicity.
Consequently, writing the combined selection probability as S(D),

\[
P(D'\leq D\mid\mathrm{detected})
=\frac{\int_0^D D'^2 S(D')\,dD'}
       {\int_0^{D_{\rm Euc}} D'^2 S(D')\,dD'}
\geq\left(\frac{D}{D_{\rm Euc}}\right)^3,
\qquad
D_{\rm median}\leq 2^{-1/3}D_{\rm Euc}
=3.6113374\ \mathrm{Gpc}.
\]

The inequality follows because average selection in the inner volume is at
least its average across the entire volume. Thus even the largest possible
median in this fixed model is below both central proxies (3.67 and 3.88 Gpc).
Their 35% allowed intervals still extend comfortably below the ceiling, so this
does not preclude passing the approved approximate comparison. The ceiling
depends on uniform intrinsic volume density, zero distance floor, and monotone
selection; the simpler exclusion of every source beyond 4.55 Gpc depends only
on the hard wall. Neither result attributes every remaining distance deficit
uniquely to the wall. A cosmological model or different selection function is a
separate study; no engine change was made here.
