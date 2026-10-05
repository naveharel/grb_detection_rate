# Does the luminosity function resolve the rate–median tension?

**It resolves the rate–distance tension to the requested approximate tolerances,
but agreement in viewing angles is not established.** One shared LF passes both
survey comparisons under both detection-window prescriptions. The default LF
helps substantially, but does not pass the distance criteria. The successful LF
makes the angular distribution somewhat more off-axis, rather than fixing it.

This is a fresh calculation on `main` at `92a4710`, using the unchanged engine.
The analysis implements the [approved plan](lf_rate_medians_plan.md). Rates and
distances below are comparisons with project benchmarks, not a statistical fit
to a uniformly selected observational sample; see the
[source and selection audit](lf_rate_medians_observations.md).

## A shared LF that works for rates and distances

The best sampled compromise across both window endpoints is

\[
\frac{dN}{dL}\propto L^{-1.75},\qquad
L_{\min}=10^{42.25}=1.78\times10^{42}\ {\rm erg\,s^{-1}},\qquad
L_{\max}=10^{46.25}=1.78\times10^{46}\ {\rm erg\,s^{-1}},
\]

where **L means intrinsic isotropic-equivalent nu L_nu at one day**, at the
model frequency, not the luminosity at deceleration. These are intrinsic LF
bounds, not the bounds or median of the detected sample. The app settings are
alpha = -1.75, log10 L_min = 42.25, log10 L_max = 46.25.

| Survey | Window | Effective rate [year^-1] | Median distance [Gpc] | Rate / target | Distance deficit |
|---|---|---:|---:|---:|---:|
| Public | Conservative | 2.228 | 2.696 | 1.11 | 26.5% |
| Public | Optimistic | 4.772 | 2.795 | 2.39 | 23.8% |
| High cadence | Conservative | 1.226 | 2.869 | 0.49 | 26.1% |
| High cadence | Optimistic | 1.337 | 2.875 | 0.53 | 25.9% |

The targets are 2 and 2.5/year, and 3.67 and 3.88 Gpc. Every row satisfies
the approved factor-three rate and 35% distance tolerances. In particular,
the high-cadence rate is still approximately a factor of two low: this is an
approximate match under the chosen criteria, not a match to all central values.

The conservative and optimistic windows require emission after the peak for
i times the cadence and (i-1) times the cadence, respectively. They are alternative
prescriptions, not statistical error bars. Each assessment uses the same
prescription for both surveys. Effective rates include the fixed 0.35 coverage
factor; raw model rates are available in the numerical results.

## What changed relative to the default model?

| Survey | Luminosity model | Effective rate [year^-1] | Median distance [Gpc] | Median q |
|---|---|---:|---:|---:|
| Public | Single default luminosity | 0.986–4.702 | 0.556–0.997 | 1.171–1.231 |
| Public | Default LF | 0.743–2.416 | 1.732–2.165 | 1.216–1.245 |
| Public | Shared LF above | 2.228–4.772 | 2.696–2.795 | 1.288–1.354 |
| High cadence | Single default luminosity | 3.764–4.622 | 2.271–2.471 | 1.065–1.087 |
| High cadence | Default LF | 0.915–1.032 | 2.433–2.457 | 1.165–1.186 |
| High cadence | Shared LF above | 1.226–1.337 | 2.869–2.875 | 1.250–1.273 |

Ranges show the minimum and maximum across window prescriptions. The default
LF has alpha=-2 and luminosity bounds 10^42.5–10^45.5 erg/s. Its rates already
pass the factor-three test; its distance medians do not pass the 35% test.

The distribution allows a relatively faint intrinsic population to coexist with
a brighter detected subset at larger distances. This removes the single-value
requirement that raising the characteristic distance must also raise the rate
of every burst. The selected LF is an existence example of this mechanism, not
a measurement of the intrinsic LF or a proposal to change the app defaults.

## Viewing angles remain a separate issue

Here q = theta_obs / theta_j and theta_j = 0.1 rad = 5.73 degrees. For the shared LF:

| Survey | Window | Median q | Median theta_obs | q < 1 | q < 1.5 | q > 2 |
|---|---|---:|---:|---:|---:|---:|
| Public | Conservative | 1.354 | 7.76 degrees | 27.5% | 60.4% | 19.8% |
| Public | Optimistic | 1.288 | 7.38 degrees | 30.3% | 66.2% | 12.3% |
| High cadence | Conservative | 1.273 | 7.29 degrees | 31.2% | 65.3% | 11.4% |
| High cadence | Optimistic | 1.250 | 7.16 degrees | 32.3% | 67.0% | 10.8% |

These medians are close to, but outside, the sharp top-hat jet edge. Most
predicted detections are **not geometrically on-axis**. The bright LF tail also
makes larger viewing angles detectable; compared with the single default
luminosity, the on-axis fraction falls and the off-axis tail grows.

Among sampled LFs that pass both endpoints, the candidate maximizing the worst
on-axis fraction across the four cases is (-1.75, 42.25, 45.75). It has
30.3–33.1% inside the jet and 8.6–13.5% at q>2. Its conservative public distance
is 2.397 Gpc, only just above the 2.3855-Gpc acceptance floor. Thus improving the
angles within this search costs some of the distance improvement.

Published individual-event fits allow on-axis and off-axis alternatives and do
not supply a reliable, selection-corrected population median q. Structured-jet
core angles also differ from this model's top-hat edge. Consequently these
near-edge medians are not automatically excluded, but the LF has **not** shown
that its angular population agrees with observations. In particular, q<1.5
must not be called a measured on-axis classification.
[Li et al.](https://arxiv.org/html/2411.07973v2)

## Search, sensitivity, and limits

The search evaluated 1,440 coarse and 715 additional refined parameter triples.
At screening resolution, 35 pass both endpoints, 116 pass one, and 2,004 fail.
These counts are grid results, not probabilities or confidence regions; highlighted
examples alone were recomputed at the final resolution. Five both-endpoint
screening matches lie on a search-domain boundary. The selected shared LF lies
inside all boundaries, so the demonstrated match does not require an extreme
slider endpoint. The best sampled conservative-only LF, (-1.625, 42.0, 47.25),
achieves distances near 3.0 Gpc but gives an optimistic public rate of 6.716/year,
above the allowed 6/year.

Two converged near-misses make the threshold explicit: (-2, 43.25, 47.0) gives
a conservative public distance of 2.380 Gpc; (-2, 42.75, 46.5) gives an optimistic
public distance of 2.379 Gpc. Both sit just below the 2.3855-Gpc floor. Their
classification should not be interpreted as a sharp astrophysical exclusion.

Coverage sensitivity of the selected shared LF, with medians unchanged:

| Coverage | Public rate, conservative / optimistic | High-cadence rate, conservative / optimistic | Joint outcome |
|---:|---:|---:|---|
| 0.20 | 1.273 / 2.727 | 0.701 / 0.764 | Neither endpoint: high-cadence rate too low |
| 0.35 | 2.228 / 4.772 | 1.226 / 1.337 | Both endpoints pass |
| 0.50 | 3.183 / 6.818 | 1.752 / 1.910 | Conservative passes; optimistic public rate too high |

Thus the existence example is conditional on the assumed survey efficiency.
No coverage parameter was fitted.

Two limitations prevent calling this a solved observational model:

- The rate settings approximate ZTF discovery selection. In the actual search,
  public events had one g and one r alert, and six high-cadence visits were
  divided among filters. The distance groups mix discovery strategies and have
  only four/five events. Agreement with these proxies needs a future comparison
  that reproduces the actual selection. [Ho et al.](https://arxiv.org/html/2201.12366v2)
- No LF can put a source beyond the engine's 4.55-Gpc wall, whereas the retained
  samples include comoving distances up to 6.26 Gpc. Moreover, for this uniform
  Euclidean volume with zero distance floor and selection decreasing with
  distance, the median cannot exceed 4.55 times 2^(-1/3) = **3.611 Gpc**.
  Even exact matching of the 3.67/3.88-Gpc central medians is therefore
  impossible under these assumptions. This does not invalidate the approved
  approximate match, or imply that the wall explains the entire remaining
  deficit. The model also uses the same distance for volume and flux dilution,
  unlike cosmology.

## Reproduction and numerical verification

Run from the repository root:

```powershell
.venv/Scripts/python.exe -B analysis/lf_rate_medians.py --workers 4
.venv/Scripts/python.exe -B -m pytest analysis/test_lf_rate_medians.py tests/test_luminosity_function.py -p no:cacheprovider -q
```

The script uses the existing full-integral APIs, the correct night model and
night-rate factor for high cadence, peak-based windows, and shared random-start
rise/fade survival. Geometry, physical times, normalization, and survey settings
are fixed to `PHYS_NEW`, `BASE_PARAMS`, and `MODES` in the older validation script,
with fading/rising cuts of 0.3/0.5 mag/day. It computes medians from normalized
marginal distributions, never averages individual luminosity medians.

The rate screen uses 500 angular points and a 1% boundary margin; surviving
distance distributions use 400 distance points. Refinement uses half the coarse
spacing around the ten best shared candidates and the ten best for each endpoint.
The ranking is the maximum of abs(ln(rate/target))/ln(3) and
abs(median_distance/target-1)/0.35; shared ranking takes the maximum over both
endpoints. It is a diagnostic score, not a likelihood. The finite grid establishes
existence and tradeoffs, not a global continuous optimum or uniqueness.

All eight highlighted configurations were checked at 1,500/1,000 angular/distance
points and then 3,000/2,000. Maximum changes were **0.118% in rates**, **0.034% in
distance medians**, and **0.00088 in median q**. Marginal rate integrals agree
within **0.131%**; angle-integrated rates agree with the scalar API within
0.00058%. These satisfy every requested numerical tolerance.

Fresh tests also pass: **22 analysis checks** (10.92 s) and **38 engine LF tests**
(61.85 s), run separately with bytecode/cache writing disabled. The analysis
checks include independent baseline values, collapsed-LF equivalence to a
single-luminosity model, CDF geometry, the distance ceiling, and coverage
classification. This is targeted verification, not a claim that the repository's
entire test suite passes.

Machine-readable outputs in [lf_rate_medians_results/](lf_rate_medians_results/):

- `scan.csv`, `coarse.json`, and `refined.json`: all search cases, with blank
  medians for cases rejected by the rate screen.
- `highlights.csv` and `highlights.json`: verified examples, angle fractions,
  convergence checks, and coverage sensitivities.
- `summary.json` and `metadata.json`: counts, selected parameters, fixed inputs,
  software versions, engine commit, and source hashes.

The engine, bridge, app, and defaults were not changed.
