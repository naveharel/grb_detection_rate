# GRB Afterglow Physics Reference

Source of truth: Bachelor's thesis "Detection Rates of GRB Afterglows in Optical Surveys — An Analytic Approach" + Hebrew presentations.

## Afterglow phases

| Phase | Range | Description |
|-------|-------|-------------|
| I  | t < t_dec | Coasting (pre-deceleration) |
| II | t_dec – t_j | Relativistic deceleration |
| III | t_j – t_nr | Post-jet-break |
| IV | t > t_nr | Newtonian |

Phase I is excluded by design (cadences of interest are >> t_dec ≈ 19 s).

## Power-law segments

**PLS G** (optical band, ν_m < ν < ν_c) is the primary regime:
- Temporal decay index Phase II: a_II = (3(p−1))/4 → with p=2.5 gives a_II = 1.125
- Temporal decay index Phase III: a_III = (3p−2)/4 · (1 − 1/3) + ... simplifies to a_III ≈ p for on-axis
- Default p = 2.5 (hardcoded; not a UI slider)

## Detection regimes (A1–A7)

Classified by which constraint is binding: distance-limit (q_Euc) vs cadence-limit (q_i), and which phase applies.

| Regime | Binding | Phase | Physical meaning |
|--------|---------|-------|-----------------|
| A1 | q_Euc > q_i | IV (Newtonian) | Distance-limited, post-Newtonian |
| A2 | q_Euc > q_i | III (jet) | Distance-limited, post-jet |
| A3 | q_Euc > q_i | II (decel.) | Distance-limited, deceleration |
| A4 | q_i > q_Euc | IV | Cadence-limited, post-Newtonian |
| A5 | q_i > q_Euc | III | Cadence-limited, post-jet |
| A6 | q_i > q_Euc | II | Cadence-limited, deceleration |
| A7 | both tight | — | Doubly limited |

**Colour semantic:** warm (A1–A3) = distance/flux-limited; cool (A4–A6) = cadence-limited; gray = A7.

## Key scales

| Symbol | Value | Notes |
|--------|-------|-------|
| t_dec | ≈ 19 s | Deceleration time |
| t_j | ≈ 10⁴ × t_dec ≈ 0.66 d | Jet-break time |
| t_nr | ≈ (√2/θ_j) × t_j | Newtonian transition |
| q_dec | ≈ 1.001 | Phase I/II boundary ratio |
| q_j | 2 | Phase II/III boundary ratio |
| q_nr | ≈ 14 (= √2/θ_j) | Phase III/IV boundary ratio |
| D_euc | ≈ 5.28 Gpc | Euclidean horizon (z ≈ 2.0, flat ΛCDM, H₀ = 70 km/s/Mpc) — hardcoded |

D_euc is deliberately hardcoded as a constant; it should not be exposed as a UI slider.

## f_live two-model architecture

Survey time is shared between day and night cadences using two separate model instances:

| Mode | t_cad | Model | f_live argument |
|------|-------|-------|----------------|
| Sub-day | t_cad < t_night / i_det | `model_night` | `f_live` (plain) |
| Multi-day | t_cad = n × day | `model_day` | `f_live × f_night` where `f_night = t_night / 86400` |

Dispatch in the surface callback and in slice sweeps: use `model_night` when `t_cad < DAY_S`, else `model_day`.

## Unphysical strategies

When `f_live × t_cad / N_exp < t_overhead`, the implied exposure time is negative → unphysical.
`t_exp_s()` and `exposure_time_s()` return `NaN` in this case (never a negative number). The surface correctly shows no detection rate for these grid points.

## Deferred issues

- **i_det = 1 gap region:** The interval `t_night / i_det ≤ t_cad < 1 day` incorrectly invalidates strategies for i_det = 1. Deferred until dedicated i_det = 1 support is added.
