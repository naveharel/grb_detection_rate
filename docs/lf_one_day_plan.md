# Label LF luminosities at one day

Approved for implementation on 2026-09-14.

## Changes

- Add “(1 day)” to the minimum/maximum luminosity labels and fiducial/population-median displays.
- Keep the existing “Luminosity function” heading. Add no explanatory text.
- Retain νL<sub>ν</sub> units in erg/s and defaults α = −2, log₁₀ L<sub>min</sub> = 42.5, log₁₀ L<sub>max</sub> = 45.5.
- Preserve control IDs, layout, slider behavior, and public interfaces.
- Retain the engine’s existing parameter-dependent conversion between one-day and peak luminosity. Regenerate the standalone HTML.

## Verification

- Check the conversion under changes to p, energy, density, Γ<sub>0</sub>, and θ<sub>j</sub>, including jet breaks before and after one day.
- Verify collapsed-LF agreement with single-luminosity rates in dominant and full-integral modes.
- Browser-check labels, derived-value updates, unchanged one-day cutoffs during physics changes, and cutoff clamping.

## Execution

Save this approved plan before changing code, then implement and verify it.
