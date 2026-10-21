# Approved ZTF observing-block update

Approved for implementation on 2026-10-21.

- Public: 7,500 square degrees nightly, alternating halves of the full 15,000
  square degree footprint; two visits in an exact four-hour observing window.
- HC: 2,500 square degrees nightly; six visits in an exact six-hour window.
- Both presets: two required detections, f_live = 0.16, 15-second overhead,
  cadence = 0.98 times the observing window divided by the visit count.
- Derived cadences are 1.96 and 0.98 hours; exposures are 27.448896 seconds.
  The public nightly footprint already accounts for alternate-night coverage;
  do not multiply its rate by another half.
- Retain control precision, physics formulas, bridge payload keys, the initial
  preset selection, and existing unrelated local edits.
- Load the observing window with each ZTF preset and include it in matching and
  drift detection. Keep markers on the displayed model and hide invalid points.
- Explain the effective window and public footprint cycle in the UI. Synchronize
  the Python presets; pin existing paper figures to their legacy configuration.
- Rebuild the standalone HTML and verify preset switching, control precision,
  exposure budgets, both calculation modes, LF behavior, and invalid counts.

Acceptance rates at the current UI physics defaults, LF/cuts off:

| Mode | Default approximation, per year | Exact with peak correction, per year |
| --- | ---: | ---: |
| Public | 39.53 | 40.31 |
| HC | 42.79 | 44.03 |

## Follow-up authorized on 2026-10-21

The later user request supersedes the live-fraction precision and strict-window
assumptions above: default and ZTF preset f_live = 0.167, step = 0.001. Keep the
4/6-hour windows using a budget comparison at that control precision. Grids and
optimizer candidates use whole field counts and exact minute/hour anchors;
reference footprints use their nearest whole counts (160 public, 53 HC).
The optical gap starts strictly above the selected detection-window boundary:
(i-1)*t_cad <= t_night or i*t_cad <= t_night. This applies to every plot and point.
No new visual indicators. The preset cadence factor remains 0.98; updated public
and HC exposures are 29.1882 and 29.4661 seconds. Approximate rates are 41.50 and
42.87/year; Exact with peak correction gives 42.32 and 44.45/year at the same
physical defaults, single luminosity, two detections, and no cuts.

## Live-fraction correction authorized on 2026-10-21

This update supersedes the 0.167 default and control-precision budget allowance
above. Set the UI default and both ZTF presets to f_live = 0.166, retaining the
0.001 control step. Enforce f_night >= f_live without a half-step tolerance:
the public preset retains its exact four-hour window; entering 0.167 instead
requires at least 4.25 hours on the existing quarter-hour grid.

Keep the 4/6-hour preset windows, 160/53 whole fields, 1.96/0.98-hour cadences,
integer strategy grids, inclusive selected detection-window constraint, and
legacy paper-figure configurations. Update preset metadata and expectations,
rebuild the HTML, and verify preset switching, the budget floor, Exact/LF modes,
and marker/surface agreement.

Expected results at the current UI physics defaults, single luminosity, two
detections, and no cuts:

| Mode | Exposure, seconds | Approximate rate, per year | Exact with peak correction, per year |
| --- | ---: | ---: | ---: |
| Public | 28.9236 | 41.2181330870 | 42.0311531062 |
| HC | 29.1998490566 | 42.8391115775 | 44.3718513870 |

Remove the newly added Constraints explanations and retain the old warning.
Record the user's app-copy rule in `AGENTS.md` and `CLAUDE.md`: show exact
proposed wording and ask for approval before adding or changing app text.
Audit app copy for concise explanations of the current model aimed at readers
arriving from the paper; present any replacement wording for approval before
applying it.

Implementation verified on 2026-10-21: applied the approved Tooltips, Settings,
and Messages wording, retained the existing setting names except the approved
peak-window label, and rebuilt the standalone HTML. All 131 targeted native
tests and 16 headless Edge tests passed, including the strict budget floor,
preset switching, numerical reference values, integer grids, inclusive window
boundary, marker/surface agreement, Exact mode, and luminosity-function mode.
All five tabs and conditional loading/error messages were inspected; embedded
JavaScript and all ten Python sources match the working tree.
