"""Shared library for the figure subsystem.

Modules
-------
style     : AAS/ApJ matplotlib style (usetex with mathtext fallback), palettes, figsize.
presets   : Python mirror of the app's survey presets (single source of truth note).
overrides : copy-on-write "model surgery" — vary quantities past the public API.
models    : build models from presets; reproduce app strategy points.
compute   : thin evaluators (q_median / D_median / rate) at a point or over a sweep.
axes      : log-axis and reference-line helpers.
io        : savefig_pub() — writes PNG to figures/output/ (PDF opt-in via formats=).
"""
