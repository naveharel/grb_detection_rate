"""Publication-figure subsystem for the GRB detection-rate model.

This package generates static, journal-quality figures (PNG by default; PDF opt-in) by importing the
read-only physics engine in ``grb_detect/``. It NEVER modifies the engine: when a
figure needs to vary a quantity the public API does not expose (e.g. the flux
normalization ``F_nu,dec``), it does so via the copy-on-write override layer in
``figures.figlib.overrides``.

See ``figures/README.md`` for conventions and the run command.
"""
