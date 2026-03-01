# plot_3d.py — GRB Detection Rate Explorer (entry point)
import os

import dash_bootstrap_components as dbc
from dash import Dash

from components.layout import create_layout
import callbacks.surface
import callbacks.sync
import callbacks.ui

# Google Fonts loaded via CSS @import in assets/style.css.
# Bootstrap CSS is needed for DBC accordion and switch components.
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
server = app.server

app.layout = create_layout()

# Register callbacks in dependency order
callbacks.sync.register(app)      # slider ↔ input sync (no physics deps)
callbacks.ui.register(app)        # theme / sidebar / view / presets
callbacks.surface.register(app)   # main surface + slices + metrics


if __name__ == "__main__":
    debug = os.environ.get("GRB_DEBUG", "1").lower() in ("1", "true", "yes")
    app.run(debug=debug)
