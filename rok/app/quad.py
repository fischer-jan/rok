"""View > Quad View: Top, Perspective, Front and Right side by side.

A button by the navigation pill opens it. The app keeps a camera per view (app.set_quad), Rust draws each view
in its own rectangle, and the mouse works in the view under it. The side
views are orthographic and locked: they pan and zoom, they do not turn.
This module draws the lines between the views and each view's name, and
runs the switch.
"""

from __future__ import annotations

from . import gfx
from .gfx import fonts
from .handlers import registry
from .opkit import Action, register

SEAM = (0.0, 0.0, 0.0, 0.55)
ACTIVE_EDGE = (0.91, 0.36, 0.23, 0.55)
LABEL = (0.905, 0.885, 0.855, 0.72)
LABEL_LEFT = 70.0  # past the rail in the left views
RAIL_W = 56.0


@register
class QuadViewAction(Action):
    command_id = "rok.quad_view"
    title = "Quad View"
    tooltip = "Top, Perspective, Front and Right side by side; again for one view"

    def execute(self, context):
        from . import render_view

        if not context.app.quad:
            render_view.set_rendered(context.app, False)  # the rendered view shows one view
        context.app.set_quad(not context.app.quad)
        context.app.set_menu()
        return {"FINISHED"}


def _draw(context) -> None:
    app = context.app
    if not app.quad:
        return
    region = context.region
    W, H = float(region.width), float(region.height)
    # The seams: a cross through the middle.
    gfx.rrect((W / 2.0 - 1.0, 0.0, 2.0, H), SEAM, 0.0)
    gfx.rrect((0.0, H / 2.0 - 1.0, W, 2.0), SEAM, 0.0)
    for view_state in app.views:
        x, y, w, h = view_state.area(region)
        if view_state is context.view_state:
            gfx.rrect((x + 1.0, y + 1.0, w - 2.0, h - 2.0), ACTIVE_EDGE, 0.0, stroke=1.0)
        left = x + (LABEL_LEFT if x < RAIL_W else 14.0)
        fonts.size(0, 12)
        fonts.color(0, *LABEL)
        fonts.position(0, left, y + h - 24.0, 0)
        fonts.draw(0, view_state.name)


registry.add_draw("quad", 505, "PIXEL", _draw)
registry.keymap.append(("rok.quad_view", {"type": "Q", "value": "PRESS", "ctrl": True, "alt": True}, None))  # was the menu item's key
