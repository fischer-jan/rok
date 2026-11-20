"""How the voxels look in the viewport: View > Look.

Each look is a set of switches for the renderer (Host.set_look; the shader's
look0 to look2): the light (lit, flat or toon bands), voxel occlusion, ink
lines on every voxel edge, a bright rim on convex edges, shadows from a fixed
sun, one clay material, and the background colour. The agent pictures
(rok/agent.py) keep the plain look whatever the window shows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

LIT, FLAT, TOON, ISO, SOFT, NIGHT, CERAMIC = 0, 1, 2, 3, 4, 5, 6


@dataclass(frozen=True)
class Look:
    label: str
    light: int = LIT
    ao: float = 0.0
    ink: float = 0.0
    rim: float = 0.0
    shadow: float = 0.0
    clay: float = 0.0
    ink_light: float = 0.0
    background: tuple[float, float, float] | None = None  # sRGB 0..1; None: the dark default
    ink_width: float = 1.0  # the ink lines' width in UI pixels

    def host_args(self) -> dict:
        d = asdict(self)
        d.pop("label")
        return d


LOOKS: dict[str, Look] = {
    "PLAIN": Look("Plain"),
    "STUDIO": Look("Studio", ao=1.0, rim=0.8),
    "SUN": Look("Sun", ao=0.8, rim=0.4, shadow=0.75, background=(0.70, 0.78, 0.86)),
    "PAPER": Look("Paper", ao=0.5, ink=1.0, ink_width=2.0, background=(0.95, 0.94, 0.91)),
    "FLAT": Look("Flat", light=FLAT, ink=0.85, ink_width=1.5, background=(0.55, 0.56, 0.58)),
    "TOON": Look("Toon", light=TOON, ink=1.0, ink_width=2.0, ao=0.25, background=(0.93, 0.90, 0.84)),
    "CLAY": Look("Clay", ao=1.0, rim=0.5, shadow=0.6, clay=1.0, background=(1.0, 1.0, 1.0)),
    "BLUEPRINT": Look("Blueprint", light=FLAT, ink=1.0, ink_width=1.5, ink_light=1.0, ao=0.6, background=(0.08, 0.20, 0.38)),
    "ISO": Look("Iso", light=ISO, ao=0.5, background=(0.17, 0.16, 0.15)),
    "SOFT": Look("Soft", light=SOFT, ao=1.0, rim=0.3, background=(0.80, 0.84, 0.88)),
    "NIGHT": Look("Night", light=NIGHT, ao=0.6, rim=0.6, background=(0.03, 0.04, 0.08)),
    "CERAMIC": Look("Ceramic", light=CERAMIC, ao=0.6, rim=0.5, background=(0.92, 0.90, 0.87)),
}
DEFAULT = "STUDIO"

# View > Ambient Occlusion: short rays through the voxels darken corners, undersides and the gaps
# between parts, in every look. Strength 0..1 and reach in cells
# (Host.set_occlusion).
AO_STRENGTH = 0.75
AO_REACH = 8.0


# --- the app side ---------------------------------------------------------------------

from pathlib import Path  # noqa: E402

from rok.core.safe_write import write_text  # noqa: E402

from .handlers import registry  # noqa: E402
from .opkit import EnumProp, Action, register  # noqa: E402

FILE_NAME = "look.txt"


def _file() -> Path:
    from . import recent

    return Path(recent.FOLDER) / FILE_NAME


def current(app) -> str:
    return getattr(app, "look", DEFAULT)


def apply(app, name: str, save: bool = True) -> None:
    """Show `name` in the viewport and remember it for the next start."""
    if name not in LOOKS:
        name = DEFAULT
    app.look = name
    args = LOOKS[name].host_args()
    if args["background"] is None:
        from . import theme

        args["background"] = theme.view_background()  # the Bright theme lightens the dark default
    app.host.set_look(**args)
    app.request_redraw()
    if save:
        try:
            _file().parent.mkdir(parents=True, exist_ok=True)
            write_text(_file(), name + "\n")
        except OSError:
            pass  # the look still shows; it is only not remembered


@register
class LookAction(Action):
    command_id = "rok.look"
    title = "Look"
    tooltip = "How the voxels look: light, occlusion, outlines, shadows, background"

    name = EnumProp(items=tuple((k, v.label, "") for k, v in LOOKS.items()), default=DEFAULT)

    def execute(self, context):
        apply(context.app, self.name)
        context.app.set_menu()
        return {"FINISHED"}


def menu_items(app) -> list:
    on = current(app)
    return [(f"rok.look(name={k!r})", v.label, "", int(k == on)) for k, v in LOOKS.items()]


def _start(context) -> None:
    try:
        saved = _file().read_text().strip()
    except OSError:
        saved = DEFAULT
    apply(context.app, saved, save=False)


registry.startup.append(_start)
