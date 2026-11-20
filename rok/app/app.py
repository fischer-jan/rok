"""Rok's app: host events in, dispatch to actions, changes out.

The host calls `RokApp.handle(kind, data)` for every event. The app turns
it into an `opkit.Event` and dispatches it in this order:

1. an open dialog takes every event;
2. modal actions, newest first (RUNNING, FINISHED and CANCELLED
   consume the event, PASS passes it on);
3. the keymap (an item whose action returns FINISHED or RUNNING
   consumes the event);
4. the view navigation (trackpad, wheel, pinch).

After each event the app flushes the grids to the GPU (`GpuSync`), and when
something asked for a redraw it runs the draw handlers, one layer each.
The editor modules register their operators, keymap items, draw
handlers and settings when `rok.app.modules` imports them.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from rok.core.undo import naming

from . import dialog as dialog_mod
from . import draw, event_profile, gfx, keymap, navigation, opkit, render_view, settings
from . import camera as camera_mod
from .camera import Camera
from .handlers import registry
from .object import document, style_of
from .opkit import CONTEXT, Area, Event, Timer, Window
from .state import state
from .sync import GpuSync
from .view3d import Region, ViewState

ORBIT_RAD_PER_PX = 0.005
ERROR_LOG = "error.log"  # in the settings folder; see RokApp.report_error
ERROR_LOG_KEEP = 200_000  # characters of older reports kept


class RokPrefs(opkit.Prefs):
    """The window manager with the rok_* settings; the modules add them."""


class RokApp:
    def __init__(self, host, path: str | None = None) -> None:
        self.host = host
        self.start_path = path
        self.width, self.height = 1440.0, 900.0  # the window in logical pixels
        self.quad = False  # View > Quad View: Top, Perspective, Front and Right
        self.buttons_down = 0
        self.main_camera = Camera.default()
        self.main_camera.lift = camera_mod.VIEW_LIFT  # clear of the palette card at the bottom
        self.sync = GpuSync(host)
        self.layers = draw.Layers(host)
        self.icons = gfx.Icons(host)
        self._measure_cache: dict = {}
        gfx.fonts.measure = self.measure_text
        self.region = Region(self.main_camera, self)
        self.view_state = ViewState(self.main_camera, self)
        self.views: list[ViewState] = [self.view_state]
        self.wm = RokPrefs()
        CONTEXT.app = self
        CONTEXT.region = self.region
        CONTEXT.view_state = self.view_state
        CONTEXT.prefs = self.wm
        CONTEXT.area = Area(self)
        CONTEXT.window = Window(self)
        self.modals: list = []
        self.timers: list[Timer] = []
        self.dialog: dialog_mod.Dialog | None = None
        self.pending: dict[str, tuple] = {}  # host dialog token -> (kind, op)
        self.mods = {"shift": False, "ctrl": False, "alt": False, "cmd_key": False}
        self.mouse: tuple[float, float] | None = None
        self.last_event = Event("NONE")
        self.redraw_wanted = True
        self.camera_dirty = True
        self.smooth = True
        self.animation_menu = False  # View > Show Animation Menu; off until you turn it on
        self.history_strip = False  # View > Show History (history_strip.py)
        self.grid = False  # View > Voxel Grid: the edges of the voxels
        self.cad = False  # View > CAD View: grey faces, corner lines, side lengths
        self.status: str | None = None
        self.message = ""
        self.message_until = 0.0
        self._token = 0
        self._started = False
        self._keymap: list[keymap.Item] | None = None

    # --- services for opkit and the editor modules ---------------------------------

    def request_redraw(self) -> None:
        self.redraw_wanted = True

    def camera_moved(self) -> None:
        self.camera_dirty = True
        self.redraw_wanted = True

    def set_status(self, text) -> None:
        self.status = text
        self.redraw_wanted = True

    def say(self, message: str, seconds: float = 3.0) -> None:
        self.message = message
        self.message_until = time.monotonic() + seconds
        self.host.tick(seconds + 0.05)
        self.redraw_wanted = True

    def measure_text(self, text: str, size: float) -> tuple[float, float]:
        key = (text, round(size, 2))
        hit = self._measure_cache.get(key)
        if hit is None:
            hit = tuple(self.host.measure_text(text, float(size)))
            if len(self._measure_cache) > 4096:
                self._measure_cache.clear()
            self._measure_cache[key] = hit
        return hit

    def warp_cursor(self, x: float, y: float) -> None:
        """Region pixels, origin bottom left."""
        self.host.warp_cursor(float(x), float(self.region.height - y))
        self.mouse = (float(x), float(self.region.height - y))

    def modal_add(self, op) -> None:
        self.modals.append(op)

    def timer_add(self, interval: float) -> Timer:
        t = Timer(interval=interval, next_at=time.monotonic() + interval)
        self.timers.append(t)
        self.host.tick(interval)
        return t

    def timer_remove(self, timer) -> None:
        if timer in self.timers:
            self.timers.remove(timer)

    def _new_token(self, kind: str) -> str:
        self._token += 1
        return f"{kind}:{self._token}"

    def open_dialog(self, op, width: int = 300, title: str = "", confirm: str = "OK") -> None:
        self.dialog = dialog_mod.Dialog(self, op, width=width, title=title, confirm=confirm)
        self.redraw_wanted = True

    def confirm(self, op, title: str, message: str, confirm: str = "OK") -> None:
        token = self._new_token("confirm")
        self.pending[token] = ("confirm", op, confirm)
        self.host.ask(token, title, message or title, [confirm, "Cancel"])

    def file_select(self, op) -> None:
        token = self._new_token("file")
        save = isinstance(op, opkit.SaveFile)
        exts = [e.strip().lstrip("*.") for e in str(getattr(op, "filter_glob", "") or "").split(";") if e.strip()]
        filters = [(op.title or "Files", exts)] if exts else []
        path = str(getattr(op, "filepath", "") or "")
        name = Path(path).name if path else None
        directory = str(Path(path).parent) if path and Path(path).parent != Path(".") else None
        self.pending[token] = ("file", op, None)
        self.host.file_dialog(token, save, op.title or ("Save" if save else "Open"), filters, name, directory)

    def keymap(self) -> list[keymap.Item]:
        if self._keymap is None:
            self._keymap = keymap.items(registry.keymap)
        return self._keymap

    def update_title(self) -> None:
        self.host.set_title(f"{document.name} — Rok")
        self.host.set_edited(bool(document.modified))

    def flush(self) -> None:
        from . import anim_pictures, cad_mode

        if anim_pictures.running():
            return  # an export steps through the clip: the parts it sent stay until its picture is back
        # Count dirty chunks before the flush clears them. A voxel edit drops the
        # CAD field on the GPU, so the view sends it again for those models.
        dirty = None
        if self.cad and document.scene is not None:
            dirty = [len(m.grid.dirty_chunks()) for m in document.scene.models]
        if document.scene is not None:
            from . import anim_mode

            shown = anim_mode.shown()
            turns = {**shown.turns, **state.turn_preview}
            if self.sync.flush(document.scene, style_of=style_of, turns=turns or None, poses=shown.poses,
                               palette=shown.palette):
                self.redraw_wanted = True
            render_view.send(self)  # the lamps follow the scene's size
        cad_mode.after_flush(self, dirty)

    @property
    def camera(self):
        """The camera of the view the mouse works in (the only one outside the quad view)."""
        view_state = CONTEXT.view_state
        return view_state.camera if view_state is not None else self.main_camera

    @camera.setter
    def camera(self, value) -> None:  # the tests swap in a camera now and then
        self.main_camera = value
        self.view_state._camera = value
        self.region._camera = value

    def set_size(self, width: float, height: float) -> None:
        self.width, self.height = float(width), float(height)
        self.layout_views()

    def layout_views(self) -> None:
        """Each view's rectangle (region pixels, origin bottom left) and camera size."""
        W, H = self.width, self.height
        if not self.quad:
            self.view_state.rect = None
            self.main_camera.width, self.main_camera.height = W, H
            return
        hw, hh = W / 2.0, H / 2.0
        places = {"Top": (0.0, hh), "Perspective": (hw, hh), "Front": (0.0, 0.0), "Right": (hw, 0.0)}
        for view_state in self.views:
            x, y = places[view_state.name]
            view_state.rect = (x, y, hw, hh)
            view_state.camera.width, view_state.camera.height = hw, hh
        self.camera_moved()

    def set_quad(self, on: bool) -> None:
        """Four views (Top, Perspective, Front, Right) or the one perspective view."""
        import math

        if on == self.quad:
            return
        if on:
            from . import cad_mode

            cad_mode.set_on(self, False)  # four views replace the CAD view
        self.quad = on
        main = self.main_camera
        if on:
            sides = []
            for name, yaw, pitch in (("Top", -math.pi / 2, math.pi / 2), ("Front", -math.pi / 2, 0.0), ("Right", 0.0, 0.0)):
                cam = Camera(target=main.target.copy(), yaw=yaw, pitch=pitch, distance=main.distance,
                             ortho=True, locked=True, up_hint=(0.0, 1.0, 0.0))
                sides.append(ViewState(cam, self, name=name))
            self.views = [sides[0], self.view_state, sides[1], sides[2]]
        else:
            self.views = [self.view_state]
            CONTEXT.view_state = self.view_state
        self.layout_views()
        self.camera_moved()

    def view_at(self, x: float, y: float):
        """The view under a region pixel (origin bottom left)."""
        for view_state in self.views:
            if view_state.contains(self.region, x, y):
                return view_state
        return self.view_state

    def send_camera(self) -> None:
        if self.quad:
            H = self.height
            self.host.set_views([
                (v.rect[0], H - v.rect[1] - v.rect[3], v.rect[2], v.rect[3], v.camera.view_proj(),
                 tuple(float(c) for c in v.camera.eye()), float(v.camera.pixel))
                for v in self.views
            ])
            self.camera_dirty = False
            return
        self.host.set_views([])
        self.main_camera.width, self.main_camera.height = self.width, self.height
        self.host.set_camera(self.main_camera.view_proj(), tuple(float(v) for v in self.main_camera.eye()))
        self.camera_dirty = False

    def set_menu(self) -> None:
        menus = []
        for build in registry.menus:
            menus.extend(build(self))
        self.host.set_menu(menus)

    # --- events ------------------------------------------------------------------

    def _event(self, type_: str, value: str = "NOTHING", **extra) -> Event:
        x, y = self.mouse if self.mouse is not None else (0.0, 0.0)
        ry = self.region.height - y
        prev = self.last_event
        return Event(
            type=type_, value=value,
            shift=self.mods["shift"], ctrl=self.mods["ctrl"], alt=self.mods["alt"], cmd_key=self.mods["cmd_key"],
            mouse_region_x=int(round(x)), mouse_region_y=int(round(ry)),
            mouse_x=int(round(x)), mouse_y=int(round(ry)),
            mouse_prev_x=prev.mouse_x, mouse_prev_y=prev.mouse_y,
            fx=float(x), fy=float(ry),
            **extra,
        )

    def handle(self, kind: str, data: dict) -> None:
        try:
            event_profile.run(kind, lambda: self._handle_and_draw(kind, data))
        except Exception:
            self.report_error(kind)
            raise

    def report_error(self, kind: str) -> None:
        """After an error in an event: end a stroke that would otherwise block
        every tool, say so in the window, and keep the traceback in error.log
        in the settings folder (a Rok started from Finder has no console)."""
        import traceback

        from . import recent
        from .state import state

        text = traceback.format_exc()
        state.stroke_active = False
        try:
            path = Path(recent.FOLDER) / ERROR_LOG
            path.parent.mkdir(parents=True, exist_ok=True)
            old = path.read_text()[-ERROR_LOG_KEEP:] if path.exists() else ""
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            path.write_text(old + f"--- {stamp} event {kind}\n{text}\n")
        except OSError:
            pass
        last = text.strip().splitlines()[-1] if text.strip() else "an error"
        self.say(f"Error: {last[:90]} (details in {ERROR_LOG})", 8.0)

    def _handle_and_draw(self, kind: str, data: dict) -> None:
        try:
            self._handle(kind, data)
        finally:
            self.after_event()

    def _handle(self, kind: str, data: dict) -> None:
        if kind == "start":
            self.on_start(**data)
        elif kind == "resize":
            self.set_size(data["width"], data["height"])
            self.camera_dirty = True
            self.redraw_wanted = True
        elif kind == "modifiers":
            new = {"shift": data["shift"], "ctrl": data["ctrl"], "alt": data["alt"], "cmd_key": data["meta"]}
            for mod, key in (("shift", "LEFT_SHIFT"), ("ctrl", "LEFT_CTRL"), ("alt", "LEFT_ALT"), ("cmd_key", "CMD")):
                if new[mod] != self.mods[mod]:
                    self.mods[mod] = new[mod]
                    self.dispatch(self._event(key, "PRESS" if new[mod] else "RELEASE"))
        elif kind == "mouse_move":
            self.mouse = (data["x"], data["y"])
            self._pick_view()
            self.dispatch(self._event("MOUSE_MOVE"))
        elif kind == "mouse_delta":
            # Raw moves while the host holds the pointer still (Host.set_relative).
            # The host's y runs down, the region's up.
            self.dispatch(self._event("MOUSE_DELTA", dx=float(data["dx"]), dy=-float(data["dy"])))
        elif kind == "mouse_leave":
            self.dispatch(self._event("FOCUS_LOST"))
        elif kind in ("mouse_down", "mouse_up"):
            button = opkit.MOUSE_BUTTONS.get(data["button"])
            if kind == "mouse_down":
                self._pick_view()
                self.buttons_down += 1
            else:
                self.buttons_down = max(0, self.buttons_down - 1)
            if button is not None:
                self.dispatch(self._event(button, "PRESS" if kind == "mouse_down" else "RELEASE"))
        elif kind == "scroll":
            if data["precise"]:
                self.dispatch(self._event("TRACKPAD_PAN", dx=data["dx"], dy=data["dy"], precise=True))
            else:
                self.dispatch(self._event("WHEEL_UP" if data["dy"] > 0 else "WHEEL_DOWN", "PRESS", dy=data["dy"]))
        elif kind == "pinch":
            self.dispatch(self._event("TRACKPAD_ZOOM", dx=data["delta"]))
        elif kind in ("key_down", "key_up"):
            if self.mods["cmd_key"] and kind == "key_down" and data["key"] not in ("Escape",):
                pass  # the menu bar owns Cmd shortcuts; the rest still reach the keymap
            ev = self._event(opkit.event_type_of_key(data["key"]), "PRESS" if kind == "key_down" else "RELEASE",
                             unicode=data.get("text", ""), is_repeat=bool(data.get("repeat")))
            self.dispatch(ev)
        elif kind == "wake":
            from . import live

            if live.current() is not None:
                live.current().on_wake()
        elif kind == "screenshot" and str(data.get("token", "")).startswith("anim:"):
            from . import anim_pictures

            anim_pictures.on_picture(self, data)
        elif kind == "screenshot" and str(data.get("token", "")).startswith("picture:"):
            from . import render_picture

            render_picture.on_picture(self, data)
        elif kind == "screenshot":
            from . import live

            if live.current() is not None:
                live.current().on_screenshot(data)
        elif kind == "tick":
            self.on_tick()
            from . import live

            if live.current() is not None:
                live.current().on_wake()
        elif kind == "menu":
            self.on_menu(data["id"])
        elif kind == "dialog":
            self.on_dialog(**data)
        elif kind == "close":
            opkit.call("rok.quit", "INVOKE") if "rok.quit" in opkit.REGISTRY else self.host.quit()
        elif kind == "focus":
            if not data["focused"]:
                self.dispatch(self._event("FOCUS_LOST"))

    def dispatch(self, ev: Event) -> None:
        self.last_event = ev
        if self.dialog is not None:
            if self.dialog.handle(ev):
                return
        for op in reversed(list(self.modals)):
            if op not in self.modals:
                continue
            try:
                with naming(opkit.undo_label(op)):
                    result = set(op.modal(CONTEXT, ev) or {"PASS"})
            except Exception:
                self.modals.remove(op)
                raise
            if result & {"FINISHED", "CANCELLED"}:
                if op in self.modals:
                    self.modals.remove(op)
            if "PASS" not in result:
                return
        if ev.type not in ("MOUSE_MOVE", "TIMER", "NONE") and ev.value in ("PRESS", "RELEASE", "NOTHING"):
            for item in self.keymap():
                if not item.matches(ev):
                    continue
                cls = opkit.REGISTRY.get(item.idname)
                if cls is None or not cls.poll(CONTEXT):
                    continue
                result = opkit.call(item.idname, "INVOKE", **(item.props or {}))
                if result & {"FINISHED", "RUNNING"} and "PASS" not in result:
                    return
        self.navigate(ev)

    def navigate(self, ev: Event) -> None:
        """The view navigation that needs no action: trackpad, wheel and pinch."""
        cam = self.camera
        if ev.type == "TRACKPAD_PAN":
            if ev.alt:
                cam.zoom(float(np.exp(ev.dy * 0.01)))
            elif ev.ctrl:
                cam.pan(ev.dx, ev.dy)
            else:
                cam.orbit(ev.dx, ev.dy)
        elif ev.type == "TRACKPAD_ZOOM":
            cam.zoom(1.0 + ev.dx)
        elif ev.type in ("WHEEL_UP", "WHEEL_DOWN"):
            from . import overlay

            # Over a height-capped palette card the wheel scrolls that section, not the view.
            if overlay.is_main_region(self.region) and overlay.dock_scroll(
                    self.region, ev.mouse_region_x, ev.mouse_region_y, 1 if ev.type == "WHEEL_UP" else -1):
                self.region.tag_redraw()
                return
            factor = navigation.wheel_zoom(ev)  # Rok > Settings: the way it turns
            if factor is None:
                return
            cam.zoom(factor)
        else:
            return
        self.camera_moved()
        # A MOUSE_MOVE after a view change makes the hover repick.
        self.dispatch(self._event("MOUSE_MOVE"))

    def _pick_view(self) -> None:
        """In the quad view the mouse works in the view under it; a drag stays
        in the view it started in."""
        if not self.quad or self.buttons_down or self.mouse is None:
            return
        view_state = self.view_at(self.mouse[0], self.height - self.mouse[1])
        if view_state is not CONTEXT.view_state:
            CONTEXT.view_state = view_state
            self.redraw_wanted = True

    def on_start(self, width: float, height: float, scale: float) -> None:
        self.set_size(width, height)
        from . import modules  # noqa: F401 - registers the editor modules

        self._keymap = None
        self.grid = settings.get("voxel_grid") == "on"
        self.animation_menu = settings.get("animation_menu") == "on"
        if self.grid:
            self.host.set_voxel_grid(True)
        self.set_occlusion(settings.get("ambient_occlusion", "on") == "on")
        state.model_box = settings.get("model_box", "on") == "on"
        self.set_menu()
        # `late` functions start after the rest (a modal that must sit on top).
        for fn in sorted(registry.startup, key=lambda f: bool(getattr(f, "late", False))):
            fn(CONTEXT)
        self._started = True
        from . import live

        live.start(self)  # the open window takes agent calls (rok/app/live.py)
        if self.start_path and "rok.open" in opkit.REGISTRY:
            opkit.call("rok.open", "EXEC", filepath=str(self.start_path))
        elif "rok.new" in opkit.REGISTRY:
            opkit.call("rok.new", "EXEC")

    def on_tick(self) -> None:
        now = time.monotonic()
        due = [t for t in self.timers if t.next_at <= now + 0.001]
        for t in due:
            t.time_duration = now - t.started
            t.next_at = now + t.interval
        for t in due:
            if t.callback is not None:
                t.callback()
        if any(t.callback is None for t in due):
            self.dispatch(self._event("TIMER"))
        if self.timers:
            nxt = min(t.next_at for t in self.timers)
            self.host.tick(max(0.001, nxt - time.monotonic()))
        self.redraw_wanted = True

    def set_occlusion(self, on: bool) -> None:
        """Ambient occlusion in the voxels, on or off (View > Ambient Occlusion)."""
        from . import looks

        self.occlusion = bool(on)
        if hasattr(self.host, "set_occlusion"):  # an older native module has none
            self.host.set_occlusion(looks.AO_STRENGTH if self.occlusion else 0.0, looks.AO_REACH)
        self.redraw_wanted = True

    def on_menu(self, ident: str) -> None:
        if ident == "rok.about":
            self.say("Rok 0.0.1 — a voxel editor")
            return
        if ident == "rok.grid":
            self.grid = not self.grid
            self.host.set_voxel_grid(self.grid)
            settings.put("voxel_grid", "on" if self.grid else "off")  # it stays
            self.set_menu()
            return
        if ident == "rok.animation_menu":
            self.animation_menu = not self.animation_menu
            settings.put("animation_menu", "on" if self.animation_menu else "off")  # it stays
            if not self.animation_menu and state.chrome_open == "ANIM_MENU":
                state.chrome_open = None
            self.set_menu()
            self.request_redraw()
            return
        if ident == "rok.history_strip":
            from . import history_strip

            history_strip.toggle(self)
            return
        if ident == "rok.occlusion":
            self.set_occlusion(not self.occlusion)
            settings.put("ambient_occlusion", "on" if self.occlusion else "off")
            self.set_menu()
            return
        if ident == "rok.smooth":
            self.smooth = not self.smooth
            self.host.set_smooth_edges(self.smooth)
            self.set_menu()
            return
        if "(" in ident:  # "rok.flip(axis='X')"
            name, args = ident.split("(", 1)
            props = eval(f"dict({args}")  # noqa: S307 - our own menu ids
        else:
            name, props = ident, {}
        if name in opkit.REGISTRY:
            opkit.call(name, "INVOKE", **props)

    def on_dialog(self, token: str, path: str | None = None, answer: str | None = None) -> None:
        kind, op, extra = self.pending.pop(token, (None, None, None))
        if op is None:
            return
        if kind == "file":
            if path:
                op.filepath = path
                with naming(opkit.undo_label(op)):
                    result = opkit.run(op.execute(CONTEXT) or set(), op)
                self._after_op(op, result)
            else:
                op.cancel(CONTEXT)
        elif kind == "confirm":
            if answer == extra:
                with naming(opkit.undo_label(op)):
                    result = opkit.run(op.execute(CONTEXT) or set(), op)
                self._after_op(op, result)
            else:
                op.cancel(CONTEXT)

    def _after_op(self, op, result: set) -> None:
        if "RUNNING" in result and op not in self.modals:
            pass  # the operator added itself as a modal handler

    def after_event(self) -> None:
        if not self._started:
            return
        self.flush()
        if self.camera_dirty:
            self.send_camera()
        if self.redraw_wanted:
            self.redraw_wanted = False
            self.redraw()

    # --- drawing -----------------------------------------------------------------

    def redraw(self) -> None:
        for h in registry.draw:
            d = draw.DrawList()
            if h.kind == "VIEW":
                with gfx.view(d, linear=h.linear):
                    h.fn(CONTEXT)
            else:
                with gfx.pixels(d, self.region, linear=h.linear):
                    h.fn(CONTEXT)
            self.layers.send(h.name, h.z, d)
        d = draw.DrawList()
        if self.dialog is not None:
            with gfx.pixels(d, self.region):
                self.dialog.draw()
        self.layers.send("~dialog", 1000, d)
        d = draw.DrawList()
        if self.message and time.monotonic() < self.message_until:
            tw, _th = self.measure_text(self.message, 13)
            w = self.width
            d.rect((w - tw) / 2 - 14, 14, tw + 28, 30, draw.PLATE, radius=15)
            d.text((w - tw) / 2, 20, self.message, 13, draw.TEXT)
        self.layers.send("~message", 1001, d)


def main(argv: list[str] | None = None) -> None:
    import rok_native

    argv = sys.argv[1:] if argv is None else argv
    root = Path(__file__).resolve().parents[2]
    icon = root / "assets" / "icon" / "rok.png"
    host = rok_native.Host("Rok", str(icon) if icon.exists() else None)
    app = RokApp(host, argv[0] if argv else None)
    host.run(app.handle)
