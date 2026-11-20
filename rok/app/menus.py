"""The macOS menu bar. Each item's id is the operator it runs (the app calls
it with INVOKE); "rok.flip(axis='X')" passes properties."""

from __future__ import annotations

from .handlers import registry


def app_menu(app) -> list:
    return [("Rok", [("rok.about", "About Rok", "", -1), None,
                     ("rok.preferences", "Settings…", "CmdOrCtrl+,", -1), None,
                     # The system's own items; they bring their own keys (Cmd+H, Alt+Cmd+H).
                     ("@hide", "Hide Rok", "", -1), ("@hide_others", "Hide Others", "", -1),
                     ("@show_all", "Show All", "", -1), None, ("rok.quit", "Quit Rok", "CmdOrCtrl+Q", -1)])]


def edit_menu(app) -> list:
    return [("Edit", [("rok.undo", "Undo", "CmdOrCtrl+Z", -1), ("rok.redo", "Redo", "CmdOrCtrl+Shift+Z", -1)])]


def view_menu(app) -> list:
    return [("View", [
        ("view.frame_all", "Frame All", "", -1),
        None,
        ("rok.smooth", "Smooth Edges", "", int(app.smooth)),
    ])]


def install() -> None:
    from . import io_ops

    registry.menus[:0] = [app_menu, io_ops.file_menu, edit_menu, view_menu]
