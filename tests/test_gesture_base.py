"""The gesture base class: every gesture the stroke operator drives has it."""

from rok.core.corner_gizmo import CornerDrag
from rok.core.edit import Stroke
from rok.core.gesture import FaceGesture, PatternGesture, ScreenGesture, ShapeGesture
from rok.core.gesture_base import Gesture
from rok.core.gizmo import GizmoDrag
from rok.core.select import ConnectedGesture, ColourGesture, MoveGesture, PickGesture

GESTURES = (
    Stroke, ShapeGesture, FaceGesture, ScreenGesture, PatternGesture, PickGesture,
    ConnectedGesture, ColourGesture, MoveGesture, GizmoDrag, CornerDrag,
)


def test_every_gesture_is_a_gesture():
    for cls in GESTURES:
        assert issubclass(cls, Gesture), cls.__name__


def test_only_the_screen_gestures_take_pixels():
    assert {cls for cls in GESTURES if cls.screen_drag} == {ScreenGesture, CornerDrag}
    for cls in GESTURES:
        assert cls.screen_drag == hasattr(cls, "extend_screen"), cls.__name__


def test_only_the_selection_move_locks_an_axis():
    assert {cls for cls in GESTURES if cls.locks_axis} == {MoveGesture}


def test_the_defaults_do_nothing():
    g = Gesture()
    g.set_shift(True)
    g.set_mouse(1.0, 2.0)
    assert g.toggle_axis_lock(0) is False
    assert g.begin_height() is False
    assert g.hit is None and g.last is None and g.viewer is None and len(g.visited) == 0
