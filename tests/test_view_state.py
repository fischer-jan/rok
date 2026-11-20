"""ViewState reports the projection of the camera it wraps."""

from rok.app.view3d import ViewState


class _Camera:
    def __init__(self, ortho: bool) -> None:
        self.ortho = ortho


def test_the_projection_follows_the_camera():
    assert ViewState(_Camera(False)).view_perspective == "PERSP"
    assert ViewState(_Camera(False)).is_perspective
    assert ViewState(_Camera(True)).view_perspective == "ORTHO"
    assert not ViewState(_Camera(True)).is_perspective


def test_the_projection_is_not_a_plain_class_attribute():
    for name in ("view_perspective", "is_perspective"):
        assert isinstance(ViewState.__dict__[name], property), name
