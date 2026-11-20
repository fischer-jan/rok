from rok.core import palette_anim as pa


def test_ease_out_ends():
    assert pa.ease_out(0) == 0 and pa.ease_out(1) == 1 and pa.ease_out(0.5) > 0.5


def test_tween_is_time_driven():
    tw = pa.Tweens()
    assert tw.value("h", 100, 0.12, at=0.0) == 100
    assert tw.value("h", 200, 0.12, at=1.0) == 100  # starts now
    mid = tw.value("h", 200, 0.12, at=1.06)
    assert 100 < mid < 200
    assert tw.running(at=1.06) and not tw.running(at=1.2)
    assert tw.value("h", 200, 0.12, at=1.2) == 200


def test_retarget_mid_tween_does_not_jump():
    tw = pa.Tweens()
    tw.value("h", 0.0, 0.1, at=0.0)
    tw.value("h", 100.0, 0.1, at=0.0)
    shown = tw.value("h", 100.0, 0.1, at=0.05)
    again = tw.value("h", 0.0, 0.1, at=0.05)
    assert again == shown


def test_colour_tween_and_jump():
    tw = pa.Tweens()
    tw.value("chip", (0.0, 0.0, 0.0), pa.CHIP_FADE, eased=False, at=0.0)
    tw.value("chip", (1.0, 1.0, 1.0), pa.CHIP_FADE, eased=False, at=0.0)
    assert tw.value("chip", (1.0, 1.0, 1.0), pa.CHIP_FADE, eased=False, at=0.04) == (0.5, 0.5, 0.5)
    tw.jump("ring", 3)
    assert tw.value("ring", 3, 1.0) == 3


def test_hover_delay():
    h = pa.Hover()
    assert h.update(("SWATCH", 1), at=0.0) == 0.0
    assert h.update(("SWATCH", 1), at=0.5) == 0.5
    assert h.update(("SWATCH", 2), at=0.6) == 0.0
    assert h.update(None, at=0.7) == 0.0


def test_clock_override():
    pa.set_clock(lambda: 42.0)
    try:
        assert pa.now() == 42.0
    finally:
        pa.set_clock()


def _fly_rect(k):
    return (10 * k, 0, 20, 20)


def test_flight_starts_when_a_colour_changes_section_and_ends():
    fl = pa.Flights()
    anchors = {"USED": (200, 100, 20, 20), "UNUSED": (200, 0, 20, 20)}
    fl.update({"USED": (1, 2), "UNUSED": (3,)}, {1: _fly_rect(1), 2: _fly_rect(2), 3: (0, 50, 20, 20)}, anchors, at=0.0)
    assert not fl.running()  # first sight: nothing flies
    after = {1: _fly_rect(1), 3: (0, 50, 20, 20), 2: (30, 50, 20, 20)}
    fl.update({"USED": (1,), "UNUSED": (2, 3)}, after, anchors, at=1.0)
    assert fl.running()
    assert fl.rects(after, anchors, at=1.0)[2] == (20, 0, 20, 20)  # starts at the old place
    mid = fl.rects(after, anchors, at=1.0 + pa.FLY / 2)[2]
    assert mid != (20, 0, 20, 20) and mid != (30, 50, 20, 20)
    assert 2 not in fl.rects(after, anchors, at=1.0 + pa.FLY)  # landed
    assert not fl.running()


def test_flight_to_a_folded_section_goes_to_its_heading():
    fl = pa.Flights()
    anchors = {"USED": (200, 100, 20, 20), "UNUSED": (200, 0, 20, 20)}
    fl.update({"USED": (1, 2), "UNUSED": ()}, {1: _fly_rect(1), 2: _fly_rect(2)}, anchors, at=0.0)
    fl.update({"USED": (1,), "UNUSED": (2,)}, {1: _fly_rect(1)}, anchors, at=1.0)  # Unused is folded
    r = fl.rects({1: _fly_rect(1)}, anchors, at=1.0 + pa.FLY * 0.99)[2]
    assert abs(r[0] - 200) < 3 and abs(r[1] - 0) < 3 and fl.rects({1: _fly_rect(1)}, anchors, at=1.0 + 0.1)


def test_flight_from_a_folded_section_starts_at_its_heading():
    fl = pa.Flights()
    anchors = {"USED": (200, 100, 20, 20), "UNUSED": (200, 0, 20, 20)}
    fl.update({"USED": (1,), "UNUSED": (2,)}, {1: _fly_rect(1)}, anchors, at=0.0)
    fl.update({"USED": (1, 2), "UNUSED": ()}, {1: _fly_rect(1), 2: _fly_rect(2)}, anchors, at=1.0)
    assert fl.rects({1: _fly_rect(1), 2: _fly_rect(2)}, anchors, at=1.0)[2] == anchors["UNUSED"]


def test_new_and_vanished_colours_and_mass_moves_do_not_fly():
    fl = pa.Flights()
    anchors = {"USED": (0, 0, 20, 20), "UNUSED": (0, 50, 20, 20)}
    fl.update({"USED": (1,), "UNUSED": ()}, {1: _fly_rect(1)}, anchors, at=0.0)
    fl.update({"USED": (), "UNUSED": (5,)}, {5: _fly_rect(5)}, anchors, at=1.0)  # 1 left, 5 is new
    assert not fl.running()
    many = tuple(range(100, 100 + pa.MAX_FLIGHTS + 1))
    fl.update({"USED": many, "UNUSED": ()}, {k: _fly_rect(1) for k in many}, anchors, at=2.0)
    fl.update({"USED": (), "UNUSED": many}, {k: _fly_rect(1) for k in many}, anchors, at=3.0)
    assert not fl.running()
