"""The timeline dock's geometry (rok/core/dock.py)."""

import pytest

from rok.core import anim
from rok.core import dock as kit


def clip_with_spin(uid: int = 7) -> anim.Clip:
    track = anim.Track(anim.Target(anim.PART, uid))
    for name, keys in anim.preset_keys("SPIN", track, anim.Clip()):
        for key in keys:
            anim.set_key(track, name, key)
    return anim.Clip(tracks=[track])


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_frames_map_to_the_lane_and_back(scale):
    layout = kit.dock_layout(scale, 60, 1000 * scale, clip_with_spin(), [(7, "Fan")], 7)
    lx0, lx1 = layout.lane
    assert layout.x_of(0) == lx0 and layout.x_of(24) == pytest.approx(lx1)
    assert all(layout.frame_at(layout.x_of(f)) == f for f in range(25))
    assert layout.rect[1] == round(kit.MARGIN * scale)


def test_a_part_with_keys_gets_a_row_and_its_channels_when_open():
    layout = kit.dock_layout(1.0, 60, 1000, clip_with_spin(), [(3, "Hub"), (7, "Fan")], 3, opened={7})
    # In the scene's order; the picked Hub has no keys and still gets its row.
    assert [(r.kind, r.uid, r.channel) for r in layout.rows] == [
        (kit.PART_ROW, 3, None), (kit.PART_ROW, 7, None), (kit.CHANNEL_ROW, 7, anim.TURN_Z)]
    assert layout.rows[1].keys == ((0, anim.LINEAR), (24, anim.LINEAR))
    assert layout.rows[2].value == "0°" and layout.rows[0].picked and not layout.rows[0].keys


def test_hits():
    layout = kit.dock_layout(1.0, 60, 1000, clip_with_spin(), [(7, "Fan")], 7)
    row = layout.rows[0]
    cy = row.rect[1] + row.rect[3] / 2
    assert kit.hit(layout, layout.x_of(24), cy) == kit.Hit(kit.KEY, uid=7, channel=None, frame=24)
    assert kit.hit(layout, layout.x_of(12), cy).kind == kit.ROW
    assert kit.hit(layout, row.rect[0] + 4, cy) == kit.Hit(kit.OPENER, uid=7)
    rx, ry, rw, rh = layout.ruler
    assert kit.hit(layout, layout.x_of(6), ry + rh / 2) == kit.Hit(kit.RULER, frame=6)
    play = layout.button("PLAY")
    assert kit.hit(layout, play.rect[0] + 2, play.rect[1] + 2) == kit.Hit(kit.BUTTON, id="PLAY")
    assert kit.hit(layout, 5, 5) is None


def test_the_buttons_do_not_overlap_and_light_what_is_on():
    layout = kit.dock_layout(2.0, 60, 2400, clip_with_spin(), [(7, "Fan")], 7, playing=True, ease=anim.SMOOTH)
    rects = [b.rect for b in layout.buttons]
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            apart_x = a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0]
            apart_y = a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1]
            assert apart_x or apart_y  # the tabs sit in their own strip over the bar
    assert layout.button("PLAY").on and layout.button("EASE_SMOOTH").on and layout.button("VOXEL_LOOK").on


def test_the_clip_tabs_sit_over_the_bar_and_light_the_active_one():
    layout = kit.dock_layout(1.0, 60, 1000, clip_with_spin(), [(7, "Fan")], 7, clips=["Idle", "Walk", "Attack"],
                             active_clip=1)
    tabs = [layout.button(kit.tab_id(i)) for i in range(3)]
    assert [t.label for t in tabs] == ["Idle", "Walk", "Attack"] and [t.on for t in tabs] == [False, True, False]
    assert all(t.rect[1] >= layout.bar[1] + layout.bar[3] for t in tabs)  # above the bar
    add = layout.button(kit.TAB_ADD)
    assert add is not None and add.rect[0] > tabs[-1].rect[0] + tabs[-1].rect[2]
    tx, ty, tw, th = tabs[1].rect
    assert kit.hit(layout, tx + tw / 2, ty + th / 2) == kit.Hit(kit.BUTTON, id="TAB_1")
    assert layout.rect[3] == kit.height(len(layout.rows), 1.0)
    one = kit.dock_layout(1.0, 60, 1000, clip_with_spin(), [(7, "Fan")], 7)
    assert one.button("TAB_0").label == "Clip 1" and one.button("TAB_0").on  # the one clip's own tab


def test_values_and_ticks_read_well():
    assert kit.value_text(anim.TURN_Z, 90.0) == "90°" and kit.value_text(anim.MOVE_X, 2.0) == "+2"
    layout = kit.dock_layout(1.0, 60, 1000, clip_with_spin(), [(7, "Fan")], 7)
    ticks = kit.ruler_ticks(layout, 24)
    assert ticks[0] == (0, True) and any(labelled for _f, labelled in ticks[1:])


def test_every_part_gets_a_row_indented_by_its_depth():
    layout = kit.dock_layout(1.0, 60, 1000, clip_with_spin(), [(3, "Body", 0), (7, "Fan", 1), (9, "Toe", 2)], 3)
    assert [(r.uid, r.depth) for r in layout.rows] == [(3, 0), (7, 1), (9, 2)]
    row = layout.rows[2]
    # The opener's hit area moves right with the indent; the name beyond it is a ROW.
    assert kit.hit(layout, row.rect[0] + 10, row.rect[1] + 5).kind == kit.OPENER
    assert kit.hit(layout, row.rect[0] + 18 + kit.INDENT * 2 + 4, row.rect[1] + 5) == kit.Hit(kit.ROW, uid=9, channel=None, frame=-1)


def test_the_markers_row_sits_under_the_parts_and_its_hits_name_the_frame():
    clip = clip_with_spin()
    clip.markers = [anim.Marker(6, "footstep"), anim.Marker(18, "footstep")]
    layout = kit.dock_layout(1.0, 60, 1000, clip, [(7, "Fan")], 7, onion=True)
    assert layout.markers == ((6, "footstep"), (18, "footstep"))
    mx, my, mw, mh = layout.markers_row
    assert my + mh <= layout.rows[-1].rect[1]  # under the last row
    assert kit.hit(layout, layout.x_of(6), my + mh / 2) == kit.Hit(kit.MARKER, uid=-1, channel=kit.MARKER_ROW, frame=6)
    assert kit.hit(layout, layout.x_of(12), my + mh / 2) == kit.Hit(kit.ROW, uid=-1, channel=kit.MARKER_ROW, frame=12)
    assert kit.hit(layout, mx + 5, my + mh / 2) == kit.Hit(kit.ROW, uid=-1, channel=kit.MARKER_ROW, frame=-1)
    assert layout.button("ONION").on and layout.rect[3] == kit.height(len(layout.rows), 1.0)
