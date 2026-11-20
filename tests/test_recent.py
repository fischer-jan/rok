"""The recent-files list logic."""

from rok.core.recent import LIMIT, parse, remember, serialise


def test_remember_puts_the_path_first_once_and_caps_the_list():
    entries = ["/a.vox", "/b.vox", "/c.vox"]
    assert remember(entries, "/b.vox") == ["/b.vox", "/a.vox", "/c.vox"]
    assert remember(entries, "/d.vox") == ["/d.vox", "/a.vox", "/b.vox", "/c.vox"]
    assert entries == ["/a.vox", "/b.vox", "/c.vox"]  # the input is not changed
    many = [f"/{i}.vox" for i in range(LIMIT)]
    out = remember(many, "/new.vox")
    assert len(out) == LIMIT and out[0] == "/new.vox" and out[-1] == f"/{LIMIT - 2}.vox"
    assert remember([], "/only.vox", limit=1) == ["/only.vox"]


def test_parse_and_serialise_round_trip_and_clean_up():
    text = "/a.vox\n\n  /b.vox  \n/a.vox\n"
    assert parse(text) == ["/a.vox", "/b.vox"]
    assert serialise(["/a.vox", "/b.vox"]) == "/a.vox\n/b.vox\n"
    assert parse(serialise(["/x y.vox", "/z.vox"])) == ["/x y.vox", "/z.vox"]
    assert parse("") == [] and serialise([]) == ""
