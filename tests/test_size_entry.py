import pytest

from rok.core.grid_ops import MAX_SIZE
from rok.core.size_entry import backspace, parse_size, start_text, type_char


@pytest.mark.parametrize(
    "text, size",
    [
        ("40 30 20", (40, 30, 20)),
        ("40x30x20", (40, 30, 20)),
        ("40 x 30 x 20", (40, 30, 20)),
        ("40, 30, 20", (40, 30, 20)),
        ("  64 ", (64, 64, 64)),  # one number: a cube
    ],
)
def test_a_typed_size_reads_as_people_type_it(text, size):
    assert parse_size(text) == size


@pytest.mark.parametrize("text", ["", "40 30", "1 2 3 4", "0 10 10", f"{MAX_SIZE + 1}", "abc"])
def test_what_is_not_a_size_is_refused(text):
    assert parse_size(text) is None


def test_the_field_opens_with_the_size_and_the_first_key_replaces_it():
    text, fresh = start_text((24, 30, 20)), True
    assert text == "24 30 20" and parse_size(text) == (24, 30, 20)
    text, fresh = type_char(text, fresh, "4")
    assert (text, fresh) == ("4", False)
    for char in "0 40q 40":  # a letter that is no separator is ignored
        text, fresh = type_char(text, fresh, char)
    assert text == "40 40 40" and parse_size(text) == (40, 40, 40)
    assert backspace(text, fresh) == ("40 40 4", False)
    assert backspace("24 30 20", True) == ("", False)  # Backspace on the selection clears it
