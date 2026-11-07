import re

from promptfence.mapped import MappedText


def test_identity_maps_each_char_to_itself():
    m = MappedText.identity("abc")
    assert m.span(0, 3) == (0, 3)
    assert m.span(1, 2) == (1, 2)


def test_sub_same_length_keeps_one_to_one_mapping():
    m = MappedText.identity("xAyBz").sub(r"[A-Z]", lambda mt: mt.group(0).lower())
    assert m.text == "xaybz"
    assert m.span(3, 4) == (3, 4)


def test_sub_shrinking_maps_replacement_to_whole_match():
    m = MappedText.identity("a<b>c</b>d").sub(re.compile(r"</?b>"), "")
    assert m.text == "acd"
    assert m.span(1, 2) == (4, 5)  # "c" still points at the original "c"
    assert m.span(0, 3) == (0, 10)


def test_sub_expanding_points_at_source_entity():
    m = MappedText.identity("x &amp; y").sub(r"&amp;", "&")
    assert m.text == "x & y"
    assert m.span(2, 3) == (2, 7)


def test_sub_without_changes_returns_same_object():
    m = MappedText.identity("hello")
    assert m.sub(r"z", "q") is m


def test_map_chars_can_delete_and_expand():
    m = MappedText.identity("a​ﬁ").map_chars(
        lambda ch: "" if ch == "​" else ("fi" if ch == "ﬁ" else ch)
    )
    assert m.text == "afi"
    assert m.span(1, 3) == (2, 3)


def test_compose_through_two_transforms():
    outer = MappedText.identity("<p>hi</p>").sub(r"</?p>", "")
    inner = MappedText.identity(outer.text).sub(r"h", "H").compose(outer)
    assert inner.text == "Hi"
    assert inner.span(0, 2) == (3, 5)


def test_constant_mapping():
    m = MappedText.constant("decoded", 10, 30)
    assert m.span(2, 4) == (10, 30)
