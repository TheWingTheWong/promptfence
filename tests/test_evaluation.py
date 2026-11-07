import json

import pytest

from promptfence.evaluation import confusion, load_dataset


def test_confusion_metrics():
    m = confusion([1, 1, 0, 0, 1], [True, False, True, False, True])
    assert (m.tp, m.fp, m.tn, m.fn) == (2, 1, 1, 1)
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 3)
    assert m.f1 == pytest.approx(2 / 3)
    assert m.false_positive_rate == pytest.approx(0.5)


def test_empty_metrics_do_not_divide_by_zero():
    m = confusion([], [])
    assert m.precision == m.recall == m.f1 == m.accuracy == 0.0


def test_load_dataset_filters_split(tmp_path):
    path = tmp_path / "d.jsonl"
    rows = [{"text": "a", "label": 0, "category": "x", "split": "train"},
            {"text": "b", "label": 1, "category": "y", "split": "test"}]  # fmt: skip
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert [e.text for e in load_dataset(path, "test")] == ["b"]
    assert len(load_dataset(path)) == 2


def test_bundled_dataset_is_well_formed():
    from pathlib import Path

    data = Path(__file__).resolve().parent.parent / "data" / "dataset.jsonl"
    examples = load_dataset(data)
    assert len(examples) > 400
    assert {e.split for e in examples} == {"train", "test"}
    assert len({e.text for e in examples}) == len(examples)
