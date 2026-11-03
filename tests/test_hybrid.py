import pytest
from app.vectordb.hybrid import rrf_fuse


def _item(i: int) -> dict[str, int]:
    return {"id": i}


def test_rrf_formula_matches_hand_calculation() -> None:
    dense = [_item(1), _item(2)]
    sparse = [_item(2), _item(3)]
    fused = {r["id"]: r["rrf_score"] for r in rrf_fuse([dense, sparse])}
    assert fused[1] == pytest.approx(1 / 61)
    assert fused[2] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[3] == pytest.approx(1 / 62)


def test_item_in_both_lists_beats_item_in_one() -> None:
    dense = [_item(1), _item(2)]
    sparse = [_item(2), _item(3)]
    assert rrf_fuse([dense, sparse])[0]["id"] == 2


def test_top_n_limits_results() -> None:
    ranked = [[_item(i) for i in range(10)]]
    assert len(rrf_fuse(ranked, top_n=3)) == 3
