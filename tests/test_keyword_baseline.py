"""Hand-written code-path strings only. Never training data."""

import json
from pathlib import Path

import pytest

from v2v.contracts import HazardType
from v2v.data.annotate import GoldRecord
from v2v.data.scoring import classification_scores, wilson_interval
from v2v.nlp import keyword_baseline
from v2v.nlp.keyword_baseline import classify, mine_keywords, predict

A = HazardType.ACCIDENT
S = HazardType.SLIPPERY
N = HazardType.NON_ACTIONABLE


def _gold(report_id: str, text: str, hazard: HazardType) -> GoldRecord:
    return GoldRecord(
        report_id=report_id,
        text=text,
        source="fixture",
        hazard=hazard,
        negated=False,
        hedged=False,
    )


@pytest.mark.parametrize(
    ("text", "hazard"),
    [
        ("truck crashed near the bridge", A),
        ("road is flooded here", S),
        ("car stalled in the left lane", HazardType.STOPPED),
        ("someone on the wrong side", HazardType.SPEEDING),
        ("nice weather today", N),
    ],
)
def test_classify_first_hit(text: str, hazard: HazardType) -> None:
    result = classify("r", text)
    assert result.hazard is hazard
    assert result.p_class == 1.0


def test_classify_matches_word_starts_only() -> None:
    assert classify("r", "no embracing here").hazard is N


def test_baseline_stays_naive_about_negation() -> None:
    assert classify("r", "accident nahi hua").hazard is A


def test_mined_words_extend_the_seed_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "keywords.json"
    path.write_text(json.dumps({"slippery_road": ["fisalan"]}), encoding="utf-8")
    monkeypatch.setattr(keyword_baseline, "KEYWORDS_PATH", path)
    keyword_baseline._rules.cache_clear()
    try:
        assert classify("r", "bahut fisalan hai").hazard is S
    finally:
        keyword_baseline._rules.cache_clear()


def test_mine_keywords_finds_distinctive_words() -> None:
    records = [_gold(f"a{i}", "takkar near the market", A) for i in range(6)]
    records += [_gold(f"n{i}", "lovely evening near the market", N) for i in range(6)]
    mined = mine_keywords(records)
    assert mined["accident"] == ["takkar"]
    assert "market" not in mined["accident"]
    assert set(mined) == {h.value for h in HazardType if h not in {N, HazardType.AMBIGUOUS}}


def test_predict_uses_the_normaliser() -> None:
    assert predict([_gold("1", "ACCIDENT ahead", A)]) == [{"report_id": "1", "hazard": "accident"}]


def test_classification_scores_known_values() -> None:
    scores = classification_scores([A, A, S, S], [A, S, S, S])
    assert scores["accuracy"] == 0.75
    assert scores["per_class"]["accident"]["recall"] == 0.5
    assert scores["per_class"]["accident"]["precision"] == 1.0
    assert scores["per_class"]["slippery_road"]["precision"] == pytest.approx(2 / 3)
    assert scores["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)


def test_classification_scores_handles_a_class_never_predicted() -> None:
    scores = classification_scores([A], [S])
    assert scores["per_class"]["accident"]["f1"] == 0.0
    assert scores["macro_f1"] == 0.0


def test_classification_scores_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        classification_scores([], [])


def test_wilson_interval() -> None:
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.490, abs=0.001)
    assert high == pytest.approx(0.943, abs=0.001)
    assert wilson_interval(0, 5)[0] == 0.0
    with pytest.raises(ValueError):
        wilson_interval(0, 0)
