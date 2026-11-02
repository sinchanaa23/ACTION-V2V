"""P2-06. Hand-written fixture records (source="fixture"): code paths only."""

from pathlib import Path

import pytest

from v2v.contracts import HazardType
from v2v.data.annotate import GoldRecord, Span
from v2v.nlp import span_tagger


def _rec(rid: str, text: str, *places: str) -> GoldRecord:
    spans = []
    for place in places:
        start = text.index(place)
        spans.append(Span(start=start, end=start + len(place), text=place))
    return GoldRecord(
        report_id=rid,
        text=text,
        source="fixture",
        hazard=HazardType.ACCIDENT,
        negated=False,
        hedged=False,
        locations=tuple(spans),
    )


_TRAIN = [
    _rec("a", "Accident near Hebbal flyover, slow down", "Hebbal flyover"),
    _rec("b", "Esteem mall ke paas truck kharab hai", "Esteem mall"),
    _rec("c", "Oil on the road near the petrol pump", "petrol pump"),
    _rec("d", "Bhai aage gaddha hai bridge ke paas", "bridge"),
    _rec("e", "Accident hua hai, careful"),
    _rec("f", "Traffic jam on Outer Ring Road", "Outer Ring Road"),
    _rec("g", "Speeding bike, be careful"),
]


def test_tokenize_and_bio_labels() -> None:
    text = "jam near Hebbal flyover now"
    tokens = span_tagger.tokenize(text)
    assert [t[0] for t in tokens] == ["jam", "near", "Hebbal", "flyover", "now"]
    start = text.index("Hebbal")
    labels = span_tagger.bio_labels(tokens, [(start, start + len("Hebbal flyover"))])
    assert labels == ["O", "O", "B", "I", "O"]
    assert span_tagger.bio_labels(tokens, []) == ["O"] * 5


def test_adjacent_spans_get_separate_b() -> None:
    tokens = span_tagger.tokenize("a b")
    assert span_tagger.bio_labels(tokens, [(0, 1), (2, 3)]) == ["B", "B"]


def test_decode_copies_spans_from_text() -> None:
    text = "jam near Hebbal Flyover and the Signal"
    tokens = span_tagger.tokenize(text)
    tags = ["O", "O", "B", "I", "O", "O", "I"]  # a stray I starts a span
    assert span_tagger.decode(tokens, tags, text) == ["Hebbal Flyover", "Signal"]


def test_features_mark_place_words_and_neighbours() -> None:
    tokens = span_tagger.tokenize("gaddha bridge ke paas")
    f = span_tagger.token_features(tokens, 1)
    assert f["place"] == "strong" and f["L1"] == "gaddha" and f["R1"] == "ke"
    assert span_tagger.token_features(tokens, 0)["L1"] == "<s>"


def test_span_scores() -> None:
    gold = [["Hebbal flyover"], [], ["bridge"]]
    pred = [["hebbal  Flyover"], [], ["the bridge"]]
    s = span_tagger.span_scores(gold, pred)
    assert s["exact"]["precision"] == 0.5 and s["exact"]["recall"] == 0.5
    assert s["token_overlap"]["recall"] == 1.0
    assert s["no_location_kept_none"] == 1.0 and s["n"] == 3


def test_train_tag_save_load_and_compare(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    tagger = span_tagger.SpanTagger()
    tagger.fit(_TRAIN)
    found = tagger.find_locations("Accident near Hebbal flyover")
    assert all(s in "Accident near Hebbal flyover" for s in found)  # never invented
    path = tmp_path / "t.pkl"
    tagger.save(path)
    loaded = span_tagger.SpanTagger.load(path)
    assert loaded.find_locations("Accident near Hebbal flyover") == found
    result = span_tagger.compare(_TRAIN, loaded)
    assert set(result) == {"rules", "tagger"}
    assert result["tagger"]["n"] == len(_TRAIN)


def test_fit_needs_location_tokens() -> None:
    pytest.importorskip("sklearn")
    with pytest.raises(ValueError):
        span_tagger.SpanTagger().fit([_rec("x", "no place here")])


def test_cli_refuses_test_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frozen test split"):
        span_tagger.main(["evaluate", "--split", str(tmp_path / "splits" / "test")])


def test_cli_train_and_evaluate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("sklearn")
    for name in ("train", "val"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "reports.jsonl").write_text(
            "".join(r.model_dump_json() + "\n" for r in _TRAIN), encoding="utf-8"
        )
    model = tmp_path / "m.pkl"
    span_tagger.main(
        [
            "train",
            "--train",
            str(tmp_path / "train"),
            "--val",
            str(tmp_path / "val"),
            "--out",
            str(model),
        ]
    )
    span_tagger.main(["evaluate", "--split", str(tmp_path / "val"), "--model", str(model)])
    assert '"tagger"' in capsys.readouterr().out


def test_extract_location_uses_tagger_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("sklearn")
    from v2v.nlp import extract

    tagger = span_tagger.SpanTagger()
    tagger.fit(_TRAIN)
    path = tmp_path / "t.pkl"
    tagger.save(path)
    text = "accident near hebbal flyover"
    monkeypatch.setenv(extract.SPAN_TAGGER_ENV_VAR, str(path))
    extract._span_tagger.cache_clear()
    assert extract.extract_location(text) == tagger.extract_location(text)
    monkeypatch.setenv(extract.SPAN_TAGGER_ENV_VAR, str(tmp_path / "absent.pkl"))
    assert extract.extract_location(text) == "hebbal flyover"  # rules
    extract._span_tagger.cache_clear()
