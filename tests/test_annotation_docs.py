"""The annotation guide and Label Studio config must match the contract."""

import xml.etree.ElementTree as ET
from pathlib import Path

from v2v.contracts import HazardType

DOCS = Path(__file__).parent.parent / "docs"


def test_label_config_matches_contract() -> None:
    root = ET.parse(DOCS / "label_studio_config.xml").getroot()
    hazard = root.find("Choices[@name='hazard']")
    assert hazard is not None
    assert {c.get("value") for c in hazard} == {h.value for h in HazardType}
    assert hazard.get("choice") == "single-radio"
    flags = root.find("Choices[@name='flags']")
    assert flags is not None
    assert {c.get("value") for c in flags} == {"negated", "hedged"}
    assert root.find("Labels/Label[@value='LOCATION']") is not None
    assert root.find("Text").get("value") == "$text"  # type: ignore[union-attr]


def test_guide_covers_every_class() -> None:
    guide = (DOCS / "annotation_guide.md").read_text(encoding="utf-8")
    for hazard in HazardType:
        assert f"`{hazard.value}`" in guide
