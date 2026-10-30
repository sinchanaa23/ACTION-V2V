"""Landmark list: named bridges, flyovers and junctions mapped to edges. Owner: P4 (P4-02)."""

from v2v.contracts import ResolvedLocation


def lookup(location_text: str, near_x: float, near_y: float) -> ResolvedLocation | None:
    """Nearest landmark matching ``location_text`` around (near_x, near_y).

    STUB(P4-02).
    """
    raise NotImplementedError("P4-02: gazetteer")
