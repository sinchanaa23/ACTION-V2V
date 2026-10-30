"""Import a real OpenStreetMap corridor and build a SUMO network. Owner: P4 (P4-01).

Map data (c) OpenStreetMap contributors, ODbL.
"""

from pathlib import Path


def build_network(bbox: tuple[float, float, float, float], out_dir: Path) -> Path:
    """Download the (south, west, north, east) box and run netconvert; return
    the ``.net.xml`` path. Generated files are git-ignored.

    STUB(P4-01).
    """
    raise NotImplementedError("P4-01: OSM import")
