"""Import a real OpenStreetMap corridor and build a SUMO network. Owner: P4 (P4-01).

    python -m v2v.sim.osm_network --corridor hebbal

Downloads the corridor from the Overpass API at a pinned snapshot date (so
every teammate gets the same roads), then runs ``netconvert`` with left-hand
traffic. Outputs go to ``sim_out/networks/<corridor>/`` and are git-ignored.

Map data (c) OpenStreetMap contributors, ODbL.
"""

import argparse
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OSM_SNAPSHOT = "2026-10-01T00:00:00Z"
DEFAULT_OUT_DIR = Path("sim_out/networks")
USER_AGENT = "A.C.T.I.O.N-V2V research (github.com/sohini-d/A.C.T.I.O.N-V2V)"
RETRY_STATUS = {429, 502, 503, 504}

# Roads a passenger vehicle can drive on. Service roads and tracks are left out.
ROAD_CLASSES = (
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
)  # fmt: skip


@dataclass(frozen=True)
class Corridor:
    name: str
    bbox: tuple[float, float, float, float]
    """(south, west, north, east) in degrees."""
    description: str


CORRIDORS: dict[str, Corridor] = {
    "hebbal": Corridor(
        name="hebbal",
        bbox=(13.028, 77.585, 13.045, 77.605),
        description="Hebbal flyover junction, Bengaluru: Bellary Road (NH44) and Outer Ring Road",
    ),
}


def overpass_query(bbox: tuple[float, float, float, float], snapshot: str = OSM_SNAPSHOT) -> str:
    south, west, north, east = bbox
    if not (south < north and west < east):
        raise ValueError(f"bbox must be (south, west, north, east), got {bbox}")
    classes = "|".join(ROAD_CLASSES)
    return (
        f'[out:xml][timeout:180][date:"{snapshot}"][bbox:{south},{west},{north},{east}];'
        f'(way["highway"~"^({classes})$"];);'
        "(._;>;);out body;"
    )


def _fetch(request: urllib.request.Request, attempts: int, backoff_s: float) -> bytes:
    """POST to Overpass, retrying the overload responses it commonly returns."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                body: bytes = response.read()
                return body
        except urllib.error.HTTPError as err:
            if err.code not in RETRY_STATUS or attempt == attempts - 1:
                raise
        except urllib.error.URLError:
            if attempt == attempts - 1:
                raise
        time.sleep(backoff_s * 2**attempt)
    raise AssertionError("unreachable")


def download_osm(
    corridor: Corridor,
    out_path: Path,
    snapshot: str = OSM_SNAPSHOT,
    attempts: int = 5,
    backoff_s: float = 10.0,
) -> Path:
    data = urllib.parse.urlencode({"data": overpass_query(corridor.bbox, snapshot)}).encode()
    request = urllib.request.Request(OVERPASS_URL, data=data, headers={"User-Agent": USER_AGENT})
    body = _fetch(request, attempts, backoff_s)
    if b"<way" not in body:
        raise RuntimeError(f"Overpass returned no roads for {corridor.name}: {body[:300]!r}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(body)
    return out_path


def sumo_binary(name: str) -> str:
    """Locate a SUMO tool from the pip ``eclipse-sumo`` package, SUMO_HOME or PATH."""
    homes: list[str] = []
    try:
        import sumo

        homes.append(sumo.SUMO_HOME)
    except ImportError:
        pass
    if os.environ.get("SUMO_HOME"):
        homes.append(os.environ["SUMO_HOME"])
    for home in homes:
        for candidate in (Path(home) / "bin" / name, Path(home) / "bin" / f"{name}.exe"):
            if candidate.exists():
                return str(candidate)
    found = shutil.which(name)
    if found is None:
        raise FileNotFoundError(f"{name} not found: pip install -e '.[sim]' or set SUMO_HOME")
    return found


def netconvert_args(
    osm_path: Path, net_path: Path, bbox: tuple[float, float, float, float]
) -> list[str]:
    south, west, north, east = bbox
    return [
        "--osm-files", str(osm_path),
        "--output-file", str(net_path),
        "--lefthand", "true",  # India drives on the left
        # Overpass returns whole ways; drop edges lying entirely outside the box.
        "--keep-edges.in-geo-boundary", f"{west},{south},{east},{north}",
        "--keep-edges.by-vclass", "passenger",
        "--remove-edges.isolated", "true",
        "--geometry.remove", "true",
        "--ramps.guess", "true",
        "--roundabouts.guess", "true",
        "--junctions.join", "true",
        "--tls.guess-signals", "true",
        "--tls.join", "true",
        "--output.street-names", "true",
        "--output.original-names", "true",
        "--log", str(net_path.with_suffix(".log")),
    ]  # fmt: skip


def build_network(corridor: Corridor, out_dir: Path = DEFAULT_OUT_DIR, force: bool = False) -> Path:
    """Return the ``.net.xml`` for ``corridor``, downloading and converting if needed."""
    work = out_dir / corridor.name
    osm_path = work / f"{corridor.name}.osm"
    net_path = work / f"{corridor.name}.net.xml"
    if net_path.exists() and not force:
        return net_path
    if force or not osm_path.exists():
        download_osm(corridor, osm_path)
    # Warnings go to <corridor>.net.log; only surface output if conversion fails.
    done = subprocess.run(
        [sumo_binary("netconvert"), *netconvert_args(osm_path, net_path, corridor.bbox)],
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        raise RuntimeError(f"netconvert failed for {corridor.name}:\n{done.stderr[-2000:]}")
    return net_path


def read_net(net_path: Path) -> Any:
    """Load a network with sumolib (imported lazily so CI without [sim] still imports this)."""
    import sumolib

    return sumolib.net.readNet(str(net_path))


def street_names(net_path: Path) -> set[str]:
    return {e.getName() for e in read_net(net_path).getEdges() if e.getName()}


def _has_reverse(edge: Any) -> bool:
    """True if another edge runs between the same two junctions the other way."""
    return any(o.getToNode() == edge.getFromNode() for o in edge.getToNode().getOutgoing())


def network_summary(net_path: Path) -> dict[str, Any]:
    """Counts and street names, for a quick check that the right roads came through."""
    net = read_net(net_path)
    edges = [e for e in net.getEdges() if e.getFunction() != "internal"]
    names = Counter(e.getName() for e in edges if e.getName())
    return {
        "edges": len(edges),
        "junctions": len(net.getNodes()),
        "traffic_lights": len(net.getTrafficLights()),
        "length_km": round(sum(e.getLength() for e in edges) / 1000.0, 1),
        "two_way_edges": sum(1 for e in edges if _has_reverse(e)),
        "top_streets": [name for name, _ in names.most_common(12)],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a SUMO network from a real OSM corridor")
    parser.add_argument("--corridor", choices=sorted(CORRIDORS), default="hebbal")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true", help="re-download and rebuild")
    args = parser.parse_args(argv)
    net_path = build_network(CORRIDORS[args.corridor], args.out, force=args.force)
    print(net_path)
    print(json.dumps(network_summary(net_path), indent=2))


if __name__ == "__main__":
    main()
