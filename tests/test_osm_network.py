import importlib.util
import io
import os
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path

import pytest

from v2v.sim.osm_network import (
    CORRIDORS,
    build_network,
    download_osm,
    netconvert_args,
    overpass_query,
)


def test_query_is_pinned_to_snapshot_and_bbox() -> None:
    query = overpass_query(CORRIDORS["hebbal"].bbox)
    assert '[date:"2026-10-01T00:00:00Z"]' in query
    assert "[bbox:13.028,77.585,13.045,77.605]" in query
    assert "trunk_link" in query and "service" not in query


def test_query_rejects_swapped_bbox() -> None:
    with pytest.raises(ValueError):
        overpass_query((13.045, 77.585, 13.028, 77.605))


def test_netconvert_is_left_hand_and_keeps_street_names() -> None:
    args = netconvert_args(Path("a.osm"), Path("a.net.xml"), CORRIDORS["hebbal"].bbox)
    assert args[args.index("--lefthand") + 1] == "true"
    assert args[args.index("--keep-edges.in-geo-boundary") + 1] == "77.585,13.028,77.605,13.045"
    assert args[args.index("--output.street-names") + 1] == "true"
    assert args[args.index("--keep-edges.by-vclass") + 1] == "passenger"


class _FlakyServer:
    """Fails with the given HTTP codes, then returns a body."""

    def __init__(self, codes: list[int], body: bytes) -> None:
        self.codes = codes
        self.body = body
        self.calls = 0

    def __call__(self, request: object, timeout: float) -> io.BytesIO:
        self.calls += 1
        if self.codes:
            code = self.codes.pop(0)
            raise urllib.error.HTTPError("u", code, "err", Message(), None)
        return io.BytesIO(self.body)


def test_download_retries_overload_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _FlakyServer([502, 429], b"<osm><way id='1'/></osm>")
    monkeypatch.setattr(urllib.request, "urlopen", server)
    out = download_osm(CORRIDORS["hebbal"], tmp_path / "a.osm", backoff_s=0)
    assert server.calls == 3
    assert out.read_bytes().startswith(b"<osm>")


def test_download_does_not_retry_client_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _FlakyServer([400], b"")
    monkeypatch.setattr(urllib.request, "urlopen", server)
    with pytest.raises(urllib.error.HTTPError):
        download_osm(CORRIDORS["hebbal"], tmp_path / "a.osm", backoff_s=0)
    assert server.calls == 1


def test_download_rejects_empty_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", _FlakyServer([], b"<osm></osm>"))
    with pytest.raises(RuntimeError, match="no roads"):
        download_osm(CORRIDORS["hebbal"], tmp_path / "a.osm", backoff_s=0)


@pytest.mark.skipif(
    importlib.util.find_spec("sumolib") is None or not os.environ.get("V2V_NETWORK_TESTS"),
    reason="needs the [sim] extras and network access; set V2V_NETWORK_TESTS=1",
)
def test_hebbal_network_builds(tmp_path: Path) -> None:
    import subprocess

    from v2v.sim.osm_network import network_summary, street_names, sumo_binary

    net_path = build_network(CORRIDORS["hebbal"], tmp_path)
    summary = network_summary(net_path)
    assert summary["edges"] > 1000
    assert summary["traffic_lights"] > 0
    assert {"Hebbal Flyover", "Bellary Road", "Outer Ring Road"} <= street_names(net_path)
    # SUMO itself must accept the network.
    subprocess.run(
        [sumo_binary("sumo"), "-n", str(net_path), "--end", "1", "--no-step-log"], check=True
    )
