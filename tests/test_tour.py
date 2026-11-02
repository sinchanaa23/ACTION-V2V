import math
import sys
from pathlib import Path

import pytest

SIM3D = Path(__file__).resolve().parents[1] / "dashboard" / "sim3d"


@pytest.fixture
def tour(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    monkeypatch.syspath_prepend(str(SIM3D))
    import tour

    yield tour
    sys.modules.pop("tour", None)


def _shots(tour):  # type: ignore[no-untyped-def]
    return [
        tour.Shot((0.0, 0.0, 0.0), 350.0, 20.0, 2000.0, 40.0),
        tour.Shot((100.0, 5.0, 50.0), 10.0, 10.0, 500.0, 46.0),
        tour.Shot((200.0, 0.0, 80.0), 80.0, 50.0, 300.0, 50.0),
    ]


def test_each_chapter_settles_exactly_on_its_shot(tour) -> None:  # type: ignore[no-untyped-def]
    shots = _shots(tour)
    for i, shot in enumerate(shots):
        pose = tour.pose_at(shots, float(i))
        assert pose.focus == pytest.approx(shot.focus)
        assert pose.yaw == pytest.approx(shot.yaw)
        assert pose.dist == pytest.approx(shot.dist)
    assert tour.pose_at(shots, -3.0).dist == pytest.approx(2000.0)  # clamped
    assert tour.pose_at(shots, 9.0).dist == pytest.approx(300.0)


def test_the_path_between_shots_is_smooth_and_turns_the_short_way(tour) -> None:  # type: ignore[no-untyped-def]
    shots = _shots(tour)
    poses = [tour.pose_at(shots, k / 100) for k in range(101)]
    yaws = tour.unwrap([p.yaw for p in poses])
    assert max(abs(b - a) for a, b in zip(yaws, yaws[1:], strict=False)) < 2.0  # 350 -> 10 via 0
    assert yaws[-1] - yaws[0] == pytest.approx(20.0)
    dists = [p.dist for p in poses]
    assert all(b <= a + 1e-6 for a, b in zip(dists, dists[1:], strict=False))  # zooms in steadily


def test_scroll_glides_after_the_wheel_and_stays_in_range(tour) -> None:  # type: ignore[no-untyped-def]
    s = tour.Scroll(6)
    s.wheel(-5)
    assert s.target == 0.0
    s.wheel(4)
    assert s.target == pytest.approx(1.0)  # four notches per chapter
    s.update(0.1)
    assert 0.0 < s.value < 1.0  # it glides, it does not jump
    for _ in range(300):  # settles within five seconds
        s.update(1 / 60)
    assert s.value == pytest.approx(1.0) and s.chapter == 1
    s.go(99)
    assert s.target == 5.0


def test_blend_and_parallax(tour) -> None:  # type: ignore[no-untyped-def]
    a, b = tour.Shot((0, 0, 0), 350.0, 10.0, 100.0), tour.Shot((10, 0, 0), 10.0, 30.0, 400.0)
    mid = tour.blend(a, b, 0.5)
    assert mid.yaw == pytest.approx(0.0) and mid.dist == pytest.approx(200.0)  # geometric zoom
    assert tour.blend(a, b, 2.0) == tour.blend(a, b, 1.0)
    p = tour.parallax(a, 0.5, 0.5)
    assert p.yaw == pytest.approx(353.5) and p.pitch == pytest.approx(8.0)
    assert tour.parallax(a, 0.0, 0.5, strength=10).pitch == 4.0  # never under the ground


def test_principal_axis_of_a_road(tour) -> None:  # type: ignore[no-untyped-def]
    north = [(0.0, float(y)) for y in range(10)]
    east = [(float(x), 0.0) for x in range(10)]
    diagonal = [(float(t), float(t)) for t in range(10)]
    assert tour.principal_axis(north) == pytest.approx(0.0)
    assert tour.principal_axis(east) == pytest.approx(90.0)
    assert tour.principal_axis(diagonal) == pytest.approx(45.0)


def test_chapters_follow_the_scene_and_the_config(tour) -> None:  # type: ignore[no-untyped-def]
    common = dict(
        hazard_name="road obstruction",
        place="Hebbal flyover",
        lanes=3089,
        buildings=10216,
        decks=51,
        radius_m=800.0,
    )
    hazard = (12.0, 7.0, -40.0)
    out = tour.chapters(
        (3785.0, 2512.0), 20.0, (1, 0, 1), None, hazard, distribution="directional", **common
    )
    assert [c.key for c in out] == ["intro", "corridor", "flyovers", "hazard", "geocast", "explore"]
    by = {c.key: c for c in out}
    assert by["hazard"].shot.focus == hazard and by["geocast"].shot.focus == hazard
    assert by["flyovers"].shot.focus == (1, 0, 1)  # no deck: falls back to the road
    assert "3,089 lanes" in by["corridor"].caption and "51 raised" in by["flyovers"].caption
    assert "800 m upstream" in by["geocast"].caption
    assert by["geocast"].shot.dist >= 2 * 800  # the whole zone is in view
    broadcast = tour.chapters(
        (3785.0, 2512.0), 20.0, (1, 0, 1), None, hazard, distribution="broadcast", **common
    )
    assert "needed or not" in broadcast[4].caption
    assert all(math.isfinite(c.shot.dist) for c in broadcast)
