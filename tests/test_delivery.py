import pytest

from v2v.contracts import ResolvedLocation, VehicleState, WarningMessage
from v2v.sim.delivery import DeliveryModel, deliver, reception_probability

HAZARD = ResolvedLocation(edge_id="e", pos_m=0, x=0, y=0, approx=False)


class LineWorld:
    """Vehicles strung along the x axis; ``relevant`` ones will reach the hazard."""

    def __init__(self, xs: dict[str, float], relevant: set[str] | None = None) -> None:
        self._v = {
            vid: VehicleState(
                vehicle_id=vid, x=x, y=0, edge_id="e", lane_pos_m=0, heading_deg=0, speed_mps=10
            )
            for vid, x in xs.items()
        }
        self._relevant = relevant or set()

    @property
    def time_s(self) -> float:
        return 0.0

    def step(self) -> None:
        pass

    def vehicles(self) -> list[VehicleState]:
        return list(self._v.values())

    def vehicle(self, vehicle_id: str) -> VehicleState:
        return self._v[vehicle_id]

    def resolve_location(self, location_text: str | None, reporter_id: str) -> None:
        return None

    def sensor_flags(self, location: ResolvedLocation, radius_m: float) -> int:
        return 0

    def will_reach(self, vehicle_id: str, location: ResolvedLocation) -> bool:
        return vehicle_id in self._relevant

    def close(self) -> None:
        pass


def _msg(recipients: list[str], msg_id: str = "m0") -> WarningMessage:
    return WarningMessage(
        msg_id=msg_id,
        cluster_id="c0",
        text="t",
        status="unverified",
        policy="directional",
        recipients=tuple(recipients),
        t_sent=0.0,
    )


def test_reception_probability_shape() -> None:
    assert reception_probability(0, 300) == pytest.approx(1.0)
    assert reception_probability(300, 300) == pytest.approx(0.4232, abs=1e-3)
    assert reception_probability(600, 300) < 0.003
    distances = [0, 100, 200, 300, 400, 600]
    probabilities = [reception_probability(d, 300) for d in distances]
    assert probabilities == sorted(probabilities, reverse=True)


def test_close_recipient_gets_it_fast_in_one_hop() -> None:
    world = LineWorld({"near": 30.0}, relevant={"near"})
    outcome = DeliveryModel().deliver(_msg(["near"]), world, HAZARD)
    (record,) = outcome.records
    assert record.delivered and record.relevant
    assert record.latency_ms is not None and 2.0 <= record.latency_ms <= 11.0
    assert outcome.hops == {"near": 1}
    assert outcome.transmissions == 2  # sender, then "near" rebroadcasts once


def test_far_recipient_needs_relays() -> None:
    chain = {f"v{i}": 150.0 * i for i in range(1, 7)}  # 150 m apart, out to 900 m
    world = LineWorld(chain)
    single = DeliveryModel(multi_hop=False).deliver(_msg(list(chain)), world, HAZARD)
    multi = DeliveryModel().deliver(_msg(list(chain)), world, HAZARD)
    by_id = {r.vehicle_id: r for r in multi.records}
    assert not {r.vehicle_id: r for r in single.records}["v6"].delivered  # 900 m, one hop
    assert by_id["v6"].delivered and multi.hops["v6"] >= 2
    assert by_id["v6"].latency_ms > by_id["v1"].latency_ms  # type: ignore[operator]
    assert multi.transmissions > single.transmissions == 1


def test_only_addressed_vehicles_relay() -> None:
    chain = {f"v{i}": 150.0 * i for i in range(1, 7)}
    world = LineWorld(chain)
    outcome = DeliveryModel().deliver(_msg(["v6"]), world, HAZARD)  # nobody in between relays
    assert not outcome.records[0].delivered
    assert outcome.transmissions == 1
    assert outcome.receptions >= 1  # unaddressed neighbours still hear it: channel load


def test_max_hops_caps_the_flood() -> None:
    chain = {f"v{i}": 150.0 * i for i in range(1, 7)}
    outcome = DeliveryModel(max_hops=2).deliver(_msg(list(chain)), LineWorld(chain), HAZARD)
    assert max(outcome.hops.values()) <= 2
    assert not {r.vehicle_id: r for r in outcome.records}["v6"].delivered


def test_sender_position_is_the_origin() -> None:
    world = LineWorld({"sender": 1000.0, "near_sender": 1020.0, "near_hazard": 10.0})
    outcome = DeliveryModel(multi_hop=False).deliver(
        _msg(["near_sender", "near_hazard"]), world, HAZARD, sender_id="sender"
    )
    by_id = {r.vehicle_id: r for r in outcome.records}
    assert by_id["near_sender"].delivered and not by_id["near_hazard"].delivered


def test_unknown_sender_falls_back_to_hazard_point() -> None:
    world = LineWorld({"near_hazard": 10.0})
    (record,) = deliver(_msg(["near_hazard"]), world, HAZARD, sender_id="gone")
    assert record.delivered


def test_same_message_same_outcome_and_recipients_that_left_are_lost() -> None:
    chain = {f"v{i}": 150.0 * i for i in range(1, 7)}
    world = LineWorld(chain)
    a = DeliveryModel().deliver(_msg(list(chain)), world, HAZARD)
    b = DeliveryModel().deliver(_msg(list(chain)), world, HAZARD)
    assert a == b
    (gone,) = DeliveryModel().deliver(_msg(["not_here"]), world, HAZARD).records
    assert not gone.delivered and gone.latency_ms is None
