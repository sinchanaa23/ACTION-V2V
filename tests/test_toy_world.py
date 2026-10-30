from v2v.sim.scenarios import toy_scenario
from v2v.sim.toy_world import ToyWorld


def test_vehicles_move_forward_in_their_direction() -> None:
    world = ToyWorld(toy_scenario(), seed=1)
    before = {v.vehicle_id: v for v in world.vehicles()}
    world.step()
    for v in world.vehicles():
        assert v.lane_pos_m > before[v.vehicle_id].lane_pos_m
        if v.edge_id == "fwd":
            assert v.heading_deg == 90.0 and v.x > before[v.vehicle_id].x
        else:
            assert v.heading_deg == 270.0 and v.x < before[v.vehicle_id].x


def test_will_reach_only_upstream_same_edge() -> None:
    scenario = toy_scenario()
    world = ToyWorld(scenario, seed=0)
    hazard = scenario.hazard_location
    assert world.will_reach("fwd_0", hazard)  # behind the hazard, same direction
    assert not world.will_reach(scenario.reporter_id, hazard)  # already past it
    assert not world.will_reach("rev_0", hazard)  # opposite carriageway


def test_unstated_location_is_not_resolved() -> None:
    world = ToyWorld(toy_scenario())
    assert world.resolve_location(None, "fwd_13") is None


def test_same_seed_same_world() -> None:
    a = ToyWorld(toy_scenario(), seed=7).vehicles()
    b = ToyWorld(toy_scenario(), seed=7).vehicles()
    assert a == b
