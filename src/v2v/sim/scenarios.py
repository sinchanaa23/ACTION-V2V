"""Scenario generation. Owner: P4 (P4-03)."""

from v2v.contracts import ResolvedLocation, Scenario


def toy_scenario(seed: int = 0) -> Scenario:
    """A hazard on the toy world's ``fwd`` edge at 1200 m, reported at t=10 s by
    ``fwd_13``, which starts just past it.

    STUB(P4-03): the real generator places hazards at real OSM landmarks and
    adds duplicates, contradictions, stale reports and three densities.
    """
    return Scenario(
        scenario_id=f"toy-{seed}",
        hazard_location=ResolvedLocation(
            edge_id="fwd", pos_m=1200.0, x=1200.0, y=0.0, approx=False
        ),
        event_occurred=True,
        reporter_id="fwd_13",
        report_time_s=10.0,
    )
