"""Shared data contracts for the A.C.T.I.O.N-V2V pipeline.

Every module talks to every other module only through the types in this file.
Changing anything here needs a PR labelled ``contract-change`` and approval
from all four members (P1-P4).

Flow: DriverReport -> HazardInterpretation -> AssessedHazard -> WarningMessage
-> DeliveryRecord, with VehicleState / ResolvedLocation / Scenario supplied by
the simulation (``World``).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Protocol, get_args

from pydantic import BaseModel, ConfigDict, Field

Source = Literal["survey", "reddit", "youtube", "news_dataset", "x", "fir", "fixture"]
"""Where a report's text came from. ``fixture`` marks hand-written strings used
only by tests and the demo run; scripts/check_data_sources.py rejects it
anywhere under ``data/``."""

REAL_SOURCES: frozenset[str] = frozenset(s for s in get_args(Source) if s != "fixture")

Status = Literal["corroborated", "unverified", "suppressed"]
HazardProcessing = Literal["keyword", "nlp"]
Distribution = Literal["broadcast", "circular", "directional"]

Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class _Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class HazardType(StrEnum):
    ACCIDENT = "accident"
    OBSTRUCTION = "road_obstruction"
    SLIPPERY = "slippery_road"
    STOPPED = "stopped_vehicle"
    SPEEDING = "speeding_concern"
    NON_ACTIONABLE = "non_actionable"
    AMBIGUOUS = "ambiguous"


NON_HAZARD_TYPES: frozenset[HazardType] = frozenset(
    {HazardType.NON_ACTIONABLE, HazardType.AMBIGUOUS}
)


class DriverReport(_Contract):
    """One informal driver report. P1 builds these from the dataset; P4's
    scenarios decide which vehicle sends one and when."""

    report_id: str
    text: str = Field(min_length=1)
    source: Source
    reporter_id: str | None = None
    t_sim: float | None = None


class HazardInterpretation(_Contract):
    """P2's reading of a report. ``p_class`` is the calibrated probability of
    ``hazard``. Missing details stay None; they are never invented."""

    report_id: str
    hazard: HazardType
    p_class: Probability
    location_text: str | None = None
    negated: bool = False
    hedged: bool = False
    details: dict[str, str] = Field(default_factory=dict)


class VehicleState(_Contract):
    """A vehicle's state as reported by the simulation. ``heading_deg`` follows
    SUMO's convention: 0 = north, 90 = east, clockwise."""

    vehicle_id: str
    x: float
    y: float
    edge_id: str
    lane_pos_m: float = Field(ge=0.0)
    heading_deg: float = Field(ge=0.0, lt=360.0)
    speed_mps: float = Field(ge=0.0)
    route_ahead: tuple[str, ...] = ()
    """Edges still to drive after ``edge_id``, in order."""


class ResolvedLocation(_Contract):
    """A point on the road network. ``approx`` is True when it came from the
    fallback policy rather than from a location stated in the report."""

    edge_id: str
    pos_m: float = Field(ge=0.0)
    x: float
    y: float
    approx: bool


class AssessedHazard(_Contract):
    """P3's verdict on a cluster of reports. Interpretation confidence and
    event evidence are deliberately separate fields."""

    cluster_id: str
    hazard: HazardType
    location: ResolvedLocation | None
    interp_conf: Probability
    event_evidence: Probability
    n_reports: int = Field(ge=1)
    age_s: float = Field(ge=0.0)
    status: Status


class WarningMessage(_Contract):
    """The structured warning P2 words and P4 addresses."""

    msg_id: str
    cluster_id: str
    text: str
    status: Status
    policy: Distribution
    recipients: tuple[str, ...]
    t_sent: float


class DeliveryRecord(_Contract):
    """One recipient's outcome. ``relevant`` is ground truth from the
    scenario: will this vehicle actually reach the hazard?"""

    msg_id: str
    vehicle_id: str
    delivered: bool
    latency_ms: float | None = Field(default=None, ge=0.0)
    relevant: bool


class Scenario(_Contract):
    """Ground truth for one run. The report text is real; the scenario decides
    who reports it, when, and whether the event really happened."""

    scenario_id: str
    hazard_location: ResolvedLocation
    event_occurred: bool
    reporter_id: str
    report_time_s: float = Field(ge=0.0)


class ExperimentConfig(_Contract):
    """One row of the experiment design (A, B, proposed, two ablations)."""

    name: str
    hazard_processing: HazardProcessing
    confidence: bool
    distribution: Distribution
    theta_i: Probability = 0.6
    theta_e: Probability = 0.75
    radius_m: float = Field(default=800.0, gt=0.0)


class RunResult(_Contract):
    """Everything one pipeline run produced, for P3's metrics and P4's dashboard."""

    config_name: str
    seed: int
    scenario_id: str
    event_occurred: bool
    report: DriverReport
    interpretation: HazardInterpretation
    assessed: AssessedHazard
    warning: WarningMessage | None
    deliveries: tuple[DeliveryRecord, ...]
    relevant_vehicle_ids: tuple[str, ...]
    """Ground truth at send time: every vehicle that will reach the hazard."""


class World(Protocol):
    """What the pipeline needs from a traffic simulation (toy world or SUMO)."""

    @property
    def time_s(self) -> float: ...

    def step(self) -> None: ...

    def vehicles(self) -> list[VehicleState]: ...

    def vehicle(self, vehicle_id: str) -> VehicleState: ...

    def resolve_location(
        self, location_text: str | None, reporter_id: str
    ) -> ResolvedLocation | None: ...

    def sensor_flags(self, location: ResolvedLocation, radius_m: float) -> int: ...

    def will_reach(self, vehicle_id: str, location: ResolvedLocation) -> bool: ...

    def close(self) -> None: ...
