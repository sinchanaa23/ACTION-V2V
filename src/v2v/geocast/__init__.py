"""Stage 4: recipient selection (P4). One selector per distribution policy."""

from collections.abc import Callable

from v2v.contracts import AssessedHazard, Distribution, VehicleState
from v2v.geocast import broadcast, circular, directional

Selector = Callable[[AssessedHazard, list[VehicleState], float], list[str]]

SELECTORS: dict[Distribution, Selector] = {
    "broadcast": broadcast.select,
    "circular": circular.select,
    "directional": directional.select,
}
