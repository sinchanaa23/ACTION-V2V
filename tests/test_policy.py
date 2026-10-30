import pytest

from v2v.confidence.policy import decide, is_actionable
from v2v.contracts import HazardInterpretation, HazardType


def _interp(
    hazard: HazardType = HazardType.ACCIDENT, negated: bool = False
) -> HazardInterpretation:
    return HazardInterpretation(report_id="r", hazard=hazard, p_class=0.9, negated=negated)


@pytest.mark.parametrize(
    ("conf", "evidence", "expected"),
    [
        (0.9, 0.8, "corroborated"),
        (0.3, 0.8, "corroborated"),
        (0.9, 0.1, "unverified"),
        (0.3, 0.1, "suppressed"),
    ],
)
def test_thresholds(conf: float, evidence: float, expected: str) -> None:
    assert decide(_interp(), conf, evidence, theta_i=0.6, theta_e=0.75) == expected


def test_negated_report_is_never_a_warning() -> None:
    interp = _interp(negated=True)
    assert not is_actionable(interp)
    assert decide(interp, 1.0, 1.0, theta_i=0.6, theta_e=0.75) == "suppressed"


@pytest.mark.parametrize("hazard", [HazardType.NON_ACTIONABLE, HazardType.AMBIGUOUS])
def test_non_hazard_classes_suppressed(hazard: HazardType) -> None:
    assert decide(_interp(hazard), 1.0, 1.0, theta_i=0.6, theta_e=0.75) == "suppressed"
