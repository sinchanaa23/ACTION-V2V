"""Baseline A: keyword hazard classifier. Owner: P1 (P1-07)."""

from v2v.contracts import HazardInterpretation, HazardType

# STUB(P1-07): a placeholder list so the pipeline runs. The real keyword list is
# built from the training split and scored against it.
_KEYWORDS: dict[str, HazardType] = {
    "accident": HazardType.ACCIDENT,
    "crash": HazardType.ACCIDENT,
    "slippery": HazardType.SLIPPERY,
    "breakdown": HazardType.STOPPED,
}


def classify(report_id: str, normalised_text: str) -> HazardInterpretation:
    """Classify by first keyword hit. Keyword rules have no real probability,
    so ``p_class`` is always 1.0."""
    for keyword, hazard in _KEYWORDS.items():
        if keyword in normalised_text:
            return HazardInterpretation(report_id=report_id, hazard=hazard, p_class=1.0)
    return HazardInterpretation(report_id=report_id, hazard=HazardType.NON_ACTIONABLE, p_class=1.0)
