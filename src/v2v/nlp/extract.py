"""Location, negation and hedging extraction. Owner: P2 (P2-02, P2-03, P2-06)."""


def extract_location(normalised_text: str) -> str | None:
    """Return the location phrase stated in the report, or None.

    STUB(P2-03): always None. Must never guess a location.
    """
    return None


def detect_negation(normalised_text: str) -> bool:
    """True if the hazard is negated ("accident nahi hua").

    STUB(P2-02): always False.
    """
    return False


def detect_hedging(normalised_text: str) -> bool:
    """True if the report is hedged ("shayad", "lagta hai").

    STUB(P2-02): always False.
    """
    return False
