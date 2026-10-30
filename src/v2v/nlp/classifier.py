"""Stage 2: NLP hazard interpretation. Owner: P2 (P2-01, P2-05)."""

from v2v.contracts import HazardInterpretation
from v2v.nlp import calibrate, extract, keyword_baseline


def interpret(report_id: str, normalised_text: str) -> HazardInterpretation:
    """Classify a report and attach extracted details and a calibrated probability.

    STUB(P2-01): delegates to the keyword baseline until the TF-IDF and
    transformer models exist.
    """
    raw = keyword_baseline.classify(report_id, normalised_text)
    return raw.model_copy(
        update={
            "p_class": calibrate.calibrate(raw.p_class),
            "location_text": extract.extract_location(normalised_text),
            "negated": extract.detect_negation(normalised_text),
            "hedged": extract.detect_hedging(normalised_text),
        }
    )
