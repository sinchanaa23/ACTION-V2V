"""The five experiment configurations from the proposal. Owner: P3."""

from v2v.contracts import ExperimentConfig

BASELINE_A = ExperimentConfig(
    name="baseline_a", hazard_processing="keyword", confidence=False, distribution="broadcast"
)
BASELINE_B = ExperimentConfig(
    name="baseline_b", hazard_processing="nlp", confidence=False, distribution="circular"
)
PROPOSED = ExperimentConfig(
    name="proposed", hazard_processing="nlp", confidence=True, distribution="directional"
)
ABLATION_DIRECTION = ExperimentConfig(
    name="ablation_direction", hazard_processing="nlp", confidence=True, distribution="circular"
)
ABLATION_CONFIDENCE = ExperimentConfig(
    name="ablation_confidence",
    hazard_processing="nlp",
    confidence=False,
    distribution="directional",
)

CONFIGS: dict[str, ExperimentConfig] = {
    c.name: c for c in (BASELINE_A, BASELINE_B, PROPOSED, ABLATION_DIRECTION, ABLATION_CONFIDENCE)
}
