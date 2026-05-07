from __future__ import annotations


def coherence_rate(coherence_scores: list[float], floor: float) -> float:
    if not coherence_scores:
        return 0.0
    return sum(score >= floor for score in coherence_scores) / len(coherence_scores)
