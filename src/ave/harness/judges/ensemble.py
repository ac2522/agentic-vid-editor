"""Multi-judge ensemble: majority-vote default, minority-veto for safety dims."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ave.harness.judges._protocol import JudgeBackend, JudgeVerdict
from ave.harness.judges.router import classify_dimension, select_judges


@dataclass(frozen=True)
class EnsembleResult:
    dimension: str
    passed: bool
    aggregated_score: float
    explanation: str
    individual_verdicts: tuple[JudgeVerdict, ...]
    disagreement: bool
    rule: str


def judge_dimension_ensemble(
    *,
    rendered_path: Path,
    dimension: str,
    prompt: str,
    pass_threshold: float,
    veto: bool,
    judges: Sequence[JudgeBackend],
) -> EnsembleResult:
    """Dispatch to compatible judges, aggregate via majority/veto."""
    dim_type = classify_dimension(dimension)
    compatible = select_judges(judges, dimension_type=dim_type)

    if not compatible:
        return EnsembleResult(
            dimension=dimension, passed=True,
            aggregated_score=0.0,
            explanation=f"no compatible judges for type={dim_type}; defaulting pass",
            individual_verdicts=(), disagreement=False,
            rule="no_compatible_judges",
        )

    verdicts = tuple(
        j.judge_dimension(
            rendered_path=rendered_path, dimension=dimension,
            prompt=prompt, pass_threshold=pass_threshold,
        )
        for j in compatible
    )
    pass_count = sum(1 for v in verdicts if v.passed)
    avg_score = sum(v.score for v in verdicts) / len(verdicts)
    disagreement = not all(v.passed == verdicts[0].passed for v in verdicts)

    if veto and pass_count < len(verdicts):
        return EnsembleResult(
            dimension=dimension, passed=False,
            aggregated_score=avg_score,
            explanation=f"veto failure: {pass_count}/{len(verdicts)} judges passed (all required)",
            individual_verdicts=verdicts, disagreement=disagreement,
            rule="veto_fail",
        )

    majority_passes = pass_count * 2 > len(verdicts)
    return EnsembleResult(
        dimension=dimension, passed=majority_passes,
        aggregated_score=avg_score,
        explanation=f"{pass_count}/{len(verdicts)} judges passed; avg score {avg_score:.2f}",
        individual_verdicts=verdicts, disagreement=disagreement,
        rule="majority_pass" if majority_passes else "majority_fail",
    )
