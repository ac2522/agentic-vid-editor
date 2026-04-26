"""VLM-judge Inspect AI scorer wrapper.

Iterates the scenario's render rubric, dispatches each dimension to the
ensemble, returns a single Score (1 if all dimensions pass, 0 otherwise).
Per-dimension verdicts go in metadata for downstream consumers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState

from ave.harness.judges._protocol import JudgeBackend
from ave.harness.judges.ensemble import judge_dimension_ensemble
from ave.harness.schema import Scenario


@scorer(metrics=[])
def vlm_judge_scorer(judges: Sequence[JudgeBackend]) -> Scorer:
    judge_list = list(judges)

    async def score(state: TaskState, target: Target) -> Score:
        meta = state.metadata or {}
        scenario: Scenario = meta["scenario"]
        rendered_path_str = meta.get("rendered_path")
        render_failed = bool(meta.get("render_failed", False))
        render_expected = scenario.expected.render

        if render_expected is None:
            return Score(value=1, explanation="no render expectations; skipping")

        if render_failed:
            return Score(
                value=0, explanation="render failed; cannot judge",
                metadata={"render_failed": True},
            )

        if not render_expected.rubric:
            return Score(value=1, explanation="no render rubric; skipping")

        if not rendered_path_str:
            return Score(
                value=0, explanation="render produced no output; cannot judge",
                metadata={"render_failed": True},
            )

        rendered_path = Path(rendered_path_str)
        results = []
        all_passed = True
        for dim in render_expected.rubric:
            r = judge_dimension_ensemble(
                rendered_path=rendered_path,
                dimension=dim.dimension, prompt=dim.prompt,
                pass_threshold=dim.pass_threshold,
                veto=dim.veto, judges=judge_list,
            )
            if not r.passed:
                all_passed = False
            results.append({
                "dimension": r.dimension,
                "passed": r.passed,
                "score": r.aggregated_score,
                "rule": r.rule,
                "explanation": r.explanation,
                "disagreement": r.disagreement,
                "individual_verdicts": [
                    {"judge": v.judge_name, "score": v.score, "passed": v.passed,
                     "explanation": v.explanation}
                    for v in r.individual_verdicts
                ],
            })

        return Score(
            value=1 if all_passed else 0,
            explanation="; ".join(
                f"{r['dimension']}={'pass' if r['passed'] else 'fail'}" for r in results
            ),
            metadata={"rubric_results": results},
        )

    return score
