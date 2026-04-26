"""Tests for the multi-judge ensemble aggregation."""
from pathlib import Path

from ave.harness.judges._protocol import JudgeVerdict


class _FakeJudge:
    def __init__(self, name, supported, score, explanation="ok"):
        self._name = name
        self._supported = supported
        self._score = score
        self._explanation = explanation
    @property
    def name(self): return self._name
    @property
    def supported_dimension_types(self): return self._supported
    def judge_dimension(self, *, rendered_path, dimension, prompt, pass_threshold):
        return JudgeVerdict(
            judge_name=self._name, dimension=dimension,
            score=self._score, passed=self._score >= pass_threshold,
            explanation=self._explanation,
        )


def test_ensemble_majority_vote_pass():
    from ave.harness.judges.ensemble import judge_dimension_ensemble
    judges = [
        _FakeJudge("a", ("still",), 0.8),
        _FakeJudge("b", ("still",), 0.7),
        _FakeJudge("c", ("still",), 0.3),
    ]
    result = judge_dimension_ensemble(
        rendered_path=Path("/x.mp4"),
        dimension="framing",
        prompt="subject centered",
        pass_threshold=0.5,
        veto=False,
        judges=judges,
    )
    assert result.passed is True
    assert len(result.individual_verdicts) == 3


def test_ensemble_majority_vote_fail():
    from ave.harness.judges.ensemble import judge_dimension_ensemble
    judges = [
        _FakeJudge("a", ("still",), 0.3),
        _FakeJudge("b", ("still",), 0.4),
        _FakeJudge("c", ("still",), 0.6),
    ]
    result = judge_dimension_ensemble(
        rendered_path=Path("/x.mp4"),
        dimension="framing", prompt="x",
        pass_threshold=0.5, veto=False, judges=judges,
    )
    assert result.passed is False


def test_ensemble_minority_veto_one_fail_kills_pass():
    """When veto=True, ANY single failure fails the dimension."""
    from ave.harness.judges.ensemble import judge_dimension_ensemble
    judges = [
        _FakeJudge("a", ("still",), 0.9),
        _FakeJudge("b", ("still",), 0.9),
        _FakeJudge("c", ("still",), 0.2),
    ]
    result = judge_dimension_ensemble(
        rendered_path=Path("/x.mp4"),
        dimension="framing", prompt="x",
        pass_threshold=0.5, veto=True, judges=judges,
    )
    assert result.passed is False


def test_ensemble_no_compatible_judges_passes_with_warning():
    """No compatible judges -> pass=True with explanation noting absence."""
    from ave.harness.judges.ensemble import judge_dimension_ensemble
    judges = [_FakeJudge("ff", ("static",), 1.0)]
    result = judge_dimension_ensemble(
        rendered_path=Path("/x.mp4"),
        dimension="framing", prompt="x",
        pass_threshold=0.5, veto=False, judges=judges,
    )
    assert "no compatible" in result.explanation.lower() or "no judges" in result.explanation.lower()


def test_ensemble_records_disagreement():
    """When judges disagree, the result should mark disagreement."""
    from ave.harness.judges.ensemble import judge_dimension_ensemble
    judges = [
        _FakeJudge("a", ("still",), 0.9),
        _FakeJudge("b", ("still",), 0.1),
    ]
    result = judge_dimension_ensemble(
        rendered_path=Path("/x.mp4"),
        dimension="x", prompt="x",
        pass_threshold=0.5, veto=False, judges=judges,
    )
    assert result.disagreement is True


def test_ensemble_unanimous_no_disagreement():
    from ave.harness.judges.ensemble import judge_dimension_ensemble
    judges = [
        _FakeJudge("a", ("still",), 0.9),
        _FakeJudge("b", ("still",), 0.85),
    ]
    result = judge_dimension_ensemble(
        rendered_path=Path("/x.mp4"),
        dimension="x", prompt="x",
        pass_threshold=0.5, veto=False, judges=judges,
    )
    assert result.disagreement is False
