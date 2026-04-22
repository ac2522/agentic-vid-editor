"""Pure scope-enforcement evaluator tests."""

from ave.agent.domains import Domain
from ave.agent.registry import ToolRegistry
from ave.harness.evaluators.scope import evaluate_scope


def _registry_with_domain_tools() -> ToolRegistry:
    reg = ToolRegistry()

    def do_audio(x: int) -> None:
        """Audio op."""

    def do_video(x: int) -> None:
        """Video op."""

    reg.register("do_audio", do_audio, domain="audio", domains_touched=(Domain.AUDIO,))
    reg.register("do_video", do_video, domain="video", domains_touched=(Domain.VIDEO,))
    return reg


def test_scope_passes_when_no_forbidden_domain_touched():
    reg = _registry_with_domain_tools()
    v = evaluate_scope(
        called_tools=["do_audio"],
        registry=reg,
        forbidden_domains=("video",),
    )
    assert v.passed is True


def test_scope_fails_when_forbidden_domain_touched():
    reg = _registry_with_domain_tools()
    v = evaluate_scope(
        called_tools=["do_video"],
        registry=reg,
        forbidden_domains=("video",),
    )
    assert v.passed is False
    assert "do_video" in v.reason
    assert "video" in v.reason.lower()


def test_scope_passes_when_forbidden_is_empty():
    reg = _registry_with_domain_tools()
    v = evaluate_scope(
        called_tools=["do_audio", "do_video"],
        registry=reg,
        forbidden_domains=(),
    )
    assert v.passed is True


def test_scope_tolerates_unknown_tool():
    """Tools not in the registry (e.g., typo) are reported but don't crash scope."""
    reg = _registry_with_domain_tools()
    v = evaluate_scope(
        called_tools=["mystery_tool"],
        registry=reg,
        forbidden_domains=("video",),
    )
    # The scope evaluator can't prove a violation for an unknown tool — passes.
    assert v.passed is True
    assert "unknown tools ignored" in v.reason


def test_scope_flags_multi_domain_tool_with_one_forbidden_domain():
    """A tool touching several domains fails scope if any are forbidden."""
    reg = ToolRegistry()

    def mixed_op(x: int) -> None:
        """Op that touches both audio and video layers."""

    reg.register(
        "mixed_op",
        mixed_op,
        domain="compositing",
        domains_touched=(Domain.AUDIO, Domain.VIDEO),
    )

    v = evaluate_scope(
        called_tools=["mixed_op"],
        registry=reg,
        forbidden_domains=("video",),
    )
    assert v.passed is False
    assert "video" in v.reason.lower()
    assert v.rule == "scope_violation"


def test_scope_rule_tags_are_set():
    reg = _registry_with_domain_tools()
    assert (
        evaluate_scope(called_tools=["do_audio"], registry=reg, forbidden_domains=("video",)).rule
        == "scope_respected"
    )
    assert (
        evaluate_scope(called_tools=["do_video"], registry=reg, forbidden_domains=("video",)).rule
        == "scope_violation"
    )
