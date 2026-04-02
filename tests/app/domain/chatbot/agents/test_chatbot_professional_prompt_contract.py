from datetime import date

from app.domain.chatbot.agents.prompts.chatbot_prompts import (
    build_response_guard_prompt,
    build_spendwise_coach_prompt,
)


def test_build_spendwise_coach_prompt_requires_professional_diagnostic_coaching_contract() -> None:
    prompt = build_spendwise_coach_prompt(today=date(2026, 3, 27)).casefold()

    assert "start with a one-sentence diagnosis" in prompt
    assert "prioritize the one or two biggest drivers" in prompt
    assert "label it as a likely pattern" in prompt
    assert "separate controllable spending from unavoidable burden" in prompt
    assert "what to do, when to do it, and why it matters" in prompt
    assert "avoid vague advice like" in prompt


def test_build_response_guard_prompt_rejects_generic_or_ungrounded_coaching() -> None:
    prompt = build_response_guard_prompt().casefold()

    assert "reject generic encouragement" in prompt
    assert "reject unsupported numeric claims" in prompt
    assert "reject answers that present inference as fact" in prompt
    assert "reject next steps that are not concrete" in prompt


def test_prompt_contract_requires_user_friendly_translation_of_internal_analysis_labels() -> None:
    coach_prompt = build_spendwise_coach_prompt(today=date(2026, 3, 27)).casefold()
    guard_prompt = build_response_guard_prompt().casefold()

    assert "translate internal labels into user-facing language" in coach_prompt
    assert "do not expose raw internal status labels" in coach_prompt
    assert "do not expose raw field names" in coach_prompt
    assert "reject backend labels" in guard_prompt
    assert "reject raw field names" in guard_prompt
    assert "reject raw key-value snippets" in guard_prompt
