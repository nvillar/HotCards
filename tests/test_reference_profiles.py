"""Tests for strict role-specific reference profile extraction."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hypergen.domain.models import ReferenceRole
from hypergen.generation.errors import ModelResponseError
from hypergen.generation.ollama_client import OllamaRuntime, OllamaSettings
from hypergen.generation.reference_profiles import (
    OllamaReferenceProfiler,
    ReferenceProfileRequest,
    SettingReferenceProfile,
    StyleReferenceProfile,
    SubjectReferenceProfile,
    build_reference_profile_prompt,
    profile_capsule,
)


class FakeOllamaClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []

    def chat(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            message=SimpleNamespace(content=self.content),
            total_duration=2_000_000,
            load_duration=500_000,
        )


@pytest.mark.parametrize(
    ("role", "expected_keys", "forbidden_rule"),
    (
        (
            ReferenceRole.SUBJECT,
            "identity, appearance, body, clothing, materials",
            "seated, standing, kneeling",
        ),
        (
            ReferenceRole.STYLE,
            "medium, linework, texture, palette, shading, rendering",
            "unrelated portrait, vehicle,",
        ),
        (
            ReferenceRole.SETTING,
            (
                "environment, architecture, materials, terrain, "
                "spatial_character, observed_weather, observed_time, "
                "observed_season"
            ),
            "occupied,\n  empty, crowded, abandoned",
        ),
    ),
)
def test_profile_prompt_repeats_exact_scalar_contract_and_role_rules(
    role: ReferenceRole,
    expected_keys: str,
    forbidden_rule: str,
) -> None:
    prompt = build_reference_profile_prompt(
        ReferenceProfileRequest(
            role=role,
            source_description="Synthetic adversarial source scene",
        )
    )

    assert f"keys must be exactly: {expected_keys}" in prompt
    assert "one concise JSON string or null" in prompt
    assert "Do not emit arrays, nested objects, Markdown" in prompt
    assert forbidden_rule in prompt
    assert "Synthetic adversarial source scene" in prompt


@pytest.mark.parametrize(
    ("role", "content", "expected_capsule"),
    (
        (
            ReferenceRole.SUBJECT,
            {
                "identity": "ceramic service robot",
                "appearance": "round faceplate, blue eyes",
                "body": "short articulated limbs",
                "clothing": None,
                "materials": "white ceramic, gold-repaired cracks",
            },
            (
                "ceramic service robot; round faceplate, blue eyes; "
                "short articulated limbs; white ceramic, gold-repaired cracks"
            ),
        ),
        (
            ReferenceRole.STYLE,
            {
                "medium": "dithered graphics",
                "linework": None,
                "texture": "pixelated",
                "palette": "black-and-white",
                "shading": "high-contrast",
                "rendering": None,
            },
            "dithered graphics; pixelated; black-and-white; high-contrast",
        ),
        (
            ReferenceRole.SETTING,
            {
                "environment": "mountain castle courtyard",
                "architecture": "towers, gatehouse, bridges",
                "materials": "black basalt, copper roofs",
                "terrain": "rocky ridge",
                "spatial_character": "enclosed, vertically layered",
                "observed_weather": "falling snow",
                "observed_time": "midnight",
                "observed_season": "winter",
            },
            (
                "mountain castle courtyard; towers, gatehouse, bridges; "
                "black basalt, copper roofs; rocky ridge; enclosed, "
                "vertically layered"
            ),
        ),
    ),
)
def test_profiler_parses_strict_profiles_and_composes_safe_capsules(
    role: ReferenceRole,
    content: dict[str, str | None],
    expected_capsule: str,
) -> None:
    client = FakeOllamaClient(json.dumps(content))
    runtime = OllamaRuntime(
        OllamaSettings(model="qwen3.5:9b-mlx"),
        client=client,  # type: ignore[arg-type]
    )

    result = OllamaReferenceProfiler(runtime).extract(
        ReferenceProfileRequest(
            role=role,
            source_description="Adversarial source scene",
        )
    )

    assert result.role is role
    assert result.capsule == expected_capsule
    assert result.model_identifier == "qwen3.5:9b-mlx"
    schema = client.calls[0]["format"]
    assert isinstance(schema, dict)
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize(
    ("role", "content"),
    (
        (
            ReferenceRole.SUBJECT,
            {
                "identity": ["pilot"],
                "appearance": None,
                "body": None,
                "clothing": None,
                "materials": None,
            },
        ),
        (
            ReferenceRole.STYLE,
            {
                "medium": "oil painting",
                "linework": None,
                "texture": None,
                "palette": None,
                "shading": None,
                "rendering": None,
                "subject": "woman",
            },
        ),
        (
            ReferenceRole.SETTING,
            {
                "environment": {"type": "castle"},
                "architecture": None,
                "materials": None,
                "terrain": None,
                "spatial_character": None,
                "observed_weather": None,
                "observed_time": None,
                "observed_season": None,
            },
        ),
    ),
)
def test_profiler_rejects_arrays_nested_objects_and_extra_keys(
    role: ReferenceRole,
    content: dict[str, object],
) -> None:
    runtime = OllamaRuntime(
        OllamaSettings(),
        client=FakeOllamaClient(json.dumps(content)),  # type: ignore[arg-type]
    )

    with pytest.raises(ModelResponseError, match="invalid"):
        OllamaReferenceProfiler(runtime).extract(
            ReferenceProfileRequest(
                role=role,
                source_description="Source",
            )
        )


def test_profiler_rejects_markdown_wrapped_json() -> None:
    content = (
        "```json\n"
        '{"medium":"oil painting","linework":null,"texture":null,'
        '"palette":null,"shading":null,"rendering":null}\n'
        "```\n"
    )
    runtime = OllamaRuntime(
        OllamaSettings(),
        client=FakeOllamaClient(content),  # type: ignore[arg-type]
    )

    with pytest.raises(ModelResponseError, match="invalid"):
        OllamaReferenceProfiler(runtime).extract(
            ReferenceProfileRequest(
                role=ReferenceRole.STYLE,
                source_description="Source",
            )
        )


def test_setting_conditions_are_auditable_but_never_injected() -> None:
    profile = SettingReferenceProfile(
        environment="underwater station",
        architecture="connected domes",
        observed_weather="turbid current",
        observed_time="dawn",
        observed_season="winter",
    )

    assert profile.observed_weather == "turbid current"
    assert profile_capsule(ReferenceRole.SETTING, profile) == (
        "underwater station; connected domes"
    )


def test_profile_models_forbid_invented_fields() -> None:
    with pytest.raises(ValueError):
        StyleReferenceProfile.model_validate(
            {
                "medium": "cut-paper collage",
                "subject": "chefs",
            }
        )
    with pytest.raises(ValueError):
        SubjectReferenceProfile.model_validate({"identity": "pilot", "pose": "seated"})
