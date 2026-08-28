"""Tests for the evaluation-only two-stage Image Prompt candidate."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from hotcards.evaluation.cli import build_parser, run_cli
from hotcards.evaluation.two_stage_image_prompts import (
    EVIDENCE_GATE_IMAGE_PROMPT_VERSION,
    TWO_STAGE_IMAGE_PROMPT_VERSION,
    ApplicableReferenceEvidenceOutput,
    EvidenceGateImagePromptPreparer,
    TwoStageImagePromptPreparer,
    build_applicable_reference_evidence_prompt,
    build_evidence_gate_synthesis_prompt,
    build_reference_account_prompt,
    build_two_stage_synthesis_prompt,
)
from hotcards.generation.errors import (
    ModelResponseError,
    ServiceUnavailableError,
)
from hotcards.generation.image_prompt_preparation import (
    ImagePromptPreparationRequest,
)
from hotcards.generation.ollama_client import OllamaRuntime, OllamaSettings


class FakeOllamaClient:
    def __init__(self, contents: list[str | Exception]) -> None:
        self.contents = contents
        self.messages: list[dict[str, object]] = []
        self.call_count = 0

    def list(self) -> SimpleNamespace:
        return SimpleNamespace(
            models=(SimpleNamespace(model="qwen3.5:9b-mlx"),)
        )

    def show(self, _model: str) -> SimpleNamespace:
        return SimpleNamespace(capabilities=("completion", "vision"))

    def chat(self, **kwargs: object) -> SimpleNamespace:
        self.messages.extend(kwargs["messages"])  # type: ignore[arg-type]
        content = self.contents[self.call_count]
        self.call_count += 1
        if isinstance(content, Exception):
            raise content
        return SimpleNamespace(
            message=SimpleNamespace(content=content),
            total_duration=2_000_000,
            load_duration=500_000,
            prompt_eval_count=100,
            eval_count=50,
            done_reason="stop",
        )


def _model_output(image_prompt: str) -> str:
    return json.dumps(
        {
            "subject_traits": "A circular metal hatch.",
            "setting_traits": "A space-station wall.",
            "visual_treatment": "Black-and-white early-Mac dithering.",
            "target_overrides": "The hatch is open.",
            "image_prompt": image_prompt,
        }
    )


def _preparer(
    contents: list[str | Exception],
) -> tuple[TwoStageImagePromptPreparer, FakeOllamaClient]:
    client = FakeOllamaClient(contents)
    runtime = OllamaRuntime(OllamaSettings(), client=client)  # type: ignore[arg-type]
    return TwoStageImagePromptPreparer(runtime), client


def _evidence_preparer(
    contents: list[str | Exception],
) -> tuple[EvidenceGateImagePromptPreparer, FakeOllamaClient]:
    client = FakeOllamaClient(contents)
    runtime = OllamaRuntime(OllamaSettings(), client=client)  # type: ignore[arg-type]
    return EvidenceGateImagePromptPreparer(runtime), client


def test_reference_account_is_target_independent() -> None:
    request = ImagePromptPreparationRequest(
        description="The hatch is open.",
        has_reference=True,
        reference_description="A closed circular hatch marked 01.",
        prompt_version=TWO_STAGE_IMAGE_PROMPT_VERSION,
    )

    prompt = build_reference_account_prompt(request)

    assert request.description not in prompt
    assert request.reference_description in prompt
    assert "before any future target is known" in prompt
    assert "Do not propose a future image" in prompt


def test_synthesis_receives_textual_account_without_pixels() -> None:
    request = ImagePromptPreparationRequest(
        description="The hatch is open.",
        has_reference=True,
        reference_description="A closed circular hatch.",
        prompt_version=TWO_STAGE_IMAGE_PROMPT_VERSION,
    )

    prompt = build_two_stage_synthesis_prompt(
        request,
        reference_account="The round metal hatch is currently closed.",
    )

    assert request.description in prompt
    assert "The round metal hatch is currently closed." in prompt
    assert "The authored Description is authoritative" in prompt
    assert "exact counts and arrangements" in prompt


def test_two_stage_preparer_separates_vision_and_synthesis(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "reference.png"
    reference_path.write_bytes(b"fixture")
    final_prompt = (
        "A circular metal space-station hatch swings open to the left as "
        "bright light pours through, rendered in black-and-white early-Mac dithering."
    )
    preparer, client = _preparer(
        [
            json.dumps(
                {
                    "reference_account": (
                        "A circular metal hatch is currently closed on a station wall, "
                        "rendered in stark black-and-white early-Mac dithering."
                    )
                }
            ),
            _model_output(final_prompt),
        ]
    )

    result = preparer.prepare(
        ImagePromptPreparationRequest(
            description=(
                "The hatch is swung open to the left, and bright light pours through."
            ),
            has_reference=True,
            reference_description="A closed circular station hatch.",
            prompt_version=TWO_STAGE_IMAGE_PROMPT_VERSION,
        ),
        reference_image_path=reference_path,
    )

    assert result.image_prompt == final_prompt
    assert [attempt.phase for attempt in result.attempts] == [
        "reference_account",
        "synthesis",
    ]
    assert client.messages[0]["images"] == [reference_path]
    assert "images" not in client.messages[1]
    assert "currently closed" in str(client.messages[1]["content"])
    assert result.total_duration_ns == 4_000_000
    assert not result.repair_applied


def test_text_only_case_skips_reference_account() -> None:
    final_prompt = "A moonlit courtyard with pale stone arches."
    preparer, client = _preparer([_model_output(final_prompt)])

    result = preparer.prepare(
        ImagePromptPreparationRequest(
            description="A moonlit courtyard with pale stone arches.",
            prompt_version=TWO_STAGE_IMAGE_PROMPT_VERSION,
        )
    )

    assert [attempt.phase for attempt in result.attempts] == ["synthesis"]
    assert client.call_count == 1
    assert "NO REFERENCE" in str(client.messages[0]["content"])


def test_evidence_selection_is_target_conditioned_and_lossy() -> None:
    request = ImagePromptPreparationRequest(
        description="A dense forest at night in the same visual style.",
        has_reference=True,
        reference_description="A laboratory in early-Mac dithering.",
        prompt_version=EVIDENCE_GATE_IMAGE_PROMPT_VERSION,
    )

    prompt = build_applicable_reference_evidence_prompt(request)

    assert request.description in prompt
    assert request.reference_description in prompt
    assert "lossy allowlist" in prompt
    assert "Include stable architecture only when setting continuity is requested" in prompt


def test_evidence_selection_requires_explicit_nullable_field() -> None:
    with pytest.raises(ValidationError):
        ApplicableReferenceEvidenceOutput.model_validate_json("{}")

    output = ApplicableReferenceEvidenceOutput.model_validate_json(
        '{"applicable_reference_evidence": null}'
    )
    assert output.applicable_reference_evidence is None


def test_evidence_synthesis_cannot_see_discarded_reference_context() -> None:
    request = ImagePromptPreparationRequest(
        description="A dense forest at night.",
        has_reference=True,
        reference_description="A laboratory with a computer.",
        prompt_version=EVIDENCE_GATE_IMAGE_PROMPT_VERSION,
    )

    prompt = build_evidence_gate_synthesis_prompt(
        request,
        applicable_reference_evidence="Black-and-white early-Mac dithering.",
    )

    assert request.description in prompt
    assert "Black-and-white early-Mac dithering." in prompt
    assert request.reference_description not in prompt
    assert "Reference was selected" not in prompt


def test_evidence_gate_separates_selection_and_synthesis(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "reference.png"
    reference_path.write_bytes(b"fixture")
    final_prompt = (
        "A dense forest at night in black-and-white early-Mac dithering."
    )
    preparer, client = _evidence_preparer(
        [
            json.dumps(
                {
                    "applicable_reference_evidence": (
                        "Black-and-white early-Mac dithering with crisp pixel textures."
                    )
                }
            ),
            _model_output(final_prompt),
        ]
    )

    result = preparer.prepare(
        ImagePromptPreparationRequest(
            description="A dense forest at night in the same visual style.",
            has_reference=True,
            reference_description="A laboratory in early-Mac dithering.",
            prompt_version=EVIDENCE_GATE_IMAGE_PROMPT_VERSION,
        ),
        reference_image_path=reference_path,
    )

    assert result.image_prompt == final_prompt
    assert [attempt.phase for attempt in result.attempts] == [
        "evidence_selection",
        "synthesis",
    ]
    assert client.messages[0]["images"] == [reference_path]
    assert "images" not in client.messages[1]
    assert "laboratory" not in str(client.messages[1]["content"])


def test_evidence_gate_retains_selection_when_synthesis_transport_fails(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "reference.png"
    reference_path.write_bytes(b"fixture")
    preparer, _client = _evidence_preparer(
        [
            json.dumps(
                {
                    "applicable_reference_evidence": (
                        "Black-and-white early-Mac dithering."
                    )
                }
            ),
            httpx.ConnectError("offline"),
        ]
    )

    with pytest.raises(ServiceUnavailableError) as caught:
        preparer.prepare(
            ImagePromptPreparationRequest(
                description="A dense forest at night.",
                has_reference=True,
                reference_description="A laboratory in early-Mac dithering.",
                prompt_version=EVIDENCE_GATE_IMAGE_PROMPT_VERSION,
            ),
            reference_image_path=reference_path,
        )

    assert [
        attempt["phase"] for attempt in caught.value.response_attempts
    ] == ["evidence_selection", "synthesis"]
    assert caught.value.response_attempts[0]["raw_response"]
    assert caught.value.response_attempts[1]["raw_response"] is None


def test_evidence_gate_records_selection_transport_failure(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "reference.png"
    reference_path.write_bytes(b"fixture")
    preparer, _client = _evidence_preparer(
        [httpx.ConnectError("offline")]
    )

    with pytest.raises(ServiceUnavailableError) as caught:
        preparer.prepare(
            ImagePromptPreparationRequest(
                description="A dense forest at night.",
                has_reference=True,
                reference_description="A laboratory in early-Mac dithering.",
                prompt_version=EVIDENCE_GATE_IMAGE_PROMPT_VERSION,
            ),
            reference_image_path=reference_path,
        )

    assert caught.value.response_attempts == (
        {"phase": "evidence_selection", "raw_response": None},
    )


def test_two_stage_candidate_uses_production_repair_contract() -> None:
    invalid = "The same hatch is open instead of closed."
    repaired = "A circular metal hatch stands open."
    preparer, client = _preparer(
        [
            _model_output(invalid),
            json.dumps({"image_prompt": repaired}),
        ]
    )

    result = preparer.prepare(
        ImagePromptPreparationRequest(
            description="A circular metal hatch stands open.",
            prompt_version=TWO_STAGE_IMAGE_PROMPT_VERSION,
        )
    )

    assert result.image_prompt == repaired
    assert [attempt.phase for attempt in result.attempts] == [
        "synthesis",
        "repair",
    ]
    assert "Detected conflict:" in str(client.messages[1]["content"])
    assert result.repair_applied


def test_two_stage_reference_repair_remains_standalone(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "reference.png"
    reference_path.write_bytes(b"fixture")
    repaired = "A circular metal hatch stands open."
    preparer, client = _preparer(
        [
            json.dumps({"reference_account": "A circular metal hatch is closed."}),
            _model_output("The same hatch is open instead of closed."),
            json.dumps({"image_prompt": repaired}),
        ]
    )

    result = preparer.prepare(
        ImagePromptPreparationRequest(
            description="A circular metal hatch stands open.",
            has_reference=True,
            reference_description="A circular metal hatch is closed.",
            prompt_version=TWO_STAGE_IMAGE_PROMPT_VERSION,
        ),
        reference_image_path=reference_path,
    )

    repair_prompt = str(client.messages[2]["content"])
    assert result.image_prompt == repaired
    assert "positive standalone description" in repair_prompt
    assert 'must call the attached\nReference exactly "image 1"' not in repair_prompt


def test_evidence_gate_reference_repair_remains_standalone(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "reference.png"
    reference_path.write_bytes(b"fixture")
    repaired = "A circular metal hatch stands open."
    preparer, client = _evidence_preparer(
        [
            json.dumps(
                {"applicable_reference_evidence": "A circular metal hatch."}
            ),
            _model_output("The same hatch is open instead of closed."),
            json.dumps({"image_prompt": repaired}),
        ]
    )

    result = preparer.prepare(
        ImagePromptPreparationRequest(
            description="A circular metal hatch stands open.",
            has_reference=True,
            reference_description="A circular metal hatch is closed.",
            prompt_version=EVIDENCE_GATE_IMAGE_PROMPT_VERSION,
        ),
        reference_image_path=reference_path,
    )

    repair_prompt = str(client.messages[2]["content"])
    assert result.image_prompt == repaired
    assert "positive standalone description" in repair_prompt
    assert 'must call the attached\nReference exactly "image 1"' not in repair_prompt


def test_synthesis_failure_retains_all_stage_metadata(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "reference.png"
    reference_path.write_bytes(b"fixture")
    preparer, _client = _preparer(
        [
            json.dumps({"reference_account": "A closed circular hatch."}),
            "",
        ]
    )

    with pytest.raises(ModelResponseError) as caught:
        preparer.prepare(
            ImagePromptPreparationRequest(
                description="The hatch is open.",
                has_reference=True,
                reference_description="A closed circular hatch.",
                prompt_version=TWO_STAGE_IMAGE_PROMPT_VERSION,
            ),
            reference_image_path=reference_path,
        )

    error = caught.value
    assert [
        attempt["phase"] for attempt in error.response_attempts
    ] == ["reference_account", "synthesis"]
    assert error.response_metadata["total_duration_ns"] == 4_000_000


def test_repair_failure_retains_failed_repair_attempt() -> None:
    preparer, _client = _preparer(
        [
            _model_output("The same hatch is open instead of closed."),
            "",
        ]
    )

    with pytest.raises(ModelResponseError) as caught:
        preparer.prepare(
            ImagePromptPreparationRequest(
                description="A circular metal hatch stands open.",
                prompt_version=TWO_STAGE_IMAGE_PROMPT_VERSION,
            )
        )

    error = caught.value
    assert [
        attempt["phase"] for attempt in error.response_attempts
    ] == ["synthesis", "repair"]
    assert error.response_metadata["total_duration_ns"] == 4_000_000


def test_cli_exposes_two_stage_candidate_options() -> None:
    args = build_parser().parse_args(
        [
            "image-prompts-two-stage",
            "--ollama-model",
            "small",
            "--repetitions",
            "3",
        ]
    )

    assert args.command == "image-prompts-two-stage"
    assert args.ollama_models == ["small"]
    assert args.repetitions == 3


def test_cli_exposes_evidence_gate_candidate_options() -> None:
    args = build_parser().parse_args(
        [
            "image-prompts-evidence-gate",
            "--ollama-model",
            "small",
            "--repetitions",
            "3",
        ]
    )

    assert args.command == "image-prompts-evidence-gate"
    assert args.ollama_models == ["small"]
    assert args.repetitions == 3


def test_cli_validates_two_stage_benchmark_without_model_calls(
    capsys: object,
) -> None:
    assert run_cli(["image-prompts-two-stage", "--validate-only"]) == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert captured.out.strip() == "evals/cases/image_prompts/benchmark.json"
