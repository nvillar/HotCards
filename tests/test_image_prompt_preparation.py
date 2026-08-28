"""Tests for optional-reference Image Prompt preparation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from hypergen.generation.errors import ModelResponseError, ModelUnavailableError
from hypergen.generation.image_prompt_preparation import (
    IMAGE_PROMPT_PREPARATION_VERSION,
    MULTI_REFERENCE_IMAGE_PROMPT_PREPARATION_VERSION,
    ImagePromptPreparationRequest,
    OllamaImagePromptPreparer,
    build_image_prompt_preparation_prompt,
)
from hypergen.generation.ollama_client import OllamaRuntime, OllamaSettings


class FakeOllamaClient:
    def __init__(
        self,
        content: str | list[str],
        *,
        capabilities: tuple[str, ...] = ("completion", "vision"),
    ) -> None:
        self.contents = [content] if isinstance(content, str) else content
        self.capabilities = capabilities
        self.messages: list[dict[str, object]] = []
        self.call_count = 0

    def list(self) -> SimpleNamespace:
        return SimpleNamespace(
            models=(SimpleNamespace(model="qwen3.5:9b-mlx"),)
        )

    def show(self, _model: str) -> SimpleNamespace:
        return SimpleNamespace(capabilities=self.capabilities)

    def chat(self, **kwargs: object) -> SimpleNamespace:
        self.messages.extend(kwargs["messages"])  # type: ignore[arg-type]
        content = self.contents[min(self.call_count, len(self.contents) - 1)]
        self.call_count += 1
        return SimpleNamespace(
            message=SimpleNamespace(content=content),
            total_duration=2_000_000,
            load_duration=500_000,
        )


def _output(
    image_prompt: str,
    *,
    visual_treatment: str = "high-contrast halftone linework",
) -> str:
    return json.dumps(
        {
            "subject_traits": "boxy CRT terminal",
            "setting_traits": None,
            "visual_treatment": visual_treatment,
            "target_overrides": "screen displays static",
            "image_prompt": image_prompt,
        }
    )


def _preparer(content: str | list[str]) -> tuple[OllamaImagePromptPreparer, FakeOllamaClient]:
    client = FakeOllamaClient(content)
    runtime = OllamaRuntime(OllamaSettings(), client=client)  # type: ignore[arg-type]
    return OllamaImagePromptPreparer(runtime), client


def test_prompt_defines_one_reviewable_result_without_clarification() -> None:
    prompt = build_image_prompt_preparation_prompt(
        ImagePromptPreparationRequest(
            description="The screen of the computer has changed.",
            has_reference=True,
            reference_description=(
                "Black-and-white dithered graphics reminiscent of early Mac "
                "and HyperCard."
            ),
        )
    )

    assert "A vague requested change may be resolved as a plausible concrete proposal" in prompt
    assert "Do not ask a" in prompt
    assert "question or discuss ambiguity" in prompt
    assert "Inspect the attached Reference image directly" in prompt
    assert 'call it exactly\n  "image 1"' in prompt
    assert '"reference card"' in prompt
    assert '"reference picture"' in prompt
    assert "inside the vehicle shown in image 1" in prompt
    assert "direct FLUX.2 editing instruction for image 1" in prompt
    assert "Include only the desired result, its relationship to image 1" in prompt
    assert "state only the important stable identity" in prompt
    assert "Never preserve a property that the authored Description changes" in prompt
    assert "append blanket preservation" in prompt
    assert "Aim for at most 100" in prompt
    assert "same visual style" in prompt
    assert "reference_generation_description" in prompt
    assert "primary semantic interpretation" in prompt
    assert "newspaper print" in prompt
    assert '"the monitor"' in prompt
    assert "Private deliberation is a checklist" in prompt
    assert "do not turn a monochrome Reference into a beige" in prompt
    assert "without requiring the same wording" in prompt
    assert "target_overrides detail" in prompt
    assert 'no "EXIT" text' in prompt
    assert "private deliberation" in prompt
    assert "Image Prompt" in prompt
    assert prompt.index(
        '"reference_generation_description"'
    ) < prompt.index('"authored_description"')
    assert prompt.index(
        "Black-and-white dithered graphics reminiscent of early Mac and HyperCard."
    ) < prompt.index("The screen of the computer has changed.")
    assert IMAGE_PROMPT_PREPARATION_VERSION == "image-prompt-preparation-v9"


def test_text_only_prompt_does_not_claim_an_attached_reference() -> None:
    prompt = build_image_prompt_preparation_prompt(
        ImagePromptPreparationRequest(description="A moonlit courtyard")
    )

    assert "NO REFERENCE" in prompt
    assert "Inspect the attached Reference image directly" not in prompt
    assert "direct FLUX.2 editing instruction for image 1" not in prompt
    assert "append blanket preservation" not in prompt
    assert "positive, standalone description" in prompt


def test_multi_reference_prompt_uses_only_multi_reference_contract() -> None:
    prompt = build_image_prompt_preparation_prompt(
        ImagePromptPreparationRequest(
            description="Place the Explorer in the Hangar.",
            has_reference=True,
            reference_description="A compact angular exploration vehicle.",
            reference_card_name="Explorer",
            additional_reference_description="A vast monochrome station hangar.",
            additional_reference_card_name="Hangar",
            prompt_version=MULTI_REFERENCE_IMAGE_PROMPT_PREPARATION_VERSION,
        )
    )

    assert "MULTI-REFERENCE INTERPRETATION" in prompt
    assert "Image 1 is the primary continuity source" in prompt
    assert '"card_name": "Explorer"' in prompt
    assert '"card_name": "Hangar"' in prompt
    assert "exact first selected card name as image 1" in prompt
    assert "exact second selected card name as image 2" in prompt
    assert "\nREFERENCE INTERPRETATION\n" not in prompt
    assert "first and only input image" not in prompt
    assert "NO REFERENCE" not in prompt


def test_multi_reference_preparer_attaches_images_in_order(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    first_path.write_bytes(b"first")
    second_path.write_bytes(b"second")
    prompt = (
        "Place the vehicle from image 1 inside the hangar from image 2, "
        "preserving the vehicle's angular construction."
    )
    preparer, client = _preparer(_output(prompt))

    result = preparer.prepare(
        ImagePromptPreparationRequest(
            description="Place the Explorer in the Hangar.",
            has_reference=True,
            reference_description="A compact angular exploration vehicle.",
            reference_card_name="Explorer",
            additional_reference_description="A vast monochrome station hangar.",
            additional_reference_card_name="Hangar",
            prompt_version=MULTI_REFERENCE_IMAGE_PROMPT_PREPARATION_VERSION,
        ),
        reference_image_path=first_path,
        additional_reference_image_path=second_path,
    )

    assert result.image_prompt == prompt
    assert result.prompt_version == MULTI_REFERENCE_IMAGE_PROMPT_PREPARATION_VERSION
    assert client.messages[0]["images"] == [first_path, second_path]


def test_request_rejects_empty_description() -> None:
    with pytest.raises(ValidationError):
        ImagePromptPreparationRequest(description="")


def test_preparer_attaches_reference_and_parses_one_image_prompt(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "reference.png"
    image_path.write_bytes(b"fixture")
    prompt = (
        "Update the boxy CRT computer terminal in image 1 so its screen is "
        "filled with dense static, preserving the high-contrast halftone linework."
    )
    preparer, client = _preparer(_output(prompt))

    result = preparer.prepare(
        ImagePromptPreparationRequest(
            description="The screen now shows dense static.",
            has_reference=True,
            reference_description="Early Mac and HyperCard dithered graphics.",
        ),
        reference_image_path=image_path,
    )

    assert result.image_prompt == prompt
    assert result.model_identifier == "qwen3.5:9b-mlx"
    assert result.total_duration_ns == 2_000_000
    assert client.call_count == 1
    assert client.messages[0]["images"] == [image_path]


def test_preparer_repairs_missing_image_one_attribution(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "reference.png"
    image_path.write_bytes(b"fixture")
    repaired = (
        "Move the camera inside the vehicle shown in image 1, framing the "
        "airlock through its windshield."
    )
    preparer, client = _preparer(
        [
            _output("Viewed from inside a small vehicle, an airlock fills the windshield."),
            json.dumps({"image_prompt": repaired}),
        ]
    )

    result = preparer.prepare(
        ImagePromptPreparationRequest(
            description="View from inside the vehicle in the reference card.",
            has_reference=True,
            reference_description="A small vehicle faces an airlock.",
        ),
        reference_image_path=image_path,
    )

    assert result.image_prompt == repaired
    assert result.repair_applied
    assert client.call_count == 2
    assert "must identify the attached Reference as 'image 1'" in str(
        client.messages[-1]["content"]
    )
    assert "name only the important stable properties" in str(
        client.messages[-1]["content"]
    )
    assert "never preserve a property that the authored Description changes" in str(
        client.messages[-1]["content"]
    )


def test_preparer_accepts_direct_image_one_edit_language(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "reference.png"
    image_path.write_bytes(b"fixture")
    prompt = (
        "Move the camera in image 1 inside its vehicle, replacing the exterior "
        "vehicle with an angular cockpit while preserving the same visual style."
    )
    preparer, client = _preparer(_output(prompt))

    result = preparer.prepare(
        ImagePromptPreparationRequest(
            description="View from inside the vehicle in the reference picture.",
            has_reference=True,
            reference_description="A vehicle in a monochrome hangar.",
        ),
        reference_image_path=image_path,
    )

    assert result.image_prompt == prompt
    assert client.call_count == 1


def test_preparer_repairs_noncanonical_reference_alias(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "reference.png"
    image_path.write_bytes(b"fixture")
    repaired = "Change the vehicle in image 1 into a view from its cockpit."
    preparer, client = _preparer(
        [
            _output("Use image 1, the reference image, as the vehicle identity."),
            json.dumps({"image_prompt": repaired}),
        ]
    )

    result = preparer.prepare(
        ImagePromptPreparationRequest(
            description="Show the vehicle from the reference image from inside.",
            has_reference=True,
            reference_description="A small angular vehicle.",
        ),
        reference_image_path=image_path,
    )

    assert result.image_prompt == repaired
    assert client.call_count == 2
    assert "must use 'image 1'" in str(client.messages[-1]["content"])


@pytest.mark.parametrize(
    "invalid_prompt",
    (
        "Use image 1 to create a moonlit courtyard.",
        "Use the Reference to create a moonlit courtyard.",
        "Use the reference card to create a moonlit courtyard.",
        "Use the reference picture to create a moonlit courtyard.",
        "Use the input image to create a moonlit courtyard.",
    ),
)
def test_text_only_preparer_repairs_input_image_language(
    invalid_prompt: str,
) -> None:
    repaired = "A moonlit courtyard with pale stone arches."
    preparer, client = _preparer(
        [
            _output(invalid_prompt),
            json.dumps({"image_prompt": repaired}),
        ]
    )

    result = preparer.prepare(
        ImagePromptPreparationRequest(description="A moonlit courtyard with pale stone arches.")
    )

    assert result.image_prompt == repaired
    assert client.call_count == 2


def test_reference_preparer_allows_reference_as_scene_terminology(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "reference.png"
    image_path.write_bytes(b"fixture")
    prompt = (
        "Add reference grid lines behind the object from image 1 while "
        "preserving its high-contrast treatment."
    )
    preparer, client = _preparer(_output(prompt))

    result = preparer.prepare(
        ImagePromptPreparationRequest(
            description="Add reference grid lines behind the object.",
            has_reference=True,
            reference_description="A geometric object in high contrast.",
        ),
        reference_image_path=image_path,
    )

    assert result.image_prompt == prompt
    assert client.call_count == 1


def test_reference_preparer_repairs_bare_reference_alias(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "reference.png"
    image_path.write_bytes(b"fixture")
    repaired = "Preserve the vehicle from image 1 while moving the camera inside it."
    preparer, client = _preparer(
        [
            _output("Use image 1 while preserving the Reference vehicle."),
            json.dumps({"image_prompt": repaired}),
        ]
    )

    result = preparer.prepare(
        ImagePromptPreparationRequest(
            description="Move inside the vehicle in the Reference.",
            has_reference=True,
            reference_description="A small angular vehicle.",
        ),
        reference_image_path=image_path,
    )

    assert result.image_prompt == repaired
    assert client.call_count == 2
    assert "must use 'image 1'" in str(client.messages[-1]["content"])


def test_preparer_requires_a_vision_capable_model() -> None:
    client = FakeOllamaClient(
        _output("A moonlit courtyard"),
        capabilities=("completion",),
    )
    runtime = OllamaRuntime(OllamaSettings(), client=client)  # type: ignore[arg-type]

    with pytest.raises(
        ModelUnavailableError,
        match="lacks required capabilities: vision",
    ):
        OllamaImagePromptPreparer(runtime).prepare(
            ImagePromptPreparationRequest(description="A moonlit courtyard")
        )

    assert client.call_count == 0


def test_ambiguous_change_returns_a_proposal_without_a_clarification_state() -> None:
    proposed = (
        "A boxy CRT computer terminal in high-contrast halftone linework. "
        "The screen displays dense static."
    )
    preparer, _client = _preparer(_output(proposed))

    result = preparer.prepare(
        ImagePromptPreparationRequest(
            description="The screen of the computer has changed."
        )
    )

    assert result.image_prompt == proposed


def test_preparer_repairs_comparison_language() -> None:
    preparer, client = _preparer(
        [
            _output(
                "The same computer now shows ERROR instead of the previous diagram."
            ),
            json.dumps(
                {
                    "image_prompt": (
                        'A boxy CRT computer displays a large red "ERROR" message.'
                    )
                }
            ),
        ]
    )

    result = preparer.prepare(
        ImagePromptPreparationRequest(
            description='The computer displays a large red "ERROR" message.'
        )
    )

    assert result.image_prompt == (
        'A boxy CRT computer displays a large red "ERROR" message.'
    )
    assert client.call_count == 2
    assert result.repair_applied
    assert [attempt.phase for attempt in result.attempts] == [
        "initial",
        "repair",
    ]
    assert result.duration_seconds >= sum(
        attempt.duration_seconds for attempt in result.attempts
    )
    assert result.total_duration_ns == 4_000_000
    assert "Detected conflict:" in client.messages[-1]["content"]


def test_preparer_retains_initial_attempt_when_repair_call_is_empty() -> None:
    preparer, client = _preparer(
        [
            _output(
                "The same computer now shows ERROR instead of the previous diagram."
            ),
            "",
        ]
    )

    with pytest.raises(ModelResponseError) as caught:
        preparer.prepare(
            ImagePromptPreparationRequest(
                description='The computer displays a large red "ERROR" message.'
            )
        )

    error = caught.value
    assert client.call_count == 2
    assert [attempt["phase"] for attempt in error.response_attempts] == [
        "initial",
        "repair",
    ]
    assert error.response_attempts[0]["raw_response"]
    assert error.response_attempts[1]["raw_response"] == ""
    assert error.response_metadata["total_duration_ns"] == 4_000_000


def test_preparer_counts_empty_initial_model_response_as_an_attempt() -> None:
    preparer, client = _preparer("")

    with pytest.raises(ModelResponseError) as caught:
        preparer.prepare(
            ImagePromptPreparationRequest(description="A moonlit courtyard.")
        )

    error = caught.value
    assert client.call_count == 1
    assert len(error.response_attempts) == 1
    assert error.response_attempts[0]["phase"] == "initial"
    assert error.response_attempts[0]["raw_response"] == ""
    assert error.response_metadata["total_duration_ns"] == 2_000_000


def test_preparer_allows_non_process_uses_of_source_and_original() -> None:
    proposed = (
        "An original oil painting of a courtyard illuminated by a warm light source."
    )
    preparer, client = _preparer(_output(proposed))

    result = preparer.prepare(
        ImagePromptPreparationRequest(
            description=(
                "An original oil painting of a courtyard with a warm light source."
            )
        )
    )

    assert result.image_prompt == proposed
    assert client.call_count == 1


def test_preparer_allows_authored_process_words_as_visible_text() -> None:
    proposed = 'A poster reads "ORIGINAL IMAGE" in bold black lettering.'
    preparer, client = _preparer(_output(proposed))

    result = preparer.prepare(
        ImagePromptPreparationRequest(description=proposed)
    )

    assert result.image_prompt == proposed
    assert client.call_count == 1


def test_preparer_repairs_invented_color_under_monochrome_treatment(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "reference.png"
    image_path.write_bytes(b"fixture")
    preparer, client = _preparer(
        [
            _output(
                "Show the CRT monitor from image 1 as an old beige close-up "
                "rendered in high-contrast black and white halftone.",
                visual_treatment="high-contrast black and white halftone",
            ),
            json.dumps(
                {
                    "image_prompt": (
                        "Show the distinctive computer monitor from image 1 in "
                        "a close-up rendered in high-contrast black-and-white halftone."
                    )
                }
            ),
        ]
    )

    result = preparer.prepare(
        ImagePromptPreparationRequest(
            description="A close-up view of the computer monitor.",
            has_reference=True,
            reference_description="Black-and-white dithered graphics.",
        ),
        reference_image_path=image_path,
    )

    assert "beige" not in result.image_prompt
    assert "black-and-white halftone" in result.image_prompt
    assert client.call_count == 2


def test_preparer_accepts_semantically_preserved_authored_style() -> None:
    prompt = (
        "A stark futuristic lab with an open hidden doorway, rendered as "
        "black-and-white dithered early-Mac HyperCard graphics."
    )
    preparer, client = _preparer(_output(prompt))

    result = preparer.prepare(
        ImagePromptPreparationRequest(
            description=(
                "A stark futuristic lab with an open hidden doorway. "
                "HyperCard, early-Mac black and white style."
            )
        )
    )

    assert result.image_prompt == prompt
    assert client.call_count == 1


def test_preparer_does_not_require_generic_same_style_process_language() -> None:
    prompt = "A boxy CRT monitor displays dense static in stark monochrome dithering."
    preparer, client = _preparer(_output(prompt))

    result = preparer.prepare(
        ImagePromptPreparationRequest(
            description="The monitor displays dense static in the same style."
        )
    )

    assert result.image_prompt == prompt
    assert client.call_count == 1


def test_preparer_repairs_explicit_object_state_conflict() -> None:
    preparer, client = _preparer(
        [
            _output("A firmly closed circular station hatch."),
            json.dumps(
                {"image_prompt": "An open circular station hatch reveals a corridor."}
            ),
        ]
    )

    result = preparer.prepare(
        ImagePromptPreparationRequest(
            description="The station hatch is open."
        )
    )

    assert "open circular station hatch" in result.image_prompt
    assert client.call_count == 2


def test_preparer_rejects_persistent_authored_state_conflict() -> None:
    preparer, client = _preparer(
        [
            _output("A firmly closed circular station hatch."),
            json.dumps(
                {"image_prompt": "A firmly closed circular station hatch."}
            ),
        ]
    )

    with pytest.raises(ModelResponseError, match="invalid Image Prompt"):
        preparer.prepare(
            ImagePromptPreparationRequest(
                description="The station hatch is open."
            )
        )

    assert client.call_count == 2


def test_preparer_requires_exact_authored_visible_text() -> None:
    preparer, _client = _preparer(
        [
            _output('A CRT screen reads "FAIL".'),
            json.dumps({"image_prompt": 'A CRT screen reads "FAIL".'}),
        ]
    )

    with pytest.raises(ModelResponseError, match="visible text"):
        preparer.prepare(
            ImagePromptPreparationRequest(
                description='A CRT screen reads "ERROR".'
            )
        )


def test_preparer_treats_negated_quoted_text_as_forbidden() -> None:
    prompt = (
        "A vintage computer screen displays a prominent emergency exit "
        "pictogram with a running human figure and arrow, showing only "
        "symbols with no text."
    )
    preparer, client = _preparer(_output(prompt))

    result = preparer.prepare(
        ImagePromptPreparationRequest(
            description=(
                'The computer shows an emergency exit pictogram. No "emergency '
                'exit" or "exit" text is displayed - only the symbols.'
            )
        )
    )

    assert result.image_prompt == prompt
    assert client.call_count == 1


def test_preparer_repairs_forbidden_quoted_text() -> None:
    repaired = "A computer screen shows only a running figure and arrow pictogram."
    preparer, client = _preparer(
        [
            _output('A computer screen reads "EXIT".'),
            json.dumps({"image_prompt": repaired}),
        ]
    )

    result = preparer.prepare(
        ImagePromptPreparationRequest(
            description=(
                'A computer screen that must omit the word "EXIT", showing '
                "only symbols."
            )
        )
    )

    assert result.image_prompt == repaired
    assert client.call_count == 2


@pytest.mark.parametrize(
    "description",
    (
        'A sign without the word "EXIT".',
        'A sign that must avoid using the word "EXIT".',
        'A sign that avoided the word "EXIT".',
    ),
)
def test_preparer_accepts_common_quoted_text_exclusions(
    description: str,
) -> None:
    prompt = "A blank sign with no lettering."
    preparer, client = _preparer(_output(prompt))

    result = preparer.prepare(
        ImagePromptPreparationRequest(description=description)
    )

    assert result.image_prompt == prompt
    assert client.call_count == 1


def test_preparer_allows_same_text_required_and_forbidden_on_different_objects() -> None:
    prompt = 'The main sign reads "EXIT" while the surrounding wall remains blank.'
    preparer, client = _preparer(_output(prompt))

    result = preparer.prepare(
        ImagePromptPreparationRequest(
            description=(
                'The main sign reads "EXIT", but no "EXIT" text appears on '
                "the surrounding wall."
            )
        )
    )

    assert result.image_prompt == prompt
    assert client.call_count == 1


def test_request_reference_state_must_match_attached_image(tmp_path: Path) -> None:
    preparer, _client = _preparer(_output("A courtyard"))

    with pytest.raises(ValueError, match="must match"):
        preparer.prepare(
            ImagePromptPreparationRequest(description="A courtyard"),
            reference_image_path=tmp_path / "unexpected.png",
        )
