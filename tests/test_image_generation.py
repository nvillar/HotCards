"""Tests for direct Description prompt composition."""

from uuid import uuid4

import pytest

from hotcards.domain.models import (
    AcceptedEdit,
    EditPreserveOptions,
    GenerateInputs,
    ImageReferenceSnapshot,
    StyleSnapshot,
)
from hotcards.generation.image_generation import (
    compose_edit_prompt,
    compose_generation_prompt,
    compose_refine_prompt,
)


def test_description_is_delivered_without_hidden_rewriting() -> None:
    description = "A stone courtyard at dusk in detailed ink and watercolor."

    inputs = GenerateInputs(description=description)

    assert compose_generation_prompt(inputs) == description


def test_reference_does_not_add_role_instructions() -> None:
    snapshot = ImageReferenceSnapshot(
        card_id=uuid4(),
        revision_id=uuid4(),
        background_id=uuid4(),
    )
    description = "A red fox in crisp monochrome halftone linework using image 1."

    result = compose_generation_prompt(
        GenerateInputs(
            description=description,
            references=(snapshot,),
        )
    )

    assert result == description
    assert "Reference" not in result


def test_selected_style_is_appended_after_the_description() -> None:
    description = "A red fox beneath a tree"
    style = StyleSnapshot(
        style_id=uuid4(),
        name="Ink",
        prompt_text="Rendered with bold black ink contours.",
    )

    result = compose_generation_prompt(
        GenerateInputs(
            description=description,
            style=style,
        )
    )

    assert result == ("A red fox beneath a tree.\n\nRendered with bold black ink contours.")


def test_blank_selected_style_does_not_change_delivery() -> None:
    description = "A red fox beneath a tree."

    result = compose_generation_prompt(
        GenerateInputs(
            description=description,
            style=StyleSnapshot(
                style_id=uuid4(),
                name="Draft Style",
                prompt_text="",
            ),
        )
    )

    assert result == description


@pytest.mark.parametrize("value", ("", "   "))
def test_description_must_be_non_empty_for_composition(value: str) -> None:
    with pytest.raises(ValueError, match="Description"):
        compose_generation_prompt(
            GenerateInputs(
                description=value,
            )
        )


def test_refine_prompt_without_edit_lineage_is_description_then_style() -> None:
    style = StyleSnapshot(
        style_id=uuid4(),
        name="Ink",
        prompt_text="Rendered with bold black ink contours.",
    )

    assert compose_refine_prompt("A red fox beneath a tree", style, ()) == (
        "A red fox beneath a tree.\n\nRendered with bold black ink contours."
    )


def test_refine_prompt_preserves_ordered_authored_edit_lineage() -> None:
    edits = (
        AcceptedEdit(
            instruction="Open the gate.",
            preserve=EditPreserveOptions(subject_identity=True),
            expanded_prompt="Hidden expansion one.",
        ),
        AcceptedEdit(
            instruction="Add ivy to the wall.",
            preserve=EditPreserveOptions(background=True),
            expanded_prompt="Hidden expansion two.",
        ),
    )

    prompt = compose_refine_prompt(
        "The gate is now closed and painted red.",
        None,
        edits,
    )

    assert prompt == (
        "The gate is now closed and painted red.\n\n"
        "The current Description is authoritative. The source image already "
        "includes these accepted edits; preserve them unless they conflict "
        "with the current Description:\n"
        "1. Open the gate.\n"
        "2. Add ivy to the wall."
    )
    assert "Hidden expansion" not in prompt
    assert prompt.index("current Description is authoritative") < prompt.index("Open the gate")


@pytest.mark.parametrize("value", ("", "   "))
def test_refine_prompt_requires_current_description(value: str) -> None:
    with pytest.raises(ValueError, match="Description"):
        compose_refine_prompt(value, None, ())


def test_edit_prompt_trims_instruction_and_preserves_every_unrequested_detail() -> None:
    assert compose_edit_prompt("  Replace the closed door with an open arch.  ") == (
        "Edit the provided image according to this instruction:\n\n"
        "Replace the closed door with an open arch.\n\n"
        "The source image is authoritative for everything not explicitly changed "
        "by the instruction. Make only the requested change and the minimum "
        "accompanying changes necessary for visual coherence. Preserve all other "
        "subjects, identities, objects, text, composition, framing, background, "
        "lighting, colors, and visual style. Do not add, remove, or reinterpret "
        "unrelated details."
    )


def test_edit_prompt_injects_selected_style_after_the_authored_instruction() -> None:
    style = StyleSnapshot(
        style_id=uuid4(),
        name="Ink",
        prompt_text="Rendered with bold black ink contours.",
    )

    result = compose_edit_prompt("Open the garden gate.", style)

    assert result.endswith(
        "Unless the Edit Instruction explicitly changes the visual treatment, "
        "keep the result consistent with this selected Style:\n\n"
        "Rendered with bold black ink contours."
    )
    assert result.index("Open the garden gate.") < result.index(style.prompt_text)


def test_blank_selected_style_does_not_change_edit_prompt() -> None:
    instruction = "Open the garden gate."
    style = StyleSnapshot(
        style_id=uuid4(),
        name="Draft Style",
        prompt_text="",
    )

    assert compose_edit_prompt(instruction, style) == compose_edit_prompt(instruction)


@pytest.mark.parametrize("value", ("", "   "))
def test_edit_prompt_requires_instruction(value: str) -> None:
    with pytest.raises(ValueError, match="Edit Instruction"):
        compose_edit_prompt(value)
