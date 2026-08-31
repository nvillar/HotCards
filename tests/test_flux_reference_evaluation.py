"""Tests for the local FLUX multi-reference feasibility suite."""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from PIL import Image

from hotcards.domain.models import (
    Card,
    CardRevision,
    DirectGenerateProvenance,
    GeneratedBackground,
    GenerateInputs,
    ImageOperationSettings,
    Stack,
)
from hotcards.evaluation.flux_references import (
    _model_config,
    run_flux_reference_evaluation,
)
from hotcards.storage.stack_store import StackStore


class FakeGeneratedImage:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def save(self, path: Path, *, overwrite: bool) -> None:
        assert not overwrite
        Image.new("RGB", (self.width, self.height), "navy").save(path, format="PNG")


class FakeReferenceModel:
    def __init__(
        self,
        requests: list[dict[str, object]],
        *,
        fail_prompt_fragment: str | None = None,
    ) -> None:
        self.requests = requests
        self.fail_prompt_fragment = fail_prompt_fragment

    def generate_image(self, **kwargs: object) -> FakeGeneratedImage:
        self.requests.append(kwargs)
        prompt = str(kwargs["prompt"])
        if self.fail_prompt_fragment and self.fail_prompt_fragment in prompt:
            raise RuntimeError("candidate failed")
        return FakeGeneratedImage(
            width=kwargs["width"],  # type: ignore[arg-type]
            height=kwargs["height"],  # type: ignore[arg-type]
        )


def _write_png(path: Path, color: str) -> None:
    Image.new("RGB", (32, 24), color).save(path, format="PNG")


def _card_with_image(
    *,
    store: StackStore,
    source: Path,
    name: str,
    description: str,
) -> Card:
    card_id = uuid4()
    background_id = uuid4()
    image_path = store.store_image_asset(
        source,
        card_id=card_id,
        asset_id=background_id,
    )
    generated_at = datetime.now(UTC)
    revision = CardRevision(
        description=description,
        background=GeneratedBackground(
            id=background_id,
            image_path=image_path,
            provenance=DirectGenerateProvenance(
                inputs=GenerateInputs(
                    description=description,
                ),
                render_prompt=description,
                settings=ImageOperationSettings(
                    model_identifier="test",
                    mflux_version="test",
                    seed=1,
                    width=1024,
                    height=768,
                    step_count=4,
                    generated_at=generated_at,
                    duration_seconds=1,
                ),
            ),
            created_at=generated_at,
        ),
    )
    return Card(
        id=card_id,
        name=name,
        revisions=(revision,),
        active_revision_id=revision.id,
    )


def _write_reference_stack(tmp_path: Path) -> Path:
    bundle = tmp_path / "References.hotcards"
    store = StackStore(bundle)
    map_source = tmp_path / "map.png"
    castle_source = tmp_path / "castle.png"
    _write_png(map_source, "lightblue")
    _write_png(castle_source, "gray")
    map_card = _card_with_image(
        store=store,
        source=map_source,
        name="Map",
        description="A hand-drawn fantasy map.",
    )
    castle_card = _card_with_image(
        store=store,
        source=castle_source,
        name="Castle",
        description="A pencil drawing of a castle.",
    )
    store.save(Stack(name="References", cards=(map_card, castle_card)))
    return bundle


def test_reference_model_config_selects_every_supported_variant() -> None:
    assert _model_config("flux2-klein-4b", edit=True).model_name.endswith("klein-4B")
    assert _model_config("flux2-klein-9b", edit=True).model_name.endswith("klein-9B")
    assert _model_config("flux2-klein-9b-kv", edit=True).model_name.endswith("klein-9b-kv")
    assert _model_config("flux2-klein-9b-kv", edit=False).model_name.endswith("klein-9B")


def test_reference_suite_preserves_order_prompts_and_provenance(
    tmp_path: Path,
) -> None:
    stack_path = _write_reference_stack(tmp_path)
    requests: list[dict[str, object]] = []
    output_dir = tmp_path / "run"

    result_path = run_flux_reference_evaluation(
        output_dir=output_dir,
        stack_path=stack_path,
        width=48,
        height=32,
        model_factory=lambda *_: FakeReferenceModel(requests),
        source_model_factory=lambda *_: FakeReferenceModel(requests),
        environment_provider=lambda: {"git_sha": "test"},
    )

    result = json.loads(result_path.read_text())
    assert result["status"] == "success"
    assert len(requests) == 8
    assert "image_paths" not in requests[0]
    combined = next(case for case in result["cases"] if case["case_id"] == "character-plus-style")
    assert combined["reference_keys"] == ["character_identity", "map_style"]
    assert [Path(path).name for path in combined["reference_paths"]] == [
        "character_identity.png",
        "map_style.png",
    ]
    assert combined["prompt"] == combined["scene"]
    assert combined["prompt"].index("image 1") < combined["prompt"].index("image 2")
    assert "REFERENCE IMAGE" not in combined["prompt"]
    assert "SCENE\n" not in combined["prompt"]
    assert [
        Path(path).name
        for path in requests[3]["image_paths"]  # type: ignore[arg-type]
    ] == ["character_identity.png", "map_style.png"]
    assert result["sources"]["map_style"]["card_id"]
    assert result["sources"]["map_style"]["revision_id"]
    assert result["sources"]["map_style"]["description"] == ("A hand-drawn fantasy map.")
    assert (output_dir / "inputs" / "map_style.png").is_file()
    assert len(list((output_dir / "outputs").glob("*.png"))) == 7
    assert (output_dir / "contact-sheet.png").is_file()
    assert len(list((output_dir / "comparisons").glob("*.png"))) == 7
    assert result["model_load"]["source"]["duration_seconds"] >= 0
    assert result["model_load"]["edit"]["duration_seconds"] >= 0
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "success"
    assert any(artifact["path"] == "inputs/map_style.png" for artifact in manifest["artifacts"])


def test_reference_suite_isolates_failure_and_failed_dependency(
    tmp_path: Path,
) -> None:
    stack_path = _write_reference_stack(tmp_path)
    requests: list[dict[str, object]] = []
    output_dir = tmp_path / "run"

    result_path = run_flux_reference_evaluation(
        output_dir=output_dir,
        stack_path=stack_path,
        width=48,
        height=32,
        model_factory=lambda *_: FakeReferenceModel(
            requests,
            fail_prompt_fragment="rear garden at ground level",
        ),
        source_model_factory=lambda *_: FakeReferenceModel(requests),
        environment_provider=lambda: {"git_sha": "test"},
    )

    result = json.loads(result_path.read_text())
    assert result["status"] == "completed_with_failures"
    failed = {
        case["case_id"]: case["generation"]["failure"]["message"]
        for case in result["cases"]
        if case["generation"]["status"] == "failed"
    }
    assert failed["building-new-view"] == "candidate failed"
    assert "building-new-view" in failed["building-two-views"]
    failed_dependency = next(
        case for case in result["cases"] if case["case_id"] == "building-two-views"
    )
    assert failed_dependency["reference_paths"] == []
    assert len(requests) == 7
    assert (output_dir / "outputs" / "reference-count-2.png").is_file()
    assert len(list((output_dir / "outputs").glob("*.png"))) == 5
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "completed_with_failures"
