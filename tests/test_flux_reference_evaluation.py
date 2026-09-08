"""Tests for the local FLUX multi-reference feasibility suite."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

import pytest
from PIL import Image
from test_generation_smoke import FakeMfluxModel

import hotcards.evaluation.cli as cli_module
import hotcards.evaluation.flux_references as reference_module
from hotcards.domain.image_dimensions import AspectRatio, ResolutionTier
from hotcards.domain.models import (
    Card,
    CardRevision,
    GeneratedBackground,
    GenerateInputs,
    GenerateOperation,
    ImageOperationSettings,
    ImageOriginFacts,
    ImageProvenance,
    PresetOutputSize,
    Stack,
)
from hotcards.evaluation.cli import build_parser
from hotcards.evaluation.flux_references import (
    run_flux_reference_evaluation,
)
from hotcards.storage.stack_store import StackStore


@pytest.fixture(autouse=True)
def fixed_memory_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reference_module, "_resident_bytes", lambda: 4096)
    monkeypatch.setattr(reference_module, "_peak_resident_bytes", lambda: 8192)


def _write_png(path: Path, color: str) -> None:
    Image.new("RGB", (256, 192), color).save(path, format="PNG")


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
    generated_at = datetime(2026, 7, 19, tzinfo=UTC)
    revision = CardRevision(
        description=description,
        background=GeneratedBackground(
            id=background_id,
            image_path=image_path,
            provenance=ImageProvenance(
                authoring=GenerateOperation(
                    inputs=GenerateInputs(
                        description=description,
                        output_size=PresetOutputSize(tier=ResolutionTier.SMALL),
                    ),
                ),
                origin=ImageOriginFacts(
                    render_prompt=description,
                    settings=ImageOperationSettings(
                        model_identifier="test",
                        mflux_version="test",
                        seed=1,
                        width=256,
                        height=192,
                        step_count=4,
                        generated_at=generated_at,
                        duration_seconds=1,
                    ),
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


def test_reference_cli_rejects_obsolete_arbitrary_dimensions() -> None:
    defaults = build_parser().parse_args(["flux-references", "--stack", "Stack.hotcards"])
    assert defaults.tier is ResolutionTier.FULL
    assert defaults.aspect_ratio is AspectRatio.LANDSCAPE
    with pytest.raises(SystemExit) as caught:
        build_parser().parse_args(["flux-references", "--stack", "Stack.hotcards", "--width", "48"])

    assert caught.value.code == 2
    with pytest.raises(SystemExit) as caught:
        build_parser().parse_args(["flux-references", "--stack", "Stack.hotcards", "--kv-cache"])

    assert caught.value.code == 2


def test_reference_suite_preserves_order_prompts_and_provenance(
    tmp_path: Path,
) -> None:
    stack_path = _write_reference_stack(tmp_path)
    requests: list[dict[str, object]] = []
    output_dir = tmp_path / "run"

    result_path = run_flux_reference_evaluation(
        output_dir=output_dir,
        stack_path=stack_path,
        tier=ResolutionTier.SMALL,
        aspect_ratio=AspectRatio.LANDSCAPE,
        model_factory=lambda *_: FakeMfluxModel(requests),
        source_model_factory=lambda *_: FakeMfluxModel(requests),
        environment_provider=lambda: {"git_sha": "test"},
    )

    result = json.loads(result_path.read_text())
    assert result["status"] == "success"
    assert len(requests) == 8
    assert {(request["width"], request["height"]) for request in requests} == {(256, 192)}
    assert result["settings"]["tier"] == "Small"
    assert result["settings"]["long_edge"] == 256
    assert result["settings"]["aspect_ratio"] == "4:3"
    assert len([request for request in requests if "image_paths" not in request]) == 1
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
    expected_snapshots = {}
    store = StackStore(stack_path)
    for card in store.load().cards:
        key = "map_style" if card.name == "Map" else "castle_identity"
        revision = card.active_revision
        background = revision.background
        assert background is not None
        snapshot = {
            "card_id": str(card.id),
            "revision_id": str(revision.id),
            "background_id": str(background.id),
        }
        expected_snapshots[key] = snapshot
        assert {field: result["sources"][key][field] for field in snapshot} == snapshot
        assert result["sources"][key]["description"] == revision.description
        source = store.asset_path(background.image_path)
        copied = output_dir / result["sources"][key]["path"]
        assert (
            hashlib.sha256(copied.read_bytes()).hexdigest()
            == hashlib.sha256(source.read_bytes()).hexdigest()
        )
        with Image.open(source) as image:
            assert (
                image.size
                == (
                    background.provenance.settings.width,
                    background.provenance.settings.height,
                )
                == (256, 192)
            )
    for key in ("character_identity", "building_new_view"):
        expected_snapshots[key] = {
            field: str(uuid5(NAMESPACE_URL, f"hotcards:flux-reference:{key}:{field}"))
            for field in ("card_id", "revision_id", "background_id")
        }
    assert [case["case_id"] for case in result["cases"]] == [
        "character-identity",
        "style-only",
        "character-plus-style",
        "building-new-view",
        "building-two-views",
        "reference-count-1",
        "reference-count-2",
    ]
    for case in result["cases"]:
        matches = [request for request in requests if request["prompt"] == case["scene"]]
        assert len(matches) == 1
        assert matches[0]["image_paths"] == [output_dir / path for path in case["reference_paths"]]
        generation = case["generation"]
        assert generation["metadata"]["origin"]["render_prompt"] == case["scene"]
        assert generation["metadata"]["authoring"]["inputs"]["references"] == [
            expected_snapshots[key] for key in case["reference_keys"]
        ]
        assert generation["resident_bytes_before"] == generation["resident_bytes_after"] == 4096
        assert generation["resident_bytes_delta"] == generation["peak_resident_bytes_delta"] == 0
    assert len(list((output_dir / "outputs").glob("*.png"))) == 7
    assert (output_dir / "contact-sheet.png").is_file()
    assert len(list((output_dir / "comparisons").glob("*.png"))) == 7
    assert (
        result["model_load"]["source"]["duration_seconds"]
        == (result["sources"]["character_identity"]["generation"]["load_duration_seconds"])
    )
    assert (
        result["model_load"]["edit"]["duration_seconds"]
        == (result["cases"][0]["generation"]["load_duration_seconds"])
    )
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "success"
    map_artifact = next(
        artifact for artifact in manifest["artifacts"] if artifact["path"] == "inputs/map_style.png"
    )
    assert map_artifact["sha256"] == hashlib.sha256((tmp_path / "map.png").read_bytes()).hexdigest()


def test_reference_suite_isolates_failure_and_failed_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = reference_module._case_specs()
    monkeypatch.setattr(
        reference_module,
        "_case_specs",
        lambda: tuple(
            case
            for case in cases
            if case["case_id"] in ("building-new-view", "building-two-views", "reference-count-2")
        ),
    )
    stack_path = _write_reference_stack(tmp_path)
    requests: list[dict[str, object]] = []
    output_dir = tmp_path / "run"

    result_path = run_flux_reference_evaluation(
        output_dir=output_dir,
        stack_path=stack_path,
        tier=ResolutionTier.SMALL,
        aspect_ratio=AspectRatio.LANDSCAPE,
        model_factory=lambda *_: FakeMfluxModel(
            requests,
            fail_prompt_contains="rear garden at ground level",
        ),
        source_model_factory=lambda *_: FakeMfluxModel(requests),
        environment_provider=lambda: {"git_sha": "test"},
    )

    result = json.loads(result_path.read_text())
    assert result["status"] == "completed_with_failures"
    failed = {
        case["case_id"]: case["generation"]["failure"]["message"]
        for case in result["cases"]
        if case["generation"]["status"] == "failed"
    }
    assert failed["building-new-view"].endswith("candidate failed")
    assert "building-new-view" in failed["building-two-views"]
    failed_dependency = next(
        case for case in result["cases"] if case["case_id"] == "building-two-views"
    )
    assert failed_dependency["reference_paths"] == []
    assert len(requests) == 3
    assert (output_dir / "outputs" / "reference-count-2.png").is_file()
    assert len(list((output_dir / "outputs").glob("*.png"))) == 1
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "completed_with_failures"


def test_reference_suite_fails_when_no_reference_case_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cases = reference_module._case_specs()[:1]
    monkeypatch.setattr(reference_module, "_case_specs", lambda: cases)
    output_dir = tmp_path / "run"
    monkeypatch.setattr(
        cli_module,
        "run_flux_reference_evaluation",
        lambda **kwargs: run_flux_reference_evaluation(
            **kwargs,
            model_factory=lambda *_: FakeMfluxModel([], fail_prompt_contains=""),
            source_model_factory=lambda *_: FakeMfluxModel([]),
            environment_provider=lambda: {"git_sha": "test"},
        ),
    )
    assert (
        cli_module.run_cli(
            [
                "flux-references",
                "--output-dir",
                str(output_dir),
                "--stack",
                str(_write_reference_stack(tmp_path)),
                "--tier",
                "Small",
            ]
        )
        == 1
    )
    assert "produced no images" in capsys.readouterr().err
    result = json.loads((output_dir / "flux-reference-results.json").read_text())
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert result["status"] == manifest["status"] == "failed"
    assert manifest["failure"]["stages"][0]["case_id"] == "character-identity"
    assert result["sources"]["character_identity"]["generation"]["status"] == "success"
    assert not (output_dir / "contact-sheet.png").exists()
