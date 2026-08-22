# HyperGen

HyperGen is an experimental, local-first authoring tool for illustrated,
spatially interactive stacks of cards, inspired by classic HyperCard.

Authors describe each card in natural language, generate a background image
locally, generate and edit clickable polygon hotspots, and run the stack as an
interactive experience. All generation runs on-device.

**Status:** early developer proof of concept. Not packaged for end users.

**Platform:** Apple Silicon macOS. Local generation requires a running
[Ollama](https://ollama.com) daemon and locally cached MFLUX image models; the
authoring shell remains usable when either service is unavailable.

Stacks are stored as self-contained `.hypergen` directory bundles and
autosaved atomically after creation or opening. At startup, HyperGen lists
projects in `~/Documents/HyperGen` and offers direct Open and Create actions.

The current Author workflow uses Background and Hotspots inspector tabs, with
the selected card's name editable above the canvas. Background keeps
Description, accepted image revisions, and a collapsible Style editor together.
Background prompts are composed directly from the author-visible Description
plus either the stack's global Style or a card-specific replacement; interaction
intent does not alter the image. Enrich produces one editable Description
proposal for explicit acceptance or discard. When an accepted background is
active, Enrich first describes its visible details and merges them with the
authored Description and effective Style; otherwise it performs text-only
enrichment. Accepted text remains one ordinary undoable Description edit.
Authors can review generated or imported card-local background drafts, with
generated revisions receiving short model-generated names. They can switch
cards without losing drafts, manage revision history, and edit manual
multi-area polygon hotspots with explicit card destinations. Authors can
also generate transient hotspot candidates from the active image and interaction
intent, correct them with the same geometry and destination controls, review
absent-subject warnings, then apply or discard the complete set. Summarize
Hotspots can reconstruct Intent offline from the active revision's applied
labels and destinations. Run mode supports deterministic hotspot navigation,
Back/Restart history, and configurable overlays. Local AI services are checked
automatically when Author mode is entered; Run mode starts no AI work.

## Setup and run

Prerequisites: Python 3.12 and [uv](https://docs.astral.sh/uv/). To enable
generation, also run Ollama with the configured model (default:
`qwen3.5:9b-mlx`) and cache the selected MFLUX model locally.

```sh
uv sync
uv run hypergen
```

## Evaluation harness

```sh
uv run hypergen-eval smoke
uv run hypergen-eval hotspots
uv run hypergen-eval images
uv run hypergen-eval e2e
```

Each command creates one immutable directory under `evals/runs/` with an
immediate, failure-safe `manifest.json`, retained raw/completed artifacts,
checksums, JSON/CSV summaries, and a static HTML report. Image and hotspot
reports include contact sheets or annotated predictions. Report rendering is
offline and never calls a model. The tracked default decision and evidence
tradeoffs are in [`evals/DECISION.md`](evals/DECISION.md).

## Architecture at a glance

- `domain/` — in-memory stack model, geometry, validation.
- `storage/` — human-readable `*.hypergen` bundle storage.
- `generation/` — deterministic MFLUX prompt composition, image generation,
  Description enrichment, and Ollama hotspot adapters.
- `application/` — document controller, typed commands, session undo, workers.
- `ui/` — PySide6 Author and Run interface.
- `evaluation/` — `hypergen-eval` harness reusing production adapters.

## Roadmap

Planned work, acceptance criteria, and progress live in GitHub Issues and the
[**HyperGen POC** milestone](https://github.com/nvillar/HyperGen/milestone/1).
