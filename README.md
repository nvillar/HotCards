# HyperGen

HyperGen is an experimental, local-first authoring tool for illustrated,
spatially interactive stacks of cards, inspired by classic HyperCard.

Authors describe each card in natural language, generate a background image
locally, place clickable polygon hotspots, and run the stack as an interactive
experience. All generation runs on-device.

**Status:** early developer proof of concept. Not packaged for end users.

**Platform:** Apple Silicon macOS. Local generation requires a running
[Ollama](https://ollama.com) daemon and locally cached MFLUX image models; the
authoring shell remains usable when either service is unavailable.

Stacks are stored as self-contained `.hypergen` directory bundles and
autosaved atomically after creation or opening. At startup, HyperGen lists
projects in `~/Documents/HyperGen` and offers direct Open and Create actions.

Each card owns one or more numbered revisions. A revision contains its
Description, selected stack Style, optional blank/generated/imported background,
and hotspot set. The compact header above the canvas edits the card name and
selects, duplicates, or deletes revisions; the toolbar manages Mode, hotspot
visibility, and the stack-wide Style library.

The Background inspector contains Description, Enrich, Generate Image, Style,
Import Image, and Clear Image controls. Background prompts are composed
deterministically from the revision Description and selected Style. Generate,
Import, Clear, and Enrich apply directly to the active revision; replacement
actions confirm before discarding an existing image, and successful changes
offer a dismissible, history-safe Undo action. With a readable image, Enrich
first describes visible details and incorporates them into the revised
Description; otherwise it performs text-only enrichment.

The Hotspots inspector is the sole source of interaction semantics. A new
hotspot is persisted and selected immediately, even before it has an area;
clicking empty canvas begins a polygon for the selected hotspot and creates one
first when needed. Remap asks the local vision model only to relocate existing
hotspots on the active image. It cannot add, delete, rename, reorder, or
retarget them, and unmatched hotspots retain their previous geometry. Run mode
supports deterministic hotspot navigation, Back/Restart history, and
configurable overlays. Local AI services are checked automatically when Author
mode is entered; Run mode starts no AI work.

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

Coordinate-grounding experiments can compare labeled grids and Ollama models
against the authored polygons in any local stack revision:

```sh
uv run hypergen-eval remap-grid \
  --stack "/path/to/Stack.hypergen" --card "Map" --revision 5
```

## Architecture at a glance

- `domain/` — in-memory stack model, geometry, validation.
- `storage/` — human-readable `*.hypergen` bundle storage.
- `generation/` — deterministic MFLUX prompt composition, image generation,
  Description enrichment, and Ollama hotspot Remap adapters.
- `application/` — document controller, typed commands, session undo, workers.
- `ui/` — PySide6 Author and Run interface.
- `evaluation/` — `hypergen-eval` harness reusing production adapters.

## Roadmap

Planned work, acceptance criteria, and progress live in GitHub Issues and the
[**HyperGen POC** milestone](https://github.com/nvillar/HyperGen/milestone/1).
