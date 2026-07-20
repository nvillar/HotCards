# HyperGen

HyperGen is an experimental, local-first authoring tool for illustrated,
spatially interactive stacks of cards, inspired by classic HyperCard.

Authors describe each card in natural language, generate a background image
locally, generate and edit clickable polygon hotspots, and run the stack as an
interactive experience. All generation runs on-device.

**Status:** early developer proof of concept. Not packaged for end users.

**Platform:** Apple Silicon macOS. Requires a running local
[Ollama](https://ollama.com) daemon and local MFLUX image models.

## Setup and run

Prerequisites: Python 3.12, [uv](https://docs.astral.sh/uv/), a running Ollama
daemon with the configured model, and MFLUX models available locally.

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
- `generation/` — MFLUX image and Ollama hotspot adapters and prompt builders.
- `application/` — document controller, typed commands, session undo, workers.
- `ui/` — PySide6 Author and Run interface.
- `evaluation/` — `hypergen-eval` harness reusing production adapters.

## Roadmap

Planned work, acceptance criteria, and progress live in GitHub Issues and the
[**HyperGen POC** milestone](https://github.com/nvillar/HyperGen/milestone/1).
