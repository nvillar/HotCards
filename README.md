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
Bundles must use the current schema version; legacy stack schemas are
intentionally unsupported and are not migrated on load.

Each card owns one or more numbered revisions. A revision contains its authored
Description, optional Enriched Description, optional generated background,
fixed Subject, Style, and Setting card references, and hotspot set. The compact
header above the canvas edits the card name and selects, duplicates, or deletes
revisions; the toolbar manages Mode and hotspot visibility.

The Background inspector follows the authoring sequence Description,
References, Enrich Description, then Image. Description and Enriched Description
share one editor with native Original and Enriched radio controls that appear
only after enrichment; the enriched text is shown by default. The editor is
sized for the recommended prompt length. Reference labels and selectors share
compact rows inside a References group, and button tooltips identify the inputs
to Enrich Description and Generate Image.
Each reference role can select one other card; the same card cannot reference
itself. Generate groups roles that use the same active source background, sends
each unique image once in Subject, Style, Setting order, and combines its role
instructions. Background prompts place Enriched Description when present,
otherwise Description, before concise natural-language instructions that assign
each numbered image its roles. Internal headings and source-card Descriptions
are not injected into the MFLUX prompt. Generated images are the only supported
background source.
Generate, image removal, and Enrich Description apply directly to the active
revision. Completed generations can be kept there with Keep, moved into a new
complete revision, or undone.
Existing images remain in place until replacement succeeds, and successful
changes offer a dismissible, history-safe Undo action in the notification bar.
Enrichment never reads an image and requires authored Description text. It
always starts from that authored text, receives no reference-card Description,
and stores the result separately. Subject, Style, and Setting are interpreted
from their actual images by FLUX.2 during image generation, preventing a
reference card's subjects, arrangement, or narrative from leaking through text
enrichment. Reference changes therefore do not stale Enriched Description. An
enrichment is Current while its authored Description, selected Ollama model,
and prompt contract still match its inputs; otherwise it remains active but is
marked Out of date. Enriched text may add visual detail but never overrides
authored actions, poses, object states, time, weather, viewpoint, crop, framing,
composition, or quoted visible text. Recognized object-state reversals receive
one constrained repair attempt and are rejected if unresolved. Generate uses
Enriched Description when present, including when it is out of date, and
otherwise falls back to Description.
The Enrich action is disabled and shown as complete while enrichment is current,
then becomes Re-enrich when its authored Description, selected Ollama model, or
prompt contract is out of date. Clearing all Enriched text removes it. Image
generation requires at least one non-empty Description source and uses only the
effective source as scene content.
Enriched text follows FLUX.2 prompt guidance: one concise natural-language
paragraph ordered by subject, action, style, context, then secondary details,
using positive descriptions and object-specific colors and materials.
A rewrite that invents quoted visible text or reverses a recognized authored
object state is rejected without changing either Description field.

The Hotspots inspector is the sole source of interaction semantics. A new
hotspot is persisted and selected immediately, even before it has an area;
its displayed name is always its destination card's current name, or
`Unresolved` when it has no resolved destination. Hotspot names are not edited
separately. While the Hotspots tab is active, clicking empty canvas begins a
polygon for the selected hotspot and creates one first when needed. Leaving the
tab cancels any unfinished polygon and hides its authoring overlays. Canvas
editing uses hierarchical hotspot, area, and vertex selection: drag an area or
vertex to move it, use the edge `+` or double-click an edge to add a vertex,
press Delete to remove the selected vertex or area, and press Escape to step
back through the selection. Context menus expose the same geometry actions, and
successful edits offer a dismissible Undo. Replacing
a background preserves its hotspots so the author can review and adjust them
manually. Run mode supports deterministic hotspot navigation, Back/Restart
history, and configurable overlays. It opens on the current Author card and,
when that differs from the configured start card, offers a dismissible restart
action. Run-only navigation and overlay controls stay hidden in Author mode.
Local AI services are checked automatically when Author mode is entered; Run
mode starts no AI work. The status bar selects
an installed Ollama LLM and either FLUX.2 Klein 4B or FLUX.2 Klein 9B KV.

Transient outcomes, failures, Run warnings, and Undo actions appear in one
notification bar below the toolbar. Reversible deletions and replacements apply
directly and offer Undo. Completed Description and image generations also offer
Create New Version, which restores the prior current revision and activates a
complete new revision containing the result, plus an explicit Keep action that
retains it on the current revision. Field validation remains beside the
responsible input.
AI failures and recovery actions stay in the notification bar rather than the
model selectors.

## Setup and run

Prerequisites: Python 3.12 and [uv](https://docs.astral.sh/uv/). To enable
generation, also run Ollama with an installed text model (default:
`qwen3.5:9b-mlx`) and cache the selected MFLUX model locally. FLUX.2 Klein 4B
uses Apache 2.0; FLUX.2 Klein 9B KV uses the FLUX Non-Commercial License. The
9B KV choice requires both the regular 9B weights for text-only generation and
the 9B KV weights for reference generation.

```sh
uv sync
uv run hypergen
```

## Evaluation harness

```sh
uv run hypergen-eval smoke
uv run hypergen-eval images
uv run hypergen-eval flux-references --stack /path/to/Stack.hypergen
```

Each command creates one immutable directory under `evals/runs/` with an
immediate, failure-safe `manifest.json` and retained artifacts. Smoke and image
runs add checksums, JSON/CSV summaries, and a static HTML report. Reference runs
write detailed JSON results plus overall and per-case contact sheets. Report
rendering is offline and never calls a model. The tracked default decision and
evidence tradeoffs are in [`evals/DECISION.md`](evals/DECISION.md).

## Architecture at a glance

- `domain/` — in-memory stack model, geometry, validation.
- `storage/` — human-readable `*.hypergen` bundle storage.
- `generation/` — deterministic MFLUX prompt composition, image generation,
  and Description enrichment.
- `application/` — document controller, typed commands, session undo, workers.
- `ui/` — PySide6 Author and Run interface.
- `evaluation/` — `hypergen-eval` harness reusing production adapters.

## Roadmap

Planned work, acceptance criteria, and progress live in GitHub Issues and the
[**HyperGen POC** milestone](https://github.com/nvillar/HyperGen/milestone/1).
