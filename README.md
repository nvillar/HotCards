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
Schema-v5 bundles are migrated in memory to the current schema when opened;
unsupported older or future schemas are rejected.

Each card owns one or more numbered revisions. A revision contains its authored
Description, optional prepared Image Prompt, optional generated background,
optional Reference card, and hotspot set. The compact header above the canvas
edits the card name and selects, duplicates, or deletes revisions; the toolbar
provides a single Author/Run mode toggle and manages hotspot visibility. An
adjacent step label and progress bar to the left of the model selectors show
actual MFLUX inference-step completion during image generation and an
indeterminate state during Image Prompt preparation.

The Background inspector follows the authoring sequence Description, optional
Reference, Prepare Image Prompt, then Generate Image. Description and Image Prompt share one
editor with native radio controls that appear after a prompt has been prepared.
Description is always the authoritative author input. Prepare Image Prompt creates one
editable Image Prompt proposal from the current Description; it never uses the
previous Image Prompt as input or inserts a separate clarification step. With a
Reference, the selected vision-capable Ollama model also inspects that card's
active generated image in the same preparation request. The exact authored
prompt used to generate that Reference image, including its reviewed Image
Prompt or legacy enriched text, provides the primary semantics for its identity
and visual style, while the pixels provide visible evidence and missing detail.
Without a Reference, the request is text-only.

The optional Reference has no fixed Subject, Style, or Setting role. The
Description states what should carry over or change, while preparation interprets
the image and produces a concrete final-image proposal for review. Definite
continuity can retain stable visible identity, construction, and rendering
treatment; an explicit target subject, state, style, palette, setting,
viewpoint, or composition overrides the corresponding reference trait. A vague
change may produce a plausible concrete proposal that the author can edit or
replace by revising the Description and enriching again.

An Image Prompt is Current only while its source Description, exact Reference
card/revision/background snapshot, selected Ollama model, and preparation
contract still match. Out-of-date prompts remain visible and editable, but
Generate requires a current prompt. Preparation validates structured output,
preserves exact affirmatively authored quoted text, respects explicit quoted-text
exclusions, rejects recognized object-state reversals and model-process
language, and requires the meaning of every explicit authored visual property
to survive without imposing special Description wording. It makes one
constrained repair attempt for a valid but conflicting proposal. Private model
deliberation is never persisted.

MFLUX receives the reviewed Image Prompt unchanged and, when selected, the same
Reference image exactly once. It receives no hidden role instructions or source
card prose. Generated images are the only supported background source.
Generate, image removal, Image Prompt preparation, and direct Image Prompt edits apply to the
active revision through document commands. Completed generation can be kept
there, moved into a new complete revision, or undone. Existing images remain
visible until replacement succeeds, and successful changes offer a
dismissible, history-safe Undo action in the notification bar.

When a schema-v5 bundle is opened, legacy Subject, Style, and Setting
assignments collapse deterministically to one Reference in that order of
precedence, and Enriched Description becomes Image Prompt. Historical
generation metadata remains readable in its original role-based form. Migrated
prompts without current model and contract provenance remain visible but must
be enriched again before generation.

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
action. Run presents Back and Restart as standard-size controls and hides the
authoring-only card name and version header. Run-only navigation and overlay
controls stay hidden in Author mode. Cards without hotspots are valid terminal
cards and do not produce a warning.
Local AI services are checked automatically when Author mode is entered; Run
mode starts no AI work and hides model selection. In Author mode, the status bar selects
an installed vision-capable Ollama model and either FLUX.2 Klein 4B or FLUX.2
Klein 9B KV.

Transient outcomes, failures, Run warnings, and Undo actions appear in one
notification bar beneath the main panes and directly above the status bar.
Reversible deletions and replacements apply directly and offer Undo. Completed
Description and image generations also offer Create New Version, which restores
the prior current revision and activates a complete new revision containing the
result, plus an explicit Keep action that retains it on the current revision.
Field validation remains beside the responsible input.
AI failures and recovery actions stay in the notification bar rather than the
model selectors.

## Setup and run

Prerequisites: Python 3.12 and [uv](https://docs.astral.sh/uv/). To enable
generation, also run Ollama with an installed vision-capable model (default:
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
uv run hypergen-eval image-prompts --validate-only
uv run hypergen-eval image-prompts
uv run hypergen-eval image-prompts-two-stage --validate-only
uv run hypergen-eval image-prompts-two-stage
uv run hypergen-eval image-prompts-evidence-gate --validate-only
uv run hypergen-eval image-prompts-evidence-gate
uv run hypergen-eval inline-references --validate-only
uv run hypergen-eval inline-references
uv run hypergen-eval flux-references --stack /path/to/Stack.hypergen
```

Each live command creates one immutable directory under `evals/runs/` with an
immediate, failure-safe `manifest.json` and retained artifacts; `--validate-only`
checks the Image Prompt dataset in place without a model call or run directory.
Smoke, image, and Image Prompt runs add checksums, JSON/CSV summaries, and a
static HTML report. Reference runs write detailed JSON results plus overall and
per-case contact sheets. Report rendering is offline and never calls a model.
The tracked default decision and evidence tradeoffs are in
[`evals/DECISION.md`](evals/DECISION.md).

The maintained
[Image Prompt benchmark](evals/cases/image_prompts/README.md) uses frozen
Reference images, checked provenance and checksums, and atomic human-scored
criteria rather than exact expected prose. `--validate-only` checks the complete
dataset without contacting a model; a live `image-prompts` run records the
production contract, raw responses, final prompts, and blank scorecards for
review.
The evaluation-only `image-prompts-two-stage` condition separates a
target-independent natural-language Reference account from text-only target
synthesis while reusing production response validation.
The `image-prompts-evidence-gate` condition instead gives final synthesis only
target-authorized Reference evidence, creating a hard boundary around discarded
visual context.
Production Image Prompt preparation presents Reference context before the
authoritative Description so the target remains the most recent input.
The paired `inline-references` suite uses four frozen benchmark References and
matched seeds to compare complete standalone prompts with explicit inline
Reference scope. It produces blinded A/B review sheets and keeps the condition
key separate until scoring is complete.

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
