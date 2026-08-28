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
Only the current schema is accepted; older and future schemas are rejected.

Each card owns one or more numbered revisions. A revision contains its authored
Description, selected stack Style, optional prepared Image Prompt, optional
generated background, up to two ordered Reference cards, and hotspot set. The compact
header above the canvas edits the card name and selects, duplicates, or deletes
revisions; the toolbar provides a single Author/Run mode toggle and manages
hotspot visibility. An
adjacent step label and progress bar to the left of the model selectors show
actual MFLUX inference-step completion during image generation and an
indeterminate state during Image Prompt preparation.

The inspector tabs are Image, Styles, Hotspots, and Keys. Image
follows the authoring sequence Description, Style, optional References, Prepare
Image Prompt, then Generate Image. Description and Image Prompt share one
editor with native radio controls that appear after a prompt has been prepared.
The Style selector chooses a named stack-wide treatment; No Style is always
available.
Description is always the authoritative author input. Prepare Image Prompt creates one
editable Image Prompt proposal from the current Description; it never uses the
previous Image Prompt as input or inserts a separate clarification step. With
References, the selected vision-capable Ollama model also inspects their active
generated images in the same preparation request. The exact authored prompt
used to generate each Reference image, including its reviewed Image Prompt or
legacy enriched text, provides the primary semantics for its identity and
visual style, while the pixels provide visible evidence and missing detail.
Without a Reference, the request is text-only. Reference order is authoritative:
the first selected card is `image 1` and takes precedence where the Description
does not resolve an ambiguity; the optional second card is `image 2` and
contributes only relevant assigned or inferred properties. Preparation
recognizes selected card names and user-facing Reference aliases, then emits
explicit canonical `image 1` and `image 2` relationships.

References have no fixed Subject, Style, or Setting roles. The
Description states what should carry over or change, while preparation interprets
the image and produces a concrete direct editing instruction for review.
Reference-backed Image Prompts explicitly state which entity, setting, or
treatment comes from each used image and how it changes. Definite continuity can
retain stable visible identity, construction, and rendering treatment; an
explicit target subject, state, style, palette, setting, viewpoint, or
composition overrides the corresponding reference trait. Preparation keeps
the instruction concise, names only the stable properties needed for the
requested continuity, and does not preserve or inventory Reference details
that the Description changes or does not need. A vague change may produce a
plausible concrete proposal that the author can edit.

An Image Prompt is Current only while its source Description, exact ordered
Reference card/revision/background snapshots, selected Ollama model, and preparation
contract still match. Out-of-date prompts remain visible and editable, but
Generate requires a current prompt. Manually editing a non-empty Image Prompt
marks the reviewed text current against the present Description, Reference,
model, and preparation contract, so generation can proceed without another
model preparation call. Preparation validates structured output,
preserves exact affirmatively authored quoted text, respects explicit quoted-text
exclusions, rejects recognized object-state reversals and inappropriate
model-process language, requires canonical numbered attribution when References are attached, and
requires the meaning of every explicit authored
visual property to survive without imposing special Description wording. It makes one
constrained repair attempt for a valid but conflicting proposal. Private model
deliberation is never persisted.

Image Prompt preparation does not consume the selected Style. At final image
generation, HyperGen deterministically appends the selected Style text to the
reviewed Image Prompt and sends each Reference image exactly once in numbered
order. MFLUX receives no hidden role instructions or source card prose. The
exact Style ID, name, text, and composed render prompt are retained in generated
image metadata. Generated images are the only supported background source.
Generate, image removal, Image Prompt preparation, and direct Image Prompt edits apply to the
active revision through document commands. Completed generation can be kept
there, moved into a new complete revision, or undone. Existing images remain
visible until replacement succeeds, and successful changes offer a
dismissible, history-safe Undo action in the notification bar.

The Styles tab manages the stack's ordered, editable Style library with compact
remove/add controls and full-width Name and Style Text fields. The built-in
library includes HyperCard, Cinematic Film, Isometric Game, Pixel Art,
Watercolor Painting, Color Pencil, Pencil Sketch, Glazed Ceramic, Graphic
Novel, and Miniature Toy. A new stack starts with HyperCard selected. An
explicit Style or No Style selection becomes the default for subsequently
created cards. Deleting a Style clears every revision that selected it and is
reversible through Undo.

The Hotspots inspector is the sole source of revision-local interaction
semantics. A new hotspot is persisted and selected immediately, even before it
has an area. Its label is derived automatically in Remove, Grant, then
destination order, with long labels using two-line list entries. The bordered
When section requires every Present key to exist and every Absent key not to
exist. The bordered Then section applies disjoint explicit Remove and Grant
sets before its optional Go to destination. Keys are stack-global free-form
names with stable internal IDs.

The Keys tab manages that global catalog with the same list, compact remove/add,
and full-width Name patterns as Styles. It reports every card revision and
hotspot that Requires, Forbids, Removes, or Grants the selected key, and can
jump to that exact hotspot. Renaming updates every display through stable
references. In-use keys cannot be deleted. Keys can also be created while
adding a condition or key change.

While the Hotspots tab is active, clicking empty canvas begins a polygon for the
selected hotspot and creates one first when needed. Leaving the tab cancels any
unfinished polygon and hides its authoring overlays. Canvas editing uses
hierarchical hotspot, area, and vertex selection: drag an area or vertex to
move it, use the edge `+` or double-click an edge to add a vertex, press Delete
to remove the selected vertex or area, and press Escape to step back through
the selection. Context menus expose the same geometry actions, and successful
edits offer a dismissible Undo. Replacing a background preserves its hotspots
so the author can review and adjust them manually.

Run mode keeps its current keys only in the session. It starts empty, Back
retains keys, Restart clears them, and leaving Run discards them. Condition-
failing and actionless placeholder hotspots are omitted from hover, overlays,
and hit testing; existing z-order selects the topmost remaining hotspot.
Activation rechecks conditions, removes keys, grants keys, and finally
navigates, so the destination sees the updated state. Pure key actions are
valid. Run also retains deterministic Back/Restart navigation history and
configurable overlays. It opens on the current Author card and, when that
differs from the configured start card, offers a dismissible restart action.
Run presents Back and Restart as standard-size controls and hides the
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
uv run hypergen-eval style-presets --validate-only
uv run hypergen-eval style-presets
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
The `style-presets` suite renders two style-neutral scenes with matched seeds
under ten proposed deterministic Style suffixes plus an unstyled control. It
produces per-scene and cross-scene contact sheets for assessing Style fidelity,
subject and composition preservation, consistency, and artifact leakage.

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
