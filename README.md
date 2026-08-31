# HotCards

HotCards is an experimental, local-first authoring tool for illustrated,
spatially interactive stacks of cards, inspired by classic HyperCard.

Authors describe each card in natural language, generate a background image
locally, place clickable polygon hotspots, and run the stack as an interactive
experience. All generation runs on-device.

**Status:** early developer proof of concept. Not packaged for end users.

**Platform:** Apple Silicon macOS. Local generation requires a locally cached
MFLUX image model; the authoring shell remains usable when it is unavailable.

Stacks are stored as self-contained `.hotcards` directory bundles and
autosaved atomically after creation or opening. At startup, HotCards lists
stacks in `~/Documents/HotCards` and offers Open, Create, and confirmed
permanent Delete actions.
Only the current schema is accepted; older and future schemas are rejected.
Each stack stores one immutable fixed aspect ratio: Square 1:1, Landscape 4:3,
Portrait 3:4, or Widescreen 16:9. New Stack offers exactly those four formats
and defaults to Landscape 4:3; the format cannot be changed after creation.

Each card owns one or more numbered revisions. A revision contains its authored
Description, selected stack Style, optional generated background, up to two
ordered Reference cards, selected Generate resolution, and hotspot set. Generate
resolution is revision-local, defaults to 512 square-equivalent pixels, and is
copied with the complete revision. The supported presets are 256, 512, 768, and
1024. The compact
header above the canvas edits the card name and selects, duplicates, or deletes
revisions; the Author toolbar places the Run toggle on the left and opens the
stack-global Styles and Keys managers from the right. In Run mode it instead
shows navigation and hotspot visibility controls. An
adjacent step label and progress bar to the left of the image-model selector
show actual MFLUX inference-step completion during image generation.

The Cards sidebar and Command-D shortcut duplicate the selected card immediately
after its source. A card duplicate contains exactly the active complete revision
with new card, revision, background, and hotspot identities. Generated image
bytes are copied into the duplicate card's own asset namespace, so either card
can be deleted independently. Self-navigation is remapped to the duplicate;
other destinations, References, Keys, Style selection, and Generate resolution
are preserved. The duplicate is one undoable change and is named `Name Copy`,
then `Name Copy 2`, and so on. The asset and manifest commit as one rollback-safe
bundle transaction. Undo/Redo history retains the independent bytes only while
needed to restore the duplicate, and discarding that history reclaims the
unreferenced duplicate-owned asset without collecting unrelated bundle files.

The inspector tabs are Generate, Refine, Edit, and Hotspots. Generate
follows the authoring sequence Description, Style, optional References,
Resolution, then Generate Image. Resolution labels show the exact output width
and height for the stack format. Generate requires a nonempty Description and
an available MFLUX model. The exact effective prompt is composed
deterministically from the Description followed by the selected Style text; no
language model prepares or rewrites it.

Reference order is authoritative. The first selected card is `image 1` and the
optional second card is `image 2`; authors use those positional labels directly
in the Description. Each active Reference background is sent to MFLUX exactly
once in that stable order, with no hidden role instructions, card-name alias
translation, or source-card prose. The exact Description, ordered Reference
snapshots, Style ID/name/text, and composed render prompt are retained in
generated-image provenance.

Image provenance is a strict typed operation record for direct Generate,
externally patched historical Generate, Refine, Edit, or an independent card
duplicate. Duplicate provenance records its immediate source and a flattened
snapshot of the original image operation without retaining a live source
dependency. Provenance retains exact
prompts, model execution settings, seed, and actual output width and height;
derived operations also identify their source revision and background. A source
revision cannot be deleted or have its background replaced while another
retained revision derives from it. Deleting an entire card may remove an
image-evolution chain contained wholly inside that card.

The production MFLUX 0.19.1 adapter routes plain Generate and Refine through
the regular model family, and Reference-backed Generate and Edit through the
Edit family. Model loading and inference share one process-local serialized
boundary with at most one compatible cached family/configuration. Cancellation
discards candidate output, and changing the selected model releases the prior
configuration.

Refine is exposed as an automatic-version workflow for the current canvas
image. It uses regular Flux2Klein img2img with Reimagine 0.25, Balanced 0.50,
or Preserve 0.75 transformation strength and reuses the current image's seed.
Only output presets with a strictly greater pixel area than the decoded current
image are offered. The current background is the sole image input; Generate
References are never resent. Its deterministic prompt contains the current
Description, selected Style, and any ordered accepted Edit instructions already
present in the source, with the current Description explicitly authoritative.
Success preserves the source and automatically appends and activates one
complete derived revision. The new image and manifest commit as one
rollback-safe transaction, and Undo/Redo retain its owned asset only while
needed.

Edit is also an automatic-version workflow for the current canvas image. It
sends that image alone to Flux2KleinEdit with the authored Edit Instruction and
six explicit Preserve choices; Description, Style, and Generate References are
copied into the new revision but are not model inputs. The expanded prompt is
deterministic and must fit the model's 512-token budget without truncation.
Output defaults to the current image's exact decoded dimensions, including
legacy non-preset sizes, and also offers strictly higher-area presets. Each Edit
uses a fresh seed and appends its accepted instruction to ordered provenance
lineage. Later Refine operations preserve that lineage unless it conflicts with
the current authoritative Description; Generate ignores it.

Generated images are the only supported background source. Generate and image
removal apply to the active revision through document commands. Completed
generation can be kept there, moved into a new complete revision, or undone.
Existing images remain visible until replacement succeeds, and successful
changes offer a dismissible, history-safe Undo action in the notification bar.

The modeless Styles utility window manages the stack's ordered, editable Style
library with compact remove/add controls and full-width Name and Style Text
fields. Reopening Styles raises the existing window rather than creating a
second manager. The built-in
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

The modeless Keys utility window manages that global catalog with the same list,
compact remove/add, and full-width Name patterns as Styles. It reports every
card revision and hotspot that Requires, Forbids, Removes, or Grants the
selected key, and can jump to that exact hotspot. Reopening Keys raises the
existing window. Renaming updates every display through stable references.
In-use keys cannot be deleted. Keys can also be created while adding a
condition or key change.

While the Hotspots tab is active, clicking empty canvas begins a polygon for the
selected hotspot and creates one first when needed. Leaving the tab cancels any
unfinished polygon and hides its authoring overlays. Canvas editing uses
hierarchical hotspot, area, and vertex selection. While drawing, overlapping
vertex clicks are ignored and clicking the first vertex closes a valid polygon.
Drag an area or vertex to move it, use the edge `+` or double-click an edge to
add a vertex, press Delete to remove the selected vertex or area, and press
Escape to step back through the selection. Context menus expose the same
geometry actions, and successful edits offer a dismissible Undo. Replacing a
background preserves its hotspots so the author can review and adjust them
manually. Complete backgrounds at different pixel resolutions are smoothly
fitted and centered inside the stack's fixed logical card format without
stretching or cropping; narrow rounding differences use letterboxing or
pillarboxing. Normalized hotspot geometry follows the fitted image bounds in
Author and Run modes.

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
Local MFLUX availability is checked automatically when Author mode is entered;
Run mode starts no AI work and hides model selection. In Author mode, the
status bar selects either FLUX.2 Klein 4B or FLUX.2 Klein 9B KV.

Transient outcomes, failures, Run warnings, and Undo actions appear in one
notification bar beneath the main panes and directly above the status bar.
Reversible deletions and replacements apply directly and offer Undo. Completed
image generation also offers Create New Version, which restores
the prior current revision and activates a complete new revision containing the
result, plus an explicit Keep action that retains it on the current revision.
Field validation remains beside the responsible input.
AI failures and recovery actions stay in the notification bar rather than the
model selectors.

## Setup and run

Prerequisites: Python 3.12 and [uv](https://docs.astral.sh/uv/). To enable
generation, cache the selected MFLUX model locally. FLUX.2 Klein 4B uses
Apache 2.0; FLUX.2 Klein 9B KV uses the FLUX Non-Commercial License. The 9B KV
choice requires both the regular 9B weights for text-only generation and the
9B KV weights for reference generation.

```sh
uv sync
uv run hotcards
```

## Evaluation harness

```sh
uv run hotcards-eval smoke
uv run hotcards-eval images
uv run hotcards-eval style-presets --validate-only
uv run hotcards-eval style-presets
uv run hotcards-eval flux-references --stack /path/to/Stack.hotcards
```

Each live command creates one immutable directory under `evals/runs/` with an
immediate, failure-safe `manifest.json` and retained artifacts. Smoke and image
runs add checksums, JSON/CSV summaries, and a static HTML report. Reference runs
write detailed JSON results plus overall and per-case contact sheets. Report
rendering is offline and never calls a model.
The tracked default decision and evidence tradeoffs are in
[`evals/DECISION.md`](evals/DECISION.md).

The `style-presets` suite renders two style-neutral scenes with matched seeds
under ten proposed deterministic Style suffixes plus an unstyled control. It
produces per-scene and cross-scene contact sheets for assessing Style fidelity,
subject and composition preservation, consistency, and artifact leakage.

## Architecture at a glance

- `domain/` — in-memory stack model, geometry, validation.
- `storage/` — human-readable `*.hotcards` bundle storage.
- `generation/` — deterministic MFLUX prompt composition, image generation,
  and Reference delivery.
- `application/` — document controller, typed commands, session undo, workers.
- `ui/` — PySide6 Author and Run interface.
- `evaluation/` — `hotcards-eval` harness reusing production adapters.

## Roadmap

Planned work, acceptance criteria, and progress live in GitHub Issues and the
[**HotCards POC** milestone](https://github.com/nvillar/HotCards/milestone/1).
