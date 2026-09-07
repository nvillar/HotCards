# HotCards

HotCards is an experimental, local-first authoring tool for illustrated,
spatially interactive stacks of cards, inspired by classic HyperCard.

Authors describe each card in natural language, generate a background image
locally, place clickable polygon hotspots, and run the stack as an interactive
experience. All generation runs on-device.

**Status:** early developer proof of concept. Not packaged for end users.

**Platform:** Apple Silicon macOS. Local generation requires locally cached
MFLUX image and Stable Audio sound models; the authoring shell remains usable
when either is unavailable.

Stacks are stored as self-contained `.hotcards` directory bundles and
autosaved atomically after creation or opening. At startup, HotCards lists
stacks in `~/Documents/HotCards` and offers Open, Create, and confirmed
permanent Delete actions.
Only the current schema (13) is accepted; older and future schemas are rejected.
There is no runtime migration of older bundles.
Each stack stores one immutable fixed aspect ratio: Square 1:1, Landscape 4:3,
Portrait 3:4, or Widescreen 16:9. New Stack offers exactly those four formats
and defaults to Landscape 4:3; the format cannot be changed after creation.

Each card owns one or more numbered revisions. A revision contains its authored
Description, selected stack Style, optional generated background, up to two
ordered Reference cards, selected Generate output size, and hotspot set. Generate
output size is revision-local, defaults to Medium, and is copied with the
complete revision. The named long-edge tiers are Small (256 px), Medium
(512 px), Large (768 px), and Full (1024 px). The stack aspect ratio determines
the shorter edge, aligned to 16 pixels; for example, Landscape Full is
1024 × 768 and Widescreen Full is 1024 × 576. The compact
header above the canvas edits the card name and selects, duplicates, or deletes
revisions; the Author toolbar places the Run toggle on the left and opens the
stack-global Styles, Sounds, and Keys managers from the right. In Run mode it instead
shows navigation and hotspot visibility controls. An
image-generation step label and progress bar in the status bar show actual
MFLUX inference-step completion.

The Cards sidebar creates a new blank card immediately after the selected card.
Deleting a selected card moves selection to the card immediately before it, or
to the new first card when there was no preceding card. Command-D duplicates the
selected card immediately after its source. A card
duplicate contains exactly the active complete revision
with new card, revision, background, and hotspot identities. Generated image
bytes are copied into the duplicate card's own asset namespace, so either card
can be deleted independently. Self-navigation is remapped to the duplicate;
other destinations, References, Keys, Style selection, and Generate output size
are preserved. The duplicate is one undoable change and is named `Name Copy`,
then `Name Copy 2`, and so on. The asset and manifest commit as one rollback-safe
bundle transaction. Undo/Redo history retains the independent bytes only while
needed to restore the duplicate, and discarding that history reclaims the
unreferenced duplicate-owned asset without collecting unrelated bundle files.

The inspector tabs are Generate, Edit, and Hotspots. Generate contains
Description, Style, optional References, Resolution, and Generate Image.
The Edit tab contains Edit Instruction, Resolution, Edit,
and the current image's Edit History. Both image-authoring tabs use flat controls without
redundant section titles or group boxes. Resolution selectors show only named tiers,
with exact dimensions in tooltips, and select the current image size when entering
a card, revision, or newly replaced image (including Undo/Redo). Ordinary refreshes
preserve deliberate resolution choices. A
compatible aligned nonstandard image uses a selectable Current row. Generate
offers every named tier. Edit offers only the current size and higher
tiers. Generate requires a nonempty Description; Edit does not.
Both require an available MFLUX model.
The exact effective prompt is composed
deterministically from the Description followed by the selected Style text; no
language model prepares or rewrites it.

References apply only to Generate, never to Edit.
Reference order is authoritative. The first selected card is `image 1` and the
optional second card is `image 2`; authors use those positional labels directly
in the Description. References and Hotspot destinations use a shared movable,
resizable modeless card picker window with active-revision thumbnails in a
searchable responsive grid. Tiles show card names without numbering; ineligible
Reference cards remain visible but disabled. Compact remove buttons clear
selections through the existing undoable commands. Each active Reference
background is sent to MFLUX exactly once in that stable order, with no hidden
role instructions, card-name alias translation, or source-card prose. The exact
Description, ordered Reference snapshots, Style ID/name/text, and composed
render prompt are retained in generated-image provenance.

Image provenance is a strict typed operation record for direct Generate,
externally patched historical Generate, historical Evolve (stored as `refine`), Edit,
or an independent card duplicate. Duplicate provenance records its immediate
source and a flattened snapshot of the original image operation without
retaining a live source dependency. Provenance retains exact
prompts, model execution settings, seed, and actual output width and height;
derived operations capture historical source card/revision/background IDs,
decoded dimensions, the original operation seed (flattening duplicates), and
inherited accepted Edit instructions. These nonrecursive snapshots contain no
source asset paths. Source cards, revisions, and backgrounds can be deleted or
replaced without invalidating later images. The inherited Edit sequence is stored
once: historical Refine provenance exposes it unchanged, and Edit adds its accepted current
instruction. Captured source dimensions and settings are validated locally;
live inference still uses a private immutable source image and rejects stale
results or source replacement races. Assets remain retained while reachable from
the current document or Undo/Redo history, not by historical source attribution.

The production MFLUX 0.19.1 adapter routes plain Generate through
the regular model family, and Reference-backed Generate and Edit through the
Edit family. Model loading and inference share one process-local serialized
boundary with at most one compatible cached family/configuration. Cancellation
discards candidate output, and changing the selected model releases the prior
configuration.

Evolve is no longer an executable authoring operation. Existing historical
Refine provenance remains readable, displayable, editable through current
operations, duplicable, and saveable without changing its recorded facts.

Edit also replaces the current version's background in place. It
sends that image alone to Flux2KleinEdit with the exact authored Edit Instruction.
When a Style is selected, its exact text is appended behind the UI as a visual
continuity addendum unless the authored instruction explicitly changes the visual
treatment. No other preservation or editing instructions are added. Description
and Generate References remain unchanged on the revision but are not model inputs.
The effective prompt is deterministic, retained in provenance, and must fit the
model's 512-token budget without truncation.
Output defaults to the current image's exact decoded dimensions, including
aligned non-preset sizes, and also offers only named tiers with strictly greater
pixel area. Each Edit
uses a fresh seed and appends its accepted instruction to ordered provenance
lineage. Undoing an Edit restores its exact authored instruction in the same
card/version for adjustment and another attempt, without overwriting newer input.
This also works across repeated Undo/Redo. Redo clears only an untouched,
automatically restored instruction, never a newer draft or a user recall.
Generate starts a fresh accepted Edit lineage.

Edit History shows only accepted authored instructions for the active image,
oldest first with horizontal separators instead of numbering, including repeated
instructions and edits inherited through historical Refine provenance or card duplication. It never
displays expanded prompts or Style addenda. The history list stays visible when
empty, keeping Edit controls consistently top-aligned. New Image starts with no
accepted edits. Click a history row, or select it and press Enter or Space, to
recall its exact instruction into the editor without
changing the document or starting image generation. A recall counts as a new draft,
even when its text matches a previously restored instruction.

Generated images are the only supported background source. Generate and Edit
durably replace the active revision's background through one shared
image-and-manifest transaction and one Undo boundary. Each completed result
offers **Create New Version** (primary), **Undo**, and **Keep** (dismiss).
Keep makes no further document change. Create New Version restores the complete
previous version and appends and activates the accepted result with a new version
identity, sharing the immutable image without another model call. It is a
separate Undo boundary: undoing version creation leaves the result on the original
version; the next Undo reverses the image operation and, for Edit, restores its
instruction. Actions expire after another command and are unavailable in Run mode
or another project. Indeterminate saves show the authoritative
image but block mutations and defer result actions until a save retry establishes
durability. Delayed completion never clears a newer Edit draft.
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

The modeless Sounds utility window manages an ordered stack-global catalog of
named sound effects. Each Sound keeps an editable generation Prompt and a
duration from 1–30 seconds, defaulting to 2 seconds. Generate and Re-generate use
Stable Audio 3 Small-SFX through its optimized MLX runtime with the Pingpong
sampler, 8 steps, CFG 1.0, and a fresh random seed. Output is stored as 16-bit
PCM stereo WAV at 44.1 kHz with exact prompt, duration, seed, model, runtime,
sampler, and timing provenance. Existing audio remains available while a
replacement is generated, and replacement is one undoable asset-and-manifest
transaction. The manager previews audio, lists every hotspot usage, jumps to
the selected usage, and blocks deletion while the Sound is referenced.
Hotspot Sound assignment uses a movable, resizable modeless picker with a
searchable responsive grid. Every tile has a Play control above the Sound name;
Sounds without generated audio remain assignable but have Play disabled. A
compact remove button clears the assignment through the existing undoable
command. Picker previews share playback with the Sounds manager and stop when
the picker is cancelled, closed, or used to select a Sound.

The Hotspots inspector is the sole source of revision-local interaction
semantics. Every persisted hotspot has exactly one polygon. Its label is derived
automatically from its highest-priority effect: Go to, Play, Gain, then Lose,
with long labels using two-line list entries. The bordered When section requires
every Present key to exist and every Absent key not to exist. The bordered Then
section applies disjoint explicit Remove and Grant sets before its optional Go
to destination and optional Play Sound action. Renaming a Sound preserves
hotspot assignments through stable IDs. Keys are stack-global free-form names
with stable internal IDs.

The modeless Keys utility window manages that global catalog with the same list,
compact remove/add, and full-width Name patterns as Styles. It reports every
card revision and hotspot that Requires, Forbids, Removes, or Grants the
selected key, and can jump to that exact hotspot. Reopening Keys raises the
existing window. Renaming updates every display through stable references.
In-use keys cannot be deleted. Keys can also be created while adding a
condition or key change.

While the Hotspots tab is active, clicking empty canvas or pressing the
inspector `+` begins drawing a new hotspot. The draft is not persisted until a
valid polygon is completed, so Escape or leaving the tab cancels without
creating a hotspot. While drawing, overlapping vertex clicks are ignored and
clicking the first vertex closes a valid polygon. Drag a hotspot or vertex to
move it, use the edge `+` or double-click an edge to add a vertex, and press
Delete to remove the selected vertex or hotspot. Context menus expose vertex
editing and hotspot deletion, but not alternate creation actions. Successful
edits offer a dismissible Undo. Replacing a background preserves its hotspots
so the author can review and adjust them manually. Complete backgrounds at
different pixel resolutions are smoothly fitted and centered inside the
stack's fixed logical card format without stretching or cropping; narrow
rounding differences use letterboxing or
pillarboxing. The canvas remains automatically fitted without zoom, pan, manual
fit, or image-clear controls. Normalized hotspot geometry follows the fitted
image bounds in Author and Run modes.

Run mode keeps its current keys only in the session. It starts empty, Back
retains keys, Restart clears them, and leaving Run discards them. Condition-
failing and actionless placeholder hotspots are omitted from hover, overlays,
and hit testing; existing z-order selects the topmost remaining hotspot.
Activation rechecks conditions, removes keys, grants keys, and finally
navigates, so the destination sees the updated state. A clicked Sound starts
after that navigation and continues on the destination card. New playback
replaces existing playback; navigation, Back, Restart, and leaving Run first
stop the previous sound. Pure key and sound actions are valid. Run also retains
deterministic Back/Restart navigation history and
configurable overlays. It opens on the current Author card and, when that
differs from the configured start card, offers a dismissible restart action.
Run presents Back and Restart as standard-size controls and hides the
authoring-only card name and version header. Run-only navigation and overlay
controls stay hidden in Author mode. Cards without hotspots are valid terminal
cards and do not produce a warning.
Local MFLUX availability is checked automatically when Author mode is entered;
Run mode starts no AI work. Settings → Models selects the Image and Sound
models. Image offers FLUX.2 Klein 4B and FLUX.2 Klein 9B KV; Sound currently
uses Stable Audio 3 Small-SFX.

Transient outcomes, failures, Run warnings, and Undo actions appear in one
notification bar beneath the main panes and directly above the status bar.
Reversible deletions and replacements apply directly and offer Undo. Completed
Generate and Edit operations also offer Create New Version, which restores
the prior current revision and activates a complete new revision containing the
result, plus an explicit Keep action that retains it on the current revision.
Field validation remains beside the responsible input.
AI failures and recovery actions stay in the notification bar.

## Setup and run

Prerequisites: Python 3.12 and [uv](https://docs.astral.sh/uv/). To enable
generation, cache the selected MFLUX model and
`stabilityai/stable-audio-3-optimized` Small-SFX MLX weights locally. Accept the
Stable Audio model terms and authenticate with Hugging Face outside HotCards;
credentials are never stored in a stack or Qt settings. FLUX.2 Klein 4B uses
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
- `generation/` — deterministic MFLUX image generation and attributed Stable
  Audio 3 Small-SFX MLX sound generation.
- `application/` — document controller, typed commands, session undo, workers.
- `ui/` — PySide6 Author and Run interface.
- `evaluation/` — `hotcards-eval` harness reusing production adapters.

## Roadmap

Planned work, acceptance criteria, and progress live in GitHub Issues and the
[**HotCards POC** milestone](https://github.com/nvillar/HotCards/milestone/1).
