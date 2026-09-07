# AGENTS.md

Durable coding and repository guidance for HotCards. For what the project is
and how to run it, see `README.md`. For planned work and acceptance criteria,
see GitHub Issues and the
[**HotCards POC** milestone](https://github.com/nvillar/HotCards/milestone/1).
Do not duplicate that material here.

## Environment and commands

Use `uv` only; do not add ad hoc `pip` instructions.

```sh
uv sync
uv run hotcards
uv run pytest
uv run ruff check .
uv run ruff format .
uv run hotcards-eval smoke
uv run hotcards-eval images
uv run hotcards-eval style-presets --validate-only
uv run hotcards-eval style-presets
uv run hotcards-eval flux-references --stack /path/to/Stack.hotcards
```

Ordinary automated tests must not require live model calls. Use recorded
responses and fakes in `pytest`; use `hotcards-eval` or explicit smoke commands
for live MFLUX runs.

## Repository layout

- `src/hotcards/domain/` — stack models, geometry, validation.
- `src/hotcards/storage/` — bundle persistence.
- `src/hotcards/generation/` — prompt builders and model adapters.
- `src/hotcards/application/` — document controller, commands, workers.
- `src/hotcards/ui/` — PySide6 widgets.
- `src/hotcards/evaluation/` — evaluation harness implementation.
- `tests/` — focused unit tests.
- `evals/cases/` — version-controlled cases, rubrics, and permitted frozen inputs.
- `evals/runs/` — generated immutable run output; keep it out of Git.

## Architectural constraints

- Keep one authoritative in-memory stack document. UI widgets render state and
  issue commands; they do not own independent domain truth.
- Use Pydantic at serialized and model-response boundaries.
- Keep image actions, model actions, and storage out of widgets.
- Reuse production prompt builders, schemas, adapters, and geometry validation
  in the evaluation harness. Do not fork generation behavior.
- Keep each card's complete authoring state in one of its numbered revisions:
  Description, selected stack Style, optional background, ordered Reference
  cards, selected Generate output size, and hotspot set. Visible revision
  numbers are positional; stable UUIDs remain internal.
- Store one immutable stack aspect ratio using only Square 1:1, Landscape 4:3,
  Portrait 3:4, or Widescreen 16:9. Generate output size is revision-local and
  defaults to Medium. Use named long-edge tiers only: Small 256 px, Medium
  512 px, Large 768 px, and Full 1024 px. Square uses the tier on both edges;
  landscape and widescreen use it as width; portrait uses it as height. Derive
  the shorter edge from the aspect ratio and round it to the nearest multiple
  of 16, minimum 16. Continue storing exact actual output width and height in
  image provenance.
- Keep at least one revision per card. Duplicate a complete revision, including
  its Generate output size, hotspot semantics, and immutable background
  reference.
- Duplicate the selected card from only its active complete revision and insert
  it immediately after the source as one undoable change. Mint new card,
  revision, background, and Interaction IDs; remap self-navigation to the new
  card and preserve other destinations, References, Keys, Style selection, and
  Generate output size. Copy exact background bytes into the duplicate card's
  own validated asset namespace through one rollback-safe asset/manifest
  transaction. Bind copy and cleanup to securely opened bundle objects and the
  owned file identity; never follow asset symlinks. Retain duplicate-owned bytes
  while reachable from the current document or Undo/Redo history, then reclaim
  them when that history is discarded. Flatten duplicate provenance to the
  original non-duplicate operation while recording the immediate source
  informationally, so deleting the source never invalidates the duplicate.
- Keep an ordered collection of at most two optional Reference cards per
  revision and reject self-references and duplicates. Display slots as `1.` and
  `2.` under one References section; clearing slot 1 promotes slot 2. Resolve
  each active accepted background for image generation. Require user-facing
  guidance to use positional `image 1` and `image 2` labels; do not translate
  card names or aliases. Send each Reference exactly once to MFLUX in stable
  order without hidden role instructions or complete source-card prose. Use one
  shared movable, resizable modeless searchable thumbnail-grid picker window for
  Reference 1, Reference 2, and Hotspot destinations. Preserve its adjusted
  geometry while the document remains open. Show card names without card
  numbers. Keep ineligible Reference cards visible but disabled without
  explanatory labels or tooltips. Provide a bottom-right Cancel button matching
  the modeless utility windows. Clear selections with the same compact `−`
  controls used elsewhere, through the existing undoable commands.
- Keep an ordered, stack-owned library of editable named Styles with stable
  UUIDs. Store the selected Style on each revision and persist the last explicit
  Style or No Style selection as the default for new cards. Deleting a Style
  clears its revision selections through one undoable command.
- Compose background prompts deterministically from the current nonempty
  Description followed by the selected Style text. Capture the exact
  Description, ordered Reference snapshots, Style ID, name, text, and composed
  prompt in generated-image provenance. Description, Style, Reference,
  background, project, revision, model, or mode changes must suppress an
  in-flight stale image result. Hotspots must not alter image-generation
  prompts.
- In Generate, Evolve, and Edit resolution selectors, show only the named tier
  names and expose exact dimensions in tooltips. Insert one selectable Current
  row for an aligned nonstandard current image. On card, revision, or background
  identity entry (including same-revision replacement and Undo/Redo), select the
  current image size in all three controls. Ordinary rerenders preserve deliberate
  choices, focused Description drafts, and valid same-revision hotspot selections.
  Generate offers every named tier; Evolve and Edit offer only the current size
  and higher-area tiers. Treat resolution selection as a passive setting change
  without a notification.
- Evolve (the persisted `refine` operation) only the readable current
  background through regular Flux2Klein
  img2img; never resend its Generate References. Offer Reimagine 0.25,
  Balanced 0.50, and Preserve 0.75 as Source Similarity choices with
  user-facing descriptions but no numeric values, default output to the exact
  current size, and offer only higher named tiers.
  Reuse the
  source operation seed after flattening duplicate provenance. Pass MFLUX a
  private immutable snapshot copied from a securely opened source asset, and
  reject the result if that logical asset changes before acceptance. Compose
  the exact Evolve prompt from current Description, selected Style text, then
  ordered authored accepted Edit instructions. State that the source already
  contains those edits, preserve them unless they conflict, and make current
  Description authoritative. On success atomically store the image and replace
  the current revision's background, preserving its other complete authoring
  state through one Undo boundary.
- Edit only the readable current background through Flux2KleinEdit with one
  direct authored Edit Instruction. Send that exact trimmed instruction without
  adding preservation or editing guidance. If a Style is selected, append only
  the exact selected Style text behind the UI as a visual-continuity addendum,
  unless the authored instruction explicitly changes the visual treatment.
  Keep legacy Preserve metadata readable, but do not expose or apply Preserve
  controls to new edits. Validate the deterministic expanded prompt against the
  FLUX Edit tokenizer's hard 512-token budget without rewriting or truncation,
  use a fresh random seed, and never send Description or Generate References.
  Retain the exact effective prompt, including Style text, in provenance.
  When Undo reverses a completed Edit, restore that Edit's exact authored
  instruction only in its card/revision context, without overwriting newer input.
  Bind restoration to that Edit's exact Undo token, retaining the session-local
  metadata across Undo/Redo and pruning it when history or the project is discarded.
  Redo may clear only the exact automatically restored draft; user recalls count
  as newer drafts even when their text is identical.
  Default output to the source image's exact decoded dimensions and additionally
  offer only higher-area presets. Pass
  MFLUX a private immutable no-follow source snapshot and reject replacement
  races before acceptance. Append the accepted Edit to inherited flattened
  lineage, atomically store the image, and replace the current revision's
  background through one Undo boundary, preserving its other authoring state.
  Clear only the matching submitted draft and context after definitive durable
  success, including delayed save retries; never clear a newer draft.
- Persist generated backgrounds with a strict discriminated provenance union
  for direct Generate, externally patched legacy Generate, Refine, Edit, and
  independent card duplication.
  Keep operation-specific prompts and settings typed rather than accumulating
  nullable fields. Accept schema 13 only, without runtime loading or migration
  of schema 12. Refine and Edit capture a nonrecursive historical source snapshot:
  card/revision/background IDs, decoded width/height, flattened source operation
  seed, and one canonical inherited accepted Edit sequence. Store no source
  paths or recursively embedded provenance. Refine exposes that sequence unchanged;
  Edit exposes it plus exactly its accepted current Edit, without persisting a
  second lineage array. Preserve exact instruction, Preserve, and expanded-prompt
  facts. Source attribution need not resolve to retained cards or revisions, and
  may name the result's revision, but source and result background IDs must differ.
  Current Generate and Refine dimensions must match their typed
  output-size selection. Preset output must match the stack aspect ratio;
  exact Generate output must be positive, 16-aligned, and aspect-compatible;
  current Refine/Edit output must equal its captured source dimensions. Derived
  preset dimensions must match their named tier and stack aspect ratio and have
  strictly greater area than the captured source. Refine must reuse the captured
  source seed. Historical source dimensions are positive facts, not necessarily
  modern aligned/aspect-compatible output sizes. Legacy
  Generate preserves its exact historical prompt and dimensions without
  imposing a modern preset. Historical provenance never blocks source revision
  deletion, background replacement, or whole card deletion. Keep asset retention
  tied to the current document and Undo/Redo history, not source attribution.
  Preserve live immutable source snapshots, decoded-dimension request checks, and
  stale-result/race rejection independently of these historical facts.
- Route all production and evaluation Generate, Evolve, and Edit inference
  through one typed MFLUX adapter. Plain Generate and Evolve use the regular
  family; Reference-backed Generate and Edit use the Edit family. Serialize
  model loading and inference through one stable process-local invocation
  thread, cache at most one compatible family/model/quantization configuration,
  and release it when switching configuration. Cancellation while queued or
  running must publish no output; interrupted active models must not be reused.
- Keep inspector tabs ordered Generate, Evolve, Edit, Hotspots. Generate and Evolve
  each expose Description and Style, bound to the same revision state. Share one
  transient Description document between the two views, commit it through the
  existing commands, and synchronize Style selections from the controller without
  duplicate commands. Preserve a focused draft in either view during same-revision
  rendering. Generate contains References, revision-local
  Resolution, and Generate Image; Evolve contains Source Similarity, Resolution,
  and Evolve. References belong to Generate only and are never sent to Evolve or
  Edit. Keep Generate, Evolve, and Edit controls flat, without redundant section
  titles or group boxes, on consistent inset scrollable content.
  Expose exact output dimensions and positional Reference guidance in tooltips,
  and keep generation provenance in the Generate button tooltip.
  Keep Evolve and Edit button tooltips to one concise action
  sentence plus a disabled-state reason when needed. The Edit tab contains
  Edit Instruction, Resolution, and Edit controls, followed by Edit History.
  The current canvas/header is the implicit source for both. Open
  Styles, Sounds, and Keys from right-aligned Author-toolbar actions into separate
  modeless singleton utility windows. Keep them as stack-global list managers
  backed by the authoritative controller, with compact remove/add controls and
  vertically stacked full-width fields. Hide the manager actions and windows in
  Run mode, and close or rebind them on project replacement. Require a nonempty
  Description for Generate and Evolve, but not Edit.
- Derive Edit History exclusively from the active background's
  `image_edit_lineage()`, including duplicate originals. Show chronological,
  oldest-first word-wrapped authored instructions without numbering, with native
  horizontal separators and vertical space between entries. Retain duplicates
  and exact text without expanded prompts, Style addenda, or timestamps.
  Keep the blank history list visible when empty and Edit controls top-aligned.
  Generate clears lineage,
  Evolve inherits it, and Edit appends. Mouse and keyboard recall only repopulate
  Edit Instruction through `Inspector.set_edit_instruction()`, advancing its draft
  serial once per action without a command, Undo entry, navigation, or model call.
  Block activation/selection signals while refreshing history on image, card,
  revision, Undo/Redo, or project changes; never overwrite the instruction draft.
- In Hotspots, keep When and Then as normal labels outside untitled grouped
  panels using the same native treatment as the New Image, Evolve, and Edit sections.
  Keep the list, ordering controls, labels, and grouped panels on one inset
  scrollable content surface matching Generate and Edit; do not nest a separate
  zero-margin rule viewport.
- Create stacks with one native format selector containing exactly Square 1:1,
  Landscape 4:3, Portrait 3:4, and Widescreen 16:9, defaulting to Landscape.
  Do not expose arbitrary dimensions or a post-creation aspect-ratio setting.
  Fit and center every complete background inside the stack's logical card
  geometry with aspect-preserving scaling and letterboxing or pillarboxing as
  needed. Use the fitted image bounds for all normalized hotspot rendering,
  gestures, and Run hit testing; bars are noninteractive.
- Support generated backgrounds only; do not add image import. Apply Generate,
  Evolve, and Edit in place through one shared durable image/manifest
  transaction and document command. Register accepted app-owned images and reclaim
  only identity-matching files after they leave the current document and all
  Undo/Redo history.
  Keep an existing image visible until replacement succeeds. An authoritative
  visible result with indeterminate durability blocks mutations and publishes no
  normal result actions until a definitive save retry creates its single history
  entry. Preserve completion context even when a durable commit reports cleanup
  errors. Every durable Generate, Evolve, or Edit publishes one typed
  `AppliedImageChange` through `image_applied`, carrying its exact Undo token,
  card/revision identity, complete previous revision, and operation-specific
  payload (including Edit's authored instruction). Capture the token at durable
  history insertion before session notifications can issue another command.
  Retain Edit Undo metadata even when a newer command expires the result actions,
  and never render an old completion snapshot over the current document.
  Offer Create New Version as
  primary, Undo as secondary, and Keep as dismiss, bound to that exact current
  history token and unavailable in Run or a stale project. Keep changes no
  document state. `CreateImageRevisionCommand` restores the prior complete revision,
  appends and activates the accepted complete result with a new revision UUID,
  and shares immutable image bytes without invoking a model. This is a separate
  Undo boundary: undoing version creation leaves the applied image on the original
  revision and does not restore an Edit instruction; the next Undo reverses the
  actual image operation and restores its instruction when appropriate.
- Apply reversible deletions and replacements without confirmation. Report
  outcomes, failures, Run warnings, and Undo actions in the global notification
  bar; keep field validation beside its input and the status bar passive. Use a
  blocking decision dialog only when proceeding could lose persisted work and
  Undo cannot recover it.
- Store each hotspot set under exactly one complete card revision. Replacing a
  background preserves its hotspots so the author can review and adjust them
  manually.
- Keep the canvas automatically fitted to the complete image. Do not expose
  zoom, pan, manual fit, or image-clear controls.
- Require exactly one polygon per persisted hotspot. Begin new hotspots as
  transient canvas drafts and persist the hotspot and polygon atomically only
  after valid completion. Canceling a draft must not mutate the document, and
  deleting the polygon deletes the hotspot through one undoable command.
  Derive one simple label from each hotspot's highest-priority action:
  `Go to <Card>`, then `Play <Sound>`, then `Gain <Key>`, then `Lose <Key>`.
- Keep Author canvas selection hierarchical: a selected vertex belongs to a
  selected polygon, which belongs to the selected hotspot. Inspector
  synchronization and same-revision edits must not discard a valid more
  specific selection. Enable geometry gestures and authoring overlays only
  while the Hotspots tab is active; leaving it cancels an unfinished polygon.
  Starting image processing must explicitly cancel an unfinished polygon and
  keep geometry editing disabled until the native invocation has unwound.
  Silently ignore draft vertex clicks that overlap an existing vertex, except
  that clicking the first vertex closes a draft once it has at least three
  vertices.
- Check the selected local MFLUX model on entry to Author mode, with concurrent
  checks deduplicated. Entering Run mode must not start AI work and must
  suppress pending AI results. Enter Run on the current Author card, keep the
  configured start card as Restart's target, and do not show a redundant
  Run-entry notification. Use one action-oriented mode button labeled Run in
  Author mode and Author in Run mode. Show standard-size Back, Restart, and
  overlay controls only in Run mode; hide the card name, version authoring
  header there.
- Keep machine-local model selection under Settings → Models, with Image and
  Sound selectors persisted through Qt settings. Offer FLUX.2 Klein 4B and
  FLUX.2 Klein 9B KV for Image and Stable Audio 3 Small-SFX for Sound. Keep
  inference, quantization, seed, credentials, and model selectors out of the
  status bar. Changing a model must cancel work using the previous setting.
- Represent a revision's applied hotspot set as `HotspotSet | None`.
  `None` means no set has been applied; an empty `HotspotSet` means an applied
  set currently contains no interactions.
- Use discriminated resolved/unresolved types for persisted references, not a
  nullable UUID plus status boolean. Store resolved runtime targets by UUID
  after selection. Go to assigns only existing catalog cards; create cards
  through the Cards sidebar. If a destination card is deleted, convert inbound
  references to unresolved while retaining the former target name; do not
  delete inbound hotspots.
- Keep an ordered, stack-owned catalog of free-form named binary Keys with
  stable UUIDs. Key names are trimmed, nonempty, and case-insensitively unique.
  Renaming preserves references; block deletion while any hotspot in any
  revision references the Key. Create Keys only through the Keys utility
  window. Keep flat, headerless condition and key-change rows in Hotspots;
  adding a row immediately selects the newest eligible catalog Key for inline
  editing. Show every reference by hotspot revision in the Keys utility window.
- Keep an ordered, stack-owned Sound catalog with stable UUIDs, trimmed
  case-insensitively unique names, editable prompts, 1–30 second durations, and
  optional immutable generated WAV assets. Generate only with Stable Audio 3
  Small-SFX through the attributed official optimized MLX subset: Pingpong,
  8 steps, CFG 1.0, random retained seed, 44.1 kHz stereo 16-bit PCM. Do not
  vendor weights or credentials. Store WAVs under deterministic
  `assets/sounds/<sound-id>/sound-<asset-id>.wav` paths, validate them as
  untrusted data, and commit each replacement with the manifest transactionally.
  Retain replaced bytes only while Undo/Redo can restore them.
- Keep one optional Sound reference per hotspot and place Play after Go to in
  Then. Assign it through one reusable movable, resizable modeless searchable
  grid picker with a Play control above every Sound name and a bottom-right
  Cancel button. Keep ungenerated Sounds selectable but disable their Play
  controls. Use a compact `−` button and the existing undoable command to clear
  an assignment. Resolve and play generated assets outside the picker, report
  failures in the global notification bar, and coordinate its preview with the
  Sounds manager's shared player. Stop picker-owned playback on cancel, close,
  selection, Run mode, project replacement, and application close without
  stopping newer playback owned by another surface. Block deletion of
  referenced Sounds and show all usages in the modeless Sounds manager. In Run,
  stop prior playback before hotspot navigation, then start the clicked Sound so
  it can continue on the destination. New playback replaces current playback;
  also stop on Back, Restart, project replacement, and Run exit. Treat
  sound-only hotspots as actionable.
- Keep hotspot state behavior closed and typed: all required Keys must be
  present, all forbidden Keys absent, and explicit Remove and Grant sets must be
  disjoint. Do not add clear-all behavior, values, counters, expressions,
  arbitrary action sequences, or scripting.
- Keep current Keys in `RunSession`, never in the authored stack or widgets.
  Enter Run empty, retain Keys through Back, clear them on Restart, and discard
  them on exit. Filter condition-failing and actionless hotspots before
  z-order hit testing, recheck conditions on activation, then execute Remove,
  Grant, and optional navigation in that order.
- Serialize MFLUX and Stable Audio loading/inference through the same stable
  process-local Metal invocation boundary. Stable Audio cancellation must be
  checked between stages and at every sampling step; ordinary tests use fakes
  and never invoke or download the live model.
- Do not add a database, web server, browser UI, plugin system,
  dependency-injection framework, event bus, arbitrary scripting engine, model
  downloader, or hosted experiment platform.

## Editing and verification

- Python 3.12.
- Prefer small, single-issue commits.
- Validate model responses strictly. No arbitrary model output may become
  executable.
- Treat paths in stack JSON as untrusted relative data. Prevent path traversal
  outside the stack directory and never allow image asset writes to overwrite
  arbitrary files.
- Keep stack-owned settings limited to portable document behavior. Store
  machine-local model/service settings through Qt settings and evaluation
  settings in versioned case files or explicit CLI arguments.
- Keep Hugging Face authentication external. Never store model tokens or other
  credentials in stack bundles, repository files, or Qt settings.
- Track only project-generated or permissively licensed evaluation inputs, with
  source and reuse provenance recorded alongside each case.
- If a change alters the concise overview, supported behavior, setup, or
  architecture summary, update `README.md` in the same change.
- If a change alters durable coding workflow or repository constraints, update
  this file in the same change.
- Track progress through issue state and the milestone. Do not add a maintained
  `PLAN.md`, `ROADMAP.md`, product spec, or duplicated checklist document.
