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
  cards, selected Generate resolution, and hotspot set. Visible revision
  numbers are positional; stable UUIDs remain internal.
- Store one immutable stack aspect ratio using only Square 1:1, Landscape 4:3,
  Portrait 3:4, or Widescreen 16:9. Generate resolution is revision-local,
  defaults to 512, and permits only 256, 512, 768, or 1024 square-equivalent
  pixels. Derive output dimensions with REM's area-preserving square-equivalent
  formulas `width = sqrt(resolution² × ratio_w / ratio_h)` and
  `height = sqrt(resolution² × ratio_h / ratio_w)`, then round each dimension
  to the nearest multiple of 16, minimum 16. Continue storing exact actual
  output width and height in image provenance.
- Keep at least one revision per card. Duplicate a complete revision, including
  its Generate resolution, hotspot semantics, and immutable background
  reference.
- Keep an ordered collection of at most two optional Reference cards per
  revision and reject self-references and duplicates. Display slots as `1.` and
  `2.` under one References section; clearing slot 1 promotes slot 2. Resolve
  each active accepted background for image generation. Require user-facing
  guidance to use positional `image 1` and `image 2` labels; do not translate
  card names or aliases. Send each Reference exactly once to MFLUX in stable
  order without hidden role instructions or complete source-card prose.
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
- Persist generated backgrounds with a strict discriminated provenance union
  for direct Generate, externally patched legacy Generate, Refine, and Edit.
  Keep operation-specific prompts and settings typed rather than accumulating
  nullable fields. Refine and Edit identify their exact source revision and
  background. A Refine inherits its source revision's accepted Edit lineage
  unchanged; an Edit appends exactly its accepted current Edit to that source
  lineage. Current Generate, Refine, and Edit dimensions must match their typed
  resolution and the stack aspect ratio. Legacy Generate preserves its exact
  historical prompt and dimensions without imposing a modern preset. Block
  source revision deletion or background replacement while any retained
  revision derives from it. Whole card deletion may remove dependencies wholly
  contained in that card, but must reject dependencies from retained cards.
- Route all production and evaluation Generate, Refine, and Edit inference
  through one typed MFLUX adapter. Plain Generate and Refine use the regular
  family; Reference-backed Generate and Edit use the Edit family. Serialize
  model loading and inference through one process-local boundary, cache at most
  one compatible family/model/quantization configuration, and release it when
  switching configuration. Cancellation while queued or running must publish
  no output; interrupted active models must not be reused.
- Keep one Description editor in the Generate inspector. Place the Style
  selector above References, place revision-local Resolution after References
  and before Generate, show exact output dimensions and positional Reference
  guidance, and keep generation provenance in button tooltips. Keep inspector
  tabs ordered Generate, Styles, Hotspots, Keys. Keep Styles and Keys as
  stack-global list managers with compact remove/add controls and vertically
  stacked full-width fields. Require a nonempty Description for image
  generation.
- Create stacks with one native format selector containing exactly Square 1:1,
  Landscape 4:3, Portrait 3:4, and Widescreen 16:9, defaulting to Landscape.
  Do not expose arbitrary dimensions or a post-creation aspect-ratio setting.
  Fit and center every complete background inside the stack's logical card
  geometry with aspect-preserving scaling and letterboxing or pillarboxing as
  needed. Use the fitted image bounds for all normalized hotspot rendering,
  gestures, and Run hit testing; bars are noninteractive.
- Support generated backgrounds only; do not add image import. Apply Generate
  and Clear directly through document commands. Keep an existing image visible
  until replacement succeeds. After Description or image generation succeeds,
  expose Create New Version and Undo bound to the exact current history token;
  an explicit Keep action retains the result on the current revision. Creating
  a version must restore the prior revision and append one complete generated
  revision.
- Apply reversible deletions and replacements without confirmation. Report
  outcomes, failures, Run warnings, and Undo actions in the global notification
  bar; keep field validation beside its input and the status bar passive. Use a
  blocking decision dialog only when proceeding could lose persisted work and
  Undo cannot recover it.
- Store each hotspot set under exactly one complete card revision. Replacing a
  background preserves its hotspots so the author can review and adjust them
  manually.
- Hotspots may have no polygons. Derive their labels from Remove, Grant, then
  destination actions and display long labels on at most two lines. Area-less
  hotspots retain all semantics in storage and are ignored by Run-mode hit
  testing.
- Keep Author canvas selection hierarchical: a selected vertex belongs to a
  selected polygon, which belongs to the selected hotspot. Inspector
  synchronization and same-revision edits must not discard a valid more
  specific selection. Enable geometry gestures and authoring overlays only
  while the Hotspots tab is active; leaving it cancels an unfinished polygon.
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
  header, and bottom model selectors there.
- Keep the image-model selector in the status bar and persist it through Qt
  settings. Offer FLUX.2 Klein 4B and FLUX.2 Klein 9B KV. Keep the model field
  out of Advanced Settings. Disable the selector while image generation is
  running; changing the model must cancel work using the previous setting.
- Represent a revision's applied hotspot set as `HotspotSet | None`.
  `None` means no set has been applied; an empty `HotspotSet` means an applied
  set currently contains no interactions.
- Use discriminated resolved/unresolved types for persisted references, not a
  nullable UUID plus status boolean. Store resolved runtime targets by UUID
  after selection. If a destination card is deleted, convert inbound references to
  unresolved while retaining the former target name; do not delete inbound
  hotspots.
- Keep an ordered, stack-owned catalog of free-form named binary Keys with
  stable UUIDs. Key names are trimmed, nonempty, and case-insensitively unique.
  Renaming preserves references; block deletion while any hotspot in any
  revision references the Key. Keep the Keys inspector after Hotspots and show
  every reference grouped by hotspot revision and semantic role.
- Keep hotspot state behavior closed and typed: all required Keys must be
  present, all forbidden Keys absent, and explicit Remove and Grant sets must be
  disjoint. Do not add clear-all behavior, values, counters, expressions,
  arbitrary action sequences, or scripting.
- Keep current Keys in `RunSession`, never in the authored stack or widgets.
  Enter Run empty, retain Keys through Back, clear them on Restart, and discard
  them on exit. Filter condition-failing and actionless hotspots before
  z-order hit testing, recheck conditions on activation, then execute Remove,
  Grant, and optional navigation in that order.
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
