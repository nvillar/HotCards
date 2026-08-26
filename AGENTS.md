# AGENTS.md

Durable coding and repository guidance for HyperGen. For what the project is
and how to run it, see `README.md`. For planned work and acceptance criteria,
see GitHub Issues and the
[**HyperGen POC** milestone](https://github.com/nvillar/HyperGen/milestone/1).
Do not duplicate that material here.

## Environment and commands

Use `uv` only; do not add ad hoc `pip` instructions.

```sh
uv sync
uv run hypergen
uv run pytest
uv run ruff check .
uv run ruff format .
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

Ordinary automated tests must not require live model calls. Use recorded
responses and fakes in `pytest`; use `hypergen-eval` or explicit smoke commands
for live Ollama and MFLUX runs.

## Repository layout

- `src/hypergen/domain/` — stack models, geometry, validation.
- `src/hypergen/storage/` — bundle persistence and migrations.
- `src/hypergen/generation/` — prompt builders and model adapters.
- `src/hypergen/application/` — document controller, commands, workers.
- `src/hypergen/ui/` — PySide6 widgets.
- `src/hypergen/evaluation/` — evaluation harness implementation.
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
- Keep the maintained Image Prompt benchmark self-contained under
  `evals/cases/image_prompts/`: freeze permitted Reference assets with checksums
  and provenance, score observable criteria rather than exact prose, and never
  read a mutable authoring stack during benchmark runs.
- Keep each card's complete authoring state in one of its numbered revisions:
  Description, selected stack Style, prepared Image Prompt with input provenance,
  optional background, optional Reference card, and hotspot set. Visible
  revision numbers are positional; stable UUIDs remain internal.
- Keep at least one revision per card. Duplicate a complete revision, including
  its hotspot semantics and immutable background reference.
- Keep at most one optional Reference card per revision and reject
  self-references. Resolve its active accepted background for both Image Prompt
  preparation and image generation. Send the Reference image once to MFLUX
  without hidden role instructions or complete source-card prose.
- Keep an ordered, stack-owned library of editable named Styles with stable
  UUIDs. Store the selected Style on each revision and persist the last explicit
  Style or No Style selection as the default for new cards. Deleting a Style
  clears its revision selections through one undoable command.
- Compose background prompts deterministically from the current reviewed Image
  Prompt followed by the selected Style text. Do not send Style to Image Prompt
  preparation. Capture the exact Style ID, name, text, and composed prompt in
  generated-image provenance; changing Style must suppress an in-flight stale
  image result without making the prepared Image Prompt stale. Hotspots must
  not alter image prompts.
- Keep one Description editor in the Background inspector. Show conditional
  native Description/Image Prompt radio controls below it, default to Image
  Prompt when it exists, place the Style selector above Reference and before
  Prepare Image Prompt and Generate, and keep generation provenance in button
  tooltips. Keep inspector tabs ordered Image, Styles, Hotspots, Keys.
  Keep Styles and Keys as stack-global list managers with compact remove/add
  controls and vertically stacked full-width fields. Encode Image
  Prompt freshness in the preparation action from the Description, exact usable
  Reference background, selected Ollama model, and preparation prompt version:
  current is a disabled completed state and stale is Update Image Prompt. Clearing the
  Image Prompt editor removes that derived value. Require a current Image Prompt
  for image generation.
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
- Image Prompt preparation requires authored Description text. Without a Reference it is
  text-only; with a readable active Reference background it is one multimodal
  Ollama request. The target Description is authoritative and the existing
  Image Prompt is never preparation input. Use the immutable effective prompt
  captured in the exact Reference background's generation metadata, including a
  reviewed Image Prompt or legacy enriched text, as the primary semantic source
  for applicable identity and style language; use the image as visual evidence
  and to fill gaps rather than relabeling explicit authored treatment. Preserve
  the meaning of every explicit target visual property without special keywords
  or required wording. Allow a plausible concrete proposal for ambiguity rather
  than adding clarification state. Store only the final Image Prompt through one
  undoable command; private deliberation must not enter the stack. Track source
  Description, exact Reference provenance, Ollama model,
  and prompt version so freshness is strict. Reject recognized authored
  object-state reversals, invented or altered affirmative quoted visible text,
  and violations of explicit quoted-text exclusions after one constrained
  repair attempt.
- Store each hotspot set under exactly one complete card revision. Replacing a
  background preserves its hotspots so the author can review and adjust them
  manually.
- Hotspots may have no polygons. Keep an optional custom Name; when absent,
  derive a short label from Remove, Grant, then destination actions and display
  long automatic labels on at most two lines. Area-less hotspots retain all
  semantics in storage and are ignored by Run-mode hit testing.
- Keep Author canvas selection hierarchical: a selected vertex belongs to a
  selected polygon, which belongs to the selected hotspot. Inspector
  synchronization and same-revision edits must not discard a valid more
  specific selection. Enable geometry gestures and authoring overlays only
  while the Hotspots tab is active; leaving it cancels an unfinished polygon.
- Suppress stale Image Prompt results after relevant target revision,
  Description, Reference assignment, source revision/background, project, or
  mode changes. Editing source text without regenerating its referenced
  background must not stale generation-time provenance.
- Check local AI services on entry to Author mode, with concurrent checks
  deduplicated. Entering Run mode must not start AI work and must suppress
  pending AI results. Enter Run on the current Author card, keep the configured
  start card as Restart's target, and do not show a redundant Run-entry
  notification. Use one action-oriented mode button labeled Run in Author mode
  and Author in Run mode. Show standard-size Back, Restart, and overlay controls
  only in Run mode; hide the card name, version authoring header, and bottom
  model selectors there.
- Keep LLM and image-model selectors in the status bar and persist them through
  Qt settings. List only installed Ollama models that advertise vision support;
  offer FLUX.2 Klein 4B and FLUX.2 Klein 9B KV. Keep model fields out of
  Advanced Settings.
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
  present, all forbidden Keys absent, Remove and Grant sets must be disjoint,
  and Clear All is exclusive. Do not add values, counters, expressions,
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
