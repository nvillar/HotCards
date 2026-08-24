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
uv run hypergen-eval flux-references --stack /path/to/Stack.hypergen
```

Ordinary automated tests must not require live model calls. Use recorded
responses and fakes in `pytest`; use `hypergen-eval` or explicit smoke commands
for live Ollama and MFLUX runs.

## Repository layout

- `src/hypergen/domain/` — stack models, geometry, validation.
- `src/hypergen/storage/` — exact-current-schema bundle persistence.
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
- Keep each card's complete authoring state in one of its numbered revisions:
  authored Description, optional Enriched Description with input provenance,
  optional background, fixed card-reference roles, and hotspot set. Visible
  revision numbers are positional; stable UUIDs remain internal.
- Keep at least one revision per card. Duplicate a complete revision, including
  its hotspot semantics and immutable background reference.
- Keep exactly three optional card-reference roles: Subject, Style, and
  Setting. Each accepts at most one other card and rejects self-references, but
  one source card may fill multiple roles. Group roles by active source
  background, feed each unique image to MFLUX once in first-role order, combine
  its role instructions, and do not inject source Descriptions into the image
  prompt. Phrase FLUX reference instructions positively by stating which
  attributes come from each image and which come from the scene or other
  assigned references.
- Compose background prompts deterministically from the active revision's
  Enriched Description when present, otherwise its authored Description, plus
  fixed instructions for assigned reference roles. Send only that effective
  Description to MFLUX. Enrichment validation keeps authored actions, poses,
  object states, time, weather, camera, mood, composition, and quoted visible
  text authoritative while deriving Enriched Description. Hotspots must not
  alter image prompts.
- Keep one Description editor in the Background inspector. Show conditional
  native Original/Enriched radio controls below it, default to Enriched when it
  exists, place compact reference rows in a References group before Enrich and
  Generate, and keep generation provenance in button tooltips. Encode enrichment
  freshness in the Enrich action: current is a disabled completed state, stale
  is Re-enrich. Clearing the Enriched editor removes that derived value. Enable
  image generation when either Description source is non-empty.
- Support generated backgrounds only; do not add image import. Apply Generate
  and Clear directly through document commands. Keep an existing image visible
  until replacement succeeds, then expose a dismissible Undo bound to the exact
  current history token.
- Apply reversible deletions and replacements without confirmation. Report
  outcomes, failures, Run warnings, and Undo actions in the global notification
  bar; keep field validation beside its input and the status bar passive. Use a
  blocking decision dialog only when proceeding could lose persisted work and
  Undo cannot recover it.
- Enrich Description is text-only and requires authored Description text. It
  must not send current or referenced images to Ollama. Always derive it from
  the authored Description, never from an existing enrichment. Give it
  each referenced background's exact generation-time effective Description as
  role-scoped provenance, then store the result separately through one undoable
  command.
  Track source Description and exact usable reference-image provenance so an
  enrichment can be marked Current or Out of date without being cleared; image
  generation continues using an out-of-date enrichment until it is cleared or
  replaced. Reject reference-derived reversals of recognized authored object
  states after one constrained repair attempt.
  Assigned references replace conflicting authored details within their role
  rather than blending both versions. Require distinctive source-language
  overlap for every assigned role, including style-specific cues for Style,
  reject recognized conflicting authored styles that remain, and reject newly
  invented quoted visible text. Shape the result for FLUX.2 as one concise
  natural-language paragraph ordered by subject, action, style, context, then
  secondary details. Use positive desired-image language, associate colors and
  materials with specific objects, and add photographic camera details only
  for explicitly photographic styles.
- Store each hotspot set under exactly one complete card revision. Replacing a
  background preserves its hotspots so the author can review and adjust them
  manually.
- Hotspots may have no polygons. Derive every hotspot's label from its resolved
  destination card's current name, or `Unresolved`; do not expose separate
  label editing. Area-less hotspots retain their destination in storage and are
  ignored by Run-mode hit testing.
- Keep Author canvas selection hierarchical: a selected vertex belongs to a
  selected polygon, which belongs to the selected hotspot. Inspector
  synchronization and same-revision edits must not discard a valid more
  specific selection. Enable geometry gestures and authoring overlays only
  while the Hotspots tab is active; leaving it cancels an unfinished polygon.
- Suppress stale enrichment results after relevant target revision,
  Description, reference assignment, source revision/background, project, or
  mode changes. Editing source text without regenerating its referenced
  background must not stale generation-time provenance.
- Check local AI services on entry to Author mode, with concurrent checks
  deduplicated. Entering Run mode must not start AI work and must suppress
  pending AI results.
- Keep LLM and image-model selectors in the status bar and persist them through
  Qt settings. List installed Ollama models; offer FLUX.2 Klein 4B and FLUX.2
  Klein 9B KV. Keep model fields out of Advanced Settings.
- Represent a revision's applied hotspot set as `HotspotSet | None`.
  `None` means no set has been applied; an empty `HotspotSet` means an applied
  set currently contains no interactions.
- Use discriminated resolved/unresolved types for persisted references, not a
  nullable UUID plus status boolean. Store resolved runtime targets by UUID
  after selection. If a destination card is deleted, convert inbound references to
  unresolved while retaining the former target name; do not delete inbound
  hotspots.
- Expose and execute only the `navigate` action initially. Keep the stored action
  representation typed and forward-compatible, and reject unknown action types.
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
