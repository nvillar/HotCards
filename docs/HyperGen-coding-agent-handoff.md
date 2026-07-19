# HyperGen: coding-agent handoff prompt

You are taking over the planning of **HyperGen**, an experimental local-first generative authoring application inspired by classic HyperCard. Read this entire brief before proposing architecture or implementation work.

Your immediate task is **not to start coding**. First inspect the repository and development environment, reconcile the brief with anything already present, identify only genuinely blocking ambiguities, and produce a concrete, phased implementation plan. Do not reopen settled product decisions merely because another design is possible. Do call out technical contradictions, hidden complexity, or risks, and recommend the smallest approach that preserves the intended experience.

The repository may not yet contain an `AGENTS.md` or `README.md`. Their absence is not a blocker. Do not generate generic versions before understanding this brief. If absent, include concise, project-specific drafts in the proposed bootstrap phase, following the source-of-truth policy below.

## 1. Product definition

HyperGen is an AI-assisted authoring tool for **illustrated, spatially interactive stacks of cards**. It is not yet intended to be a general-purpose HyperCard replacement.

Each stack contains cards. A card has:

- A stable internal ID and a unique human-readable name.
- A natural-language scene description.
- A natural-language interaction description.
- An optional card-specific style addition.
- One or more accepted background-image revisions.
- A selected active background revision.
- A hotspot set associated specifically with each image revision.

A hotspot combines:

- A human-facing label.
- One typed action; the first version supports only navigation to another card.
- One or more polygon components defining clickable image regions.
- An explicit ordering used for rendering and overlap hit testing.

Authors create cards from natural language, generate background images locally, generate candidate hotspots with a local vision-language model, edit the resulting polygons and destinations, and then run the stack as an interactive experience.

The first implementation is a **developer POC**, primarily for the project owner on an Apple Silicon Mac. It should be visually restrained and pleasant, but it does not need end-user installation, first-run model downloads, or distributable `.app` packaging yet. It should run through `uv`.

## 2. Fixed technology decisions

- Python 3.12.
- `uv` for project creation, virtual environments, dependency locking, commands, and development workflow. Do not use ad hoc `pip` instructions.
- PySide6 for the desktop UI.
- MFLUX through its Python API for local image generation.
- Initial image model: **FLUX.2 Klein 4B**.
- Ollama as the separately running local model runtime.
- Use the official `ollama` Python client, not hand-written HTTP requests.
- Initial Ollama candidate: **`qwen3.5:9b`**.
- Use the configured Ollama model both to derive the MFLUX render prompt and to generate hotspot proposals.
- Pydantic models for serialized boundaries and structured Ollama output.
- Human-readable JSON plus relative image assets for stack persistence.
- No database, web server, browser UI, plugin system, dependency-injection framework, event bus, or arbitrary scripting engine.

The initial development machine is an M4 Max MacBook Pro with 128 GB unified memory. The implementation should remain proportionate and should not assume that every future user has that much memory.

## Source-of-truth and repository documentation policy

Minimize the number of places that must be kept synchronized. Do not create a maintained `product-spec.md`, `PLANS.md`, roadmap document, progress checklist, or similar parallel specification.

Use these authorities:

- **Code, serialized schemas, and tests** are the source of truth for current implemented behavior and file-format invariants.
- **`README.md`** is the canonical concise project overview. Keep it accurate and brief: what HyperGen is, current maturity/status at a high level, supported platform, setup/run commands, architecture at a glance, and links to the active GitHub milestone/issues. Do not turn it into a detailed product specification or duplicate issue checklists.
- **`AGENTS.md`** contains durable coding and repository guidance only: layout, `uv` commands, test/lint/evaluation commands, architectural constraints, editing conventions, verification expectations, and rules about updating README/issues. It should reference rather than repeat README content or issue plans.
- **GitHub Issues** are the canonical source for planned implementation work, acceptance criteria, dependencies, and progress. Use issue state and a GitHub milestone to represent progress rather than maintaining a second checklist in the repository.

Issues describe intended work; they are not the sole documentation of current behavior. When an issue is completed, the implementation and tests become authoritative. If completion changes the concise product overview, setup, supported behavior, or architecture summary, update `README.md` in the same change. If it changes durable coding workflow or repository constraints, update `AGENTS.md`. Do not keep closed-issue prose synchronized line-for-line with code.

For the initial POC, prefer one GitHub milestone and a set of small, independently verifiable issues. Avoid a tracking issue whose duplicated checkbox list can drift from its child issues; milestone progress is sufficient. Each implementation issue should normally contain:

- Context and intended outcome.
- In-scope and explicitly out-of-scope work.
- Acceptance criteria.
- Verification commands and manual checks.
- Dependencies on other issues when relevant.

Keep labels modest and functional. Do not introduce a GitHub Project board, elaborate label taxonomy, or automated issue workflow unless real coordination needs emerge.

During the planning turn, **propose** the milestone and issue set but do not create or modify GitHub Issues without explicit approval. If there is no GitHub remote, `gh` authentication, or permission to create issues, return issue-ready titles and bodies in the response rather than creating a local planning document.

This handoff brief is bootstrap input for planning; it is not intended to become another maintained product specification. Once its settled decisions have been represented by the initial README, AGENTS guidance, code invariants, and approved GitHub issues, do not require ongoing synchronization of this handoff file. It may remain outside the repository or be archived/removed after bootstrap.

## 3. Product scope and explicit exclusions

The first implementation must support:

- Creating, renaming, reordering, and deleting cards.
- Selecting a stack start card.
- Editing the scene, interaction, and card-style descriptions.
- Generating a background-image candidate.
- Importing an existing background image as an image revision.
- Applying or discarding background candidates.
- Retaining every applied image revision until explicitly deleted.
- Returning to an earlier image revision and restoring its associated hotspot set.
- Generating a candidate hotspot set from the active image and interaction description.
- Reviewing and editing generated polygons and card destinations before applying them.
- Drawing, editing, grouping, ordering, and deleting hotspots manually.
- Author and Run modes.
- Navigation history with Back and Restart player controls.
- Configurable Run-mode hotspot visibility.
- Autosaving the stack atomically.
- Opening and running existing stacks when AI services are unavailable.
- A retained model-evaluation harness described later in this brief.

Explicitly out of scope for the first implementation:

- Arbitrary user scripts or LLM-generated executable code.
- Variables, conditions, timers, animation, audio, external URLs, show/hide actions, or other action types in the UI.
- A code sandbox.
- Image inpainting, outpainting, reference-image character consistency, or general image editing.
- Polygon holes.
- Cross-stack links.
- Collaboration, concurrent editing, cloud sync, accounts, telemetry, or hosted services.
- SQLite or another database.
- Persistent undo history.
- General-purpose model installation or download management.
- A packaged macOS application.
- A generic ML experiment-management platform.

The stored action representation should be typed and forward-compatible, but only card navigation is exposed and executed initially.

## 4. Core authoring interaction

Creating a card immediately creates an empty card; do not use a large modal wizard. The card inspector exposes:

- Unique name.
- Scene description.
- Interaction description.
- Optional card-specific style.

The intended workflow is deliberately two-stage:

1. Define the card's source fields.
2. Generate or import a background candidate.
3. Inspect it and Apply or Discard it.
4. Generate hotspots against the accepted active image, or draw them manually.
5. Inspect destinations and geometry, edit as needed, then Apply or Discard the hotspot candidate.
6. Enter Run mode at any time, even if the stack is incomplete.

Do not initially offer a one-click operation that chains image and hotspot generation. A poor image should not automatically cause a hotspot inference job.

### Background generation

Interactions must inform image generation, but raw navigation prose must not be passed directly to MFLUX.

The configured Ollama model should receive the scene description, interaction description, stack art direction, and card-specific style. It should extract visually relevant interactive subjects and construct a derived render prompt that:

- Preserves the intended scene.
- Applies the stack art direction.
- Applies the card-specific style addition.
- Requests that interactive subjects be clearly visible and spatially distinct.
- Avoids written labels and interface-like elements unless explicitly requested by the scene.

The user clicks Generate once. Prompt derivation and MFLUX generation proceed automatically. The derived prompt is stored with and inspectable from the candidate/revision metadata, but is not a separately editable authoring field. If it is wrong, the author edits the source descriptions and regenerates.

Background generation creates a transient candidate. Applying it creates an immutable image revision and makes it active. Discarding it removes the candidate. Do not overwrite an accepted image.

Every generated image revision records at least:

- Original author-controlled inputs used for the request.
- Derived render prompt.
- Model identifier.
- MFLUX and relevant dependency versions.
- Seed.
- Width and height.
- Step count.
- Quantization and other effective inference settings.
- Generation timestamp and duration.

The stack has a fixed logical canvas size, configurable per stack and defaulting to **1024 × 768**. Every card in the stack uses that size and aspect ratio.

### Imported images

Authors may import an image as a normal image revision. Copy the image into the stack; never retain an absolute dependency on its original location.

If its aspect ratio does not match the stack, show a simple crop-to-fill preview that lets the user reposition the crop. Store a correctly sized copy. Do not build a general image editor. Imported revisions should record their imported origin and source filename where useful, while generated-only metadata remains optional.

### Image revisions and hotspot association

Every hotspot set is tied to exactly one image revision. A hotspot set must never silently be reused against a different image.

Applying a new image revision initially gives that revision no applied hotspot set. The author may generate a new set or draw one manually. Returning to an older image revision restores that revision's own applied hotspots.

Retain all applied image revisions until the author explicitly deletes them. Deleting a revision deletes its associated hotspot set after confirmation. Deleting the active revision requires selecting another revision or leaving the card without a background.

## 5. Hotspot generation and acceptance

AI-created hotspot workflow state must be minimal.

- Generation produces a transient, editable candidate.
- The user explicitly chooses Apply or Discard.
- Only an applied hotspot set is stored or used by Run mode.
- Do **not** persist workflow flags such as `unreviewed`, `accepted`, `manually_edited`, or `stale`.
- Applied presence is acceptance.
- Manual and generated hotspots behave identically after application.
- Generation provenance may be stored for reproducibility, but is not a user-visible status system.

If the active image already has hotspots, the existing set stays active while the candidate is reviewed. Applying the candidate replaces the complete existing hotspot set as one undoable command. Label-based merging is explicitly out of scope. The Apply button should clearly communicate `Replace Hotspots` when replacement will occur.

The hotspot-generation pipeline should accept a UI-independent request object containing:

- Image input.
- Interaction description.
- Current card catalogue.
- Coordinate extent, initially 0–1000 for model output.
- Prompt/schema version and model settings.

The result should contain:

- Proposed interactions.
- Proposed polygon components.
- Proposed destination resolution.
- Validation warnings.
- Reproducibility and timing metadata.
- Access to the raw model response in evaluation/debugging contexts.

The model's integer coordinates are converted to normalized document coordinates in `[0.0, 1.0]`. Clamp and validate them before they enter the document.

Do not expose model self-reported confidence as a trustworthy quality measure. Human review is mandatory by interaction design.

## 6. Card-reference resolution

UUIDs are completely invisible to authors.

Each card has:

- A stable UUID used by runtime links.
- A unique, case-insensitively enforced human-readable name.
- Its scene description, which may assist semantic matching.

Authors reference destinations in ordinary natural language, using names or descriptions. No explicit `@mention`, bracket syntax, or UUID syntax is required in the first version.

For each hotspot-generation request, construct a request-local card catalogue containing short opaque tokens such as `C1`, plus card names and descriptions. Ask the model to select a supplied token, propose a new card name, or report an unresolved reference. Do not ask the model to reproduce UUIDs.

Conceptual target variants are:

- `existing`: a request-local token selected from the supplied catalogue.
- `new`: no existing card fits and the model proposes a human-readable destination name.
- `unresolved`: the destination is ambiguous.

Map an accepted existing token to its stable UUID. Exact normalized name matching may be deterministic. Semantic model selection is safe because it remains visible in the candidate review and is not stored until Apply.

Candidate review must allow the author to:

- Confirm the proposed existing destination.
- Choose a different existing card.
- Create a blank card using a proposed name.
- Leave the destination unresolved.

Do not automatically create cards from model output. A one-click explicit creation action is required. Creating a proposed destination creates only a blank named card; do not speculatively generate its scene or image.

Once applied, runtime uses the target UUID and never silently re-resolves the link because names or descriptions change. If a destination card is deleted, convert inbound actions to unresolved references while retaining the former target name. Do not delete inbound hotspots.

The natural-language interaction description is authoring input, not a continuously synchronized program. Editing it does not mutate applied hotspots. Pressing Generate Hotspots creates a new candidate. Manual hotspot changes do not rewrite the description.

## 7. Polygon geometry and editing

An interaction supports one or more simple polygon components. All components share its label, action, and ordering. Polygon holes are not supported initially.

The authoring canvas should support at least:

- Click to add vertices.
- Double-click or Return to close a polygon.
- Escape to cancel drawing.
- Drag a vertex.
- Insert a vertex on an edge.
- Delete a vertex, polygon component, or interaction.
- Drag a complete polygon component.
- Add another component to an existing interaction.
- Change label and destination.
- Reorder interactions.
- Undo and redo geometry changes.
- Fit-to-window, zoom, and pan appropriate to a graphics editor.

Reject or clearly flag self-intersecting polygons, polygons with fewer than three distinct points, near-zero-area polygons, and coordinates outside the canvas.

Hotspot list order determines both painting and hit testing. Later/topmost interactions win where regions overlap. Author mode should make overlaps discoverable rather than leaving ambiguous runtime behavior.

Manual hotspot creation is direct author intent and does not use a separate Apply step. Drawing a polygon creates an interaction whose label and destination can then be set. It does not need to update the natural-language description.

## 8. Author and Run UI

Use a restrained three-pane desktop layout:

- Left: stack/card sidebar with thumbnails, card order, add-card affordance, and start-card indication.
- Centre: the card canvas.
- Right: an inspector.
- Top: compact toolbar containing Author/Run selection, relevant generation actions, overlay display, and player navigation where appropriate.

The inspector should use progressive disclosure rather than presenting all settings at once. Suggested sections:

1. Card: name, scene, interactions, and card style.
2. Background: active revision, Generate/Import, revision history, and disclosed metadata.
3. Hotspots: interaction list, selected interaction properties, component management, ordering, and Generate Hotspots.

Author mode always displays hotspot geometry with restrained translucent fills, selected outlines, and vertex handles.

Run mode removes editing chrome, preserves the card aspect ratio, and fits the canvas into the window. It maintains navigation history and provides Back and Restart as player controls, not stored card actions.

Run-mode hotspot display is a stack setting with:

- Hidden — the default; pointer feedback still indicates a clickable region.
- Highlight on hover.
- Always visible.

Entering Run mode with missing images, unresolved links, or other incomplete content is allowed. Show useful warnings, but do not block incremental preview. Silently ignoring problems is not acceptable.

## 9. Stack persistence and lifecycle

Target stacks of up to approximately **100 cards**. Keep the entire JSON document in memory; load image assets lazily as appropriate.

Store a stack as a human-readable directory bundle, for example:

```text
Castle.hypergen/
├── stack.json
└── assets/
    └── cards/
        └── <card-uuid>/
            ├── image-<revision-uuid>.png
            └── ...
```

Use relative asset paths. Include a schema version from the beginning. Prefer a data model that can be migrated deliberately rather than relying on accidental Pydantic compatibility.

Autosave accepted document changes. Debounce text changes and write `stack.json` atomically: serialize and validate, write a temporary file, flush it, and replace the active JSON. Ensure an asset is fully written before JSON begins referencing it.

Transient image and hotspot candidates are not part of the stack until applied. Temporary image candidates may use an application temporary directory.

Undo/redo history is limited to the current application session. It should cover text edits at sensible boundaries, card creation/deletion/reordering, revision activation, hotspot replacement, polygon changes, destinations, and ordering. Do not implement a persistent command log.

Deleting a card requires confirmation, preserves inbound interactions as unresolved references, and handles the start-card case explicitly.

A completed stack must open, support manual authoring, and run without Ollama, MFLUX, or network access. Missing AI services disable generation actions with useful diagnostics; they do not disable the document or player.

## 10. Model services and background execution

### Ollama

Use `ollama.Client` from the official Python package. Ollama remains a separately managed local daemon. The application should not reimplement HTTP, pull models automatically, or embed an Ollama runtime.

Run synchronous Ollama client calls in a dedicated Qt worker. The Advanced Settings dialog initially exposes at least:

- Ollama endpoint.
- Ollama model tag, provisionally `qwen3.5:9b`.
- Relevant thinking/temperature options if retained after evaluation.

Use client diagnostics such as model listing/showing to give actionable availability errors. Do not repeatedly interrupt the user when a service is unavailable.

### MFLUX

Use the MFLUX Python API in-process behind a narrow adapter. Load the selected model lazily and cache it. Serialize image-generation work through a dedicated worker thread so the Qt UI remains responsive.

Do not start with a persistent subprocess or invoke the CLI per generation. Keep the adapter boundary narrow enough that process isolation could be introduced later if in-process execution proves unreliable.

Do not promise true inference cancellation unless MFLUX supports it safely. It is acceptable for a cancelled UI operation to mark a result as unwanted and discard it when computation completes.

Advanced Settings initially exposes:

- MFLUX model, default FLUX.2 Klein 4B.
- Step count, initially four.
- Quantization and seed behavior as established through evaluation.

Effective settings are always recorded per accepted generated artifact.

## 11. Model Evaluation Harness

Retain a developer-facing **HyperGen Model Evaluation Harness**. Do not call it a spike in code or documentation.

Suggested surface:

- Repository area: `evals/` for cases and documentation.
- CLI: `hypergen-eval`.
- Output: immutable run directories plus generated HTML reports.

The harness must reuse production prompt builders, Pydantic schemas, Ollama adapter, MFLUX adapter, and geometry validation. It must not contain a second implementation of generation behavior.

Support three separate suites:

### Hotspot evaluation

Run Ollama hotspot generation against frozen, version-controlled images so model and prompt comparisons are not confounded by changing image generation.

Capture:

- Structured-output validity.
- Destination resolution.
- Missing or invented hotspots.
- Single and multi-component polygons.
- Human edit-cost score.
- Cold and warm latency.
- Raw response and annotated prediction images.

The initial candidate is `qwen3.5:9b`. Keep the model configurable so 4B, 35B, future models, and prompt variants can be compared later. Choose the smallest model that produces reliably useful editable results; do not hard-code a quality assumption into the architecture.

Use the harness to settle the exact production contract, including:

- Original image versus a labelled grid.
- Grid density and styling if a grid helps.
- Thinking configuration.
- Shared prompt wording.
- Structured schema.
- Request-local card-token behavior.
- Polygon-complexity limits.
- Geometry cleanup.
- Typical latency.

### Image-generation evaluation

Run fixed scene, interaction, style, seed, and model configurations. Cases should identify required visual elements and unwanted artifacts.

Evaluate:

- Scene fidelity.
- Presence and visibility of interactive subjects.
- Suitability of those subjects as hotspots.
- Composition/readability.
- Stack-style consistency.
- Unwanted text or UI-like artifacts.
- Cold/warm latency and effective configuration.

### End-to-end evaluation

Generate a background and then generate hotspots from it. Use this to assess the complete authoring value proposition while retaining the isolated suites for diagnosis.

### Evaluation method

Use a preserved human rubric as the primary assessment. Model-assisted observations may flag missing subjects or artifacts, but must not become the final quality score—especially when the same vision model is evaluating images it later grounds.

Do not add a database, distributed runner, hosted dashboard, model downloader, or experiment server. Versioned cases, immutable run manifests, JSON/CSV summaries, annotated images, contact sheets, and readable HTML reports are sufficient.

Each run manifest should capture:

- Git commit.
- Hardware and OS.
- Python/dependency versions.
- Ollama and MFLUX versions.
- Exact model identifiers.
- Prompt and schema versions or hashes.
- Seeds and effective inference settings.
- Per-stage timings.
- Raw outputs and validation warnings.

## 12. Implementation architecture guidance

Prefer a compact architecture with clear responsibility boundaries, for example:

```text
src/hypergen/
├── main.py
├── domain/
│   ├── models.py
│   ├── geometry.py
│   └── validation.py
├── storage/
│   └── stack_store.py
├── generation/
│   ├── image_prompts.py
│   ├── mflux_generator.py
│   ├── hotspot_prompts.py
│   └── ollama_hotspots.py
├── application/
│   ├── document_controller.py
│   ├── commands.py
│   └── workers.py
├── evaluation/
│   ├── cli.py
│   ├── runners.py
│   └── reports.py
└── ui/
    ├── main_window.py
    ├── card_canvas.py
    ├── card_sidebar.py
    ├── inspector.py
    └── settings_dialog.py
```

This layout is illustrative. Adjust it if the repository or a simpler cohesive design warrants it. Preserve these principles:

- One authoritative in-memory stack document.
- UI widgets render state and issue commands; they do not own independent domain truth.
- Pydantic validates serialized and model-response boundaries.
- Image actions, model actions, and storage are not implemented inside widgets.
- Production adapters are reused by evaluation tooling.
- Keep model-specific prompt/schema logic versioned and isolated.
- Avoid speculative abstractions.

The distinction between interaction semantics and polygon geometry must remain clear even if they are stored together within an image revision's hotspot set. Runtime actions must not be encoded implicitly in labels or geometry. An older image revision must retain enough data to restore its complete working hotspot set.

## 13. Quality, testing, and safety expectations

Use focused automated tests for deterministic behavior:

- Stack serialization round trips.
- Schema-version handling and migrations.
- Atomic persistence behavior where practical.
- Unique-name validation.
- Card deletion and inbound-reference conversion.
- Coordinate conversion and clamping.
- Polygon validity and hit-order rules.
- Candidate Apply/Discard behavior.
- Image-revision and hotspot association.
- Prompt construction from fixed fixtures.
- Ollama response parsing using recorded responses, without requiring a live model.
- MFLUX adapter behavior behind fakes, without generating images in ordinary unit tests.
- Evaluation manifest reproducibility.

Keep real model calls out of the ordinary automated test suite. They belong in `hypergen-eval` and explicit integration/smoke commands.

Use `ruff` and `pytest` as development tools unless repository constraints indicate otherwise. Consider a small number of Qt tests where they materially protect command/controller behavior; do not build a brittle pixel-level GUI suite.

No arbitrary model output may become executable. Validate model responses strictly. Reject unknown action types. Treat paths in stack JSON as untrusted relative data: prevent traversal outside the stack directory. Do not allow image imports to overwrite arbitrary files.

## 14. POC completion criteria

The implementation can be considered a successful POC when a developer can:

1. Create a new stack with art direction and a 1024×768 canvas.
2. Create at least three cards, including forward/unresolved references.
3. Generate or import backgrounds and manage accepted revisions.
4. Generate hotspot candidates with `qwen3.5:9b` using the evaluated production contract.
5. Review card destinations, create a missing destination card explicitly, and edit polygons.
6. Draw a multi-component hotspot manually.
7. Apply hotspots, autosave, close, reopen, and retain correct links and geometry.
8. Navigate the stack in Run mode, including Back and Restart.
9. Switch among hidden, hover, and visible Run overlays.
10. Delete a destination card and see inbound links become unresolved rather than disappear.
11. Reopen and run the stack with Ollama unavailable.
12. Run hotspot, image, and end-to-end evaluation suites and inspect an HTML report.

Visual polish matters, but prioritize a coherent, reliable authoring loop over custom theming. Use system typography and native-feeling controls; avoid adding a third-party theme package merely to make Qt look novel.

## 15. Your required response now

Do not implement yet. After inspecting the repository and any applicable `AGENTS.md` files, if present:

1. Summarize your understanding of the product in a concise paragraph.
2. Report the relevant current repository state, including existing files, dependencies, tests, and any conflicting decisions.
   - If no `README.md` or `AGENTS.md` exists, say so and propose their essential, non-overlapping contents as part of the bootstrap plan; do not treat this as an error.
3. Identify technical risks or contradictions in this brief. Distinguish blockers from choices that can safely be resolved during implementation.
4. Propose the minimum domain model that satisfies image revision, hotspot, and reference invariants without unnecessary state tracking.
5. Propose a phased implementation plan with small, verifiable milestones. The first functional milestone must recreate the model evaluation harness using production adapters before building the complete UI. Structure the proposed plan so it can become a GitHub milestone and a set of implementation issues rather than a checked-in plan document.
6. For every phase, specify:
   - Intended outcome.
   - Files/modules likely affected.
   - Tests and evaluation commands.
   - Manual verification.
   - Exit criteria.
7. Identify the critical vertical slice and the earliest point at which the complete card-authoring loop can be exercised.
8. Provide the exact `uv`-based setup, run, lint, test, and evaluation commands you expect the repository to expose.
9. Recommend how to divide work into reviewable commits.
10. Provide a proposed GitHub milestone name and an ordered set of issue-ready titles. For each issue, include outcome, scope, acceptance criteria, verification, and dependencies. Do not create the issues until explicitly authorized.
11. Draft the minimal contents of `README.md` and `AGENTS.md`, clearly avoiding duplicated product or planning material.
12. Ask only questions whose answers would materially change the plan and cannot be safely inferred from this brief or the repository.

When proposing the plan, preserve established scope. If you recommend a change, explain the concrete failure mode it prevents and the complexity it adds. Do not expand HyperGen into a general HyperCard clone, media engine, model manager, or experiment platform.
