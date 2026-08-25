# Image Prompt benchmark

`benchmark.json` is the maintained HyperGen benchmark for Image Prompt
preparation. It is a self-contained dataset: cases refer only to frozen images
under `references/`, and each image has a checked SHA-256 digest and reuse
provenance. The benchmark never opens or depends on the authoring stack that
seeded its first cases.

The benchmark evaluates meaning rather than exact wording. Each case contains
observable pass/fail criteria in these categories:

- `target_fidelity` — authored subjects, state, viewpoint, composition, spatial
  attachment, and lighting.
- `continuity` — stable Reference details that should remain.
- `reference_restraint` — Reference details that must not leak into the target.
- `override` — authored state, treatment, palette, or local-color changes.
- `visible_text` — exact requested text and explicit text exclusions.
- `prompt_quality` — one standalone positive description without process or
  machine language.

`critical` criteria are hard-failure conditions. `major` and `minor` criteria
capture meaningful quality differences without redefining the target.
Score each criterion as `pass`, `fail`, or `uncertain`; treat `uncertain` as a
failure when applying a release gate. Mark the complete row as a hard failure
when generation fails or any `critical` criterion fails. Compare candidates by
hard-failure count first, then major and minor pass rates; do not average a
critical omission away with successes on cosmetic criteria.

## Validate

```sh
uv run hypergen-eval image-prompts --validate-only
```

Validation checks strict JSON structure, unique case and criterion IDs, safe
relative asset paths, complete asset references, and image checksums. It makes
no model call.

## Run the production baseline

```sh
uv run hypergen-eval image-prompts
uv run hypergen-eval image-prompts \
  --ollama-model qwen3.5:9b-mlx \
  --repetitions 3
```

Runs are immutable under `evals/runs/`. Each result includes the candidate
contract digest, every initial and repair response, aggregate complete-call
metrics, final Image Prompts, and an unscored copy of every common and
case-specific criterion. Human reviewers copy the scorecard to a separate
review artifact instead of modifying the run; tracked benchmark inputs and raw
runs remain unchanged.
Experimental candidates should call `run_image_prompt_benchmark` with their own
factory, candidate ID, prompt version, and contract digest rather than replacing
production code merely to run a comparison.

## Maintain the dataset

1. Write criteria before evaluating candidate output. Do not use an existing
   generated Image Prompt as an expected answer.
2. Make every criterion independently observable and assign `critical` only
   when failure makes the result unsafe or unusable.
3. Put reusable scene details under `continuity` and transient or unrelated
   Reference details under `reference_restraint`.
4. Add authored edge cases when they exercise a genuinely new semantic boundary;
   do not add paraphrases solely to increase case count.
5. Copy permitted frozen assets into `references/`, record provenance and reuse
   terms, and update their SHA-256 digest.
6. Keep development cases for active diagnosis, regression cases for behavior
   that already works, and edge cases for generalization beyond the seed deck.
7. Run validation and ordinary tests before committing benchmark changes.
