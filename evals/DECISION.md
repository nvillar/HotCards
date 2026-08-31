# Local image-generation defaults decision

**Decision date:** 2026-08-22
**Status:** Active

## Image defaults

FLUX.2 Klein 4B remains the permissively licensed default. Controlled
Description-only runs measured approximately 11.2-11.6 seconds per render,
versus 30.7-32.1 seconds for regular 9B, without a consistent authoring-quality
gain from 9B. This is a local development tradeoff, not a broad quality-parity
claim.

The optional FLUX.2 Klein 9B KV selection is adopted for users who accept the
FLUX Non-Commercial License. A zero-reference request uses regular 9B; a
reference request uses the 9B KV edit configuration. Both repositories must be
cached.

## Prompt and Reference contract

Production sends the authored Description directly to MFLUX, followed by the
selected Style text when present. No language model prepares, enriches, or
rewrites the prompt.

Each revision accepts up to two ordered Reference cards. Their active
backgrounds are sent exactly once as `image 1` and `image 2`; authors use those
positional labels in the Description. Production adds no hidden role
instructions, aliases, or source-card prose. Generation provenance captures the
exact Description, Style snapshot, render prompt, and ordered source card,
revision, and background IDs used by the request. Results are discarded if any
captured generation input changes before completion.

## Go/no-go

- **Direct Description generation: GO**, with deterministic Style suffixing.
- **Up to two positional References: GO**, with ordered inputs, exact provenance
  snapshots, and stale-result suppression.
- **Hidden role prompting and card-name aliases: NO-GO**.
- **Novel-view identity claims: NO-GO** because local evidence was weak.
- **9B KV production smoke: GO** for regular-9B zero-reference generation and
  reference-backed KV edit generation.
