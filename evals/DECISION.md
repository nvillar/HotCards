# Local model defaults decision

**Decision date:** 2026-08-22
**Status:** Active

## Ollama default: `qwen3.5:9b-mlx`

Ollama is used only for text-only Description enrichment. Enrich takes the
authored Description and role-scoped generation-time text provenance for
referenced backgrounds, applies one undoable rewrite, and never reads current
or referenced images. The 9B MLX build remains the development default because
it is the locally optimized middle resource point among the previously
exercised 4B, 9B, and 35B candidates.

Hotspot geometry is authored manually. Earlier Remap experiments are retired
and do not describe production behavior.

## Image defaults

FLUX.2 Klein 4B remains the permissively licensed default. Controlled
Description-only runs measured approximately 11.2-11.6 seconds per render,
versus 30.7-32.1 seconds for regular 9B, without a consistent authoring-quality
gain from 9B. This is a local development tradeoff, not a broad quality-parity
claim.

The optional FLUX.2 Klein 9B KV selection is adopted for users who accept the
FLUX Non-Commercial License. A zero-reference request uses regular 9B; a
reference request uses the 9B KV edit configuration. Both repositories must be
cached. Live validation on 2026-08-23 completed both production dispatch paths:
a zero-reference request used regular 9B with KV disabled, and a three-reference
Identity, Visual style, and Setting request used the 9B KV edit model with KV
enabled. Both produced readable 1024x768 RGB images without warnings.

## Card-reference contract

A focused 4B feasibility run found useful transfer for character identity,
visual style, and setting with up to three ordered reference images. A fourth
reference increased latency, memory use, role competition, and identity drift;
novel-view inference also remained weak. Production therefore exposes exactly
three optional, revision-local roles:

1. Identity
2. Visual style
3. Setting

Each role accepts at most one other card, and one card may fill several roles.
The referenced card's active background is authoritative. Assignments sharing
one background are grouped into one unique image with combined role
instructions. Missing roles are omitted and remaining images are renumbered.
Source Descriptions are not injected into the MFLUX prompt; fixed role
instructions and the target Description form that deterministic prompt.

Generation metadata captures the source card, active revision, and background
IDs used by the request. Results are discarded if a role assignment or any
captured source identity changes before completion. Text and edit models are
not retained in memory simultaneously.

## Go/no-go

- **Deterministic three-role references: GO**, with ordered inputs, exact
  provenance snapshots, and stale-result suppression.
- **Four or more references: NO-GO** because local evidence showed worse
  resource use and role competition.
- **Novel-view identity claims: NO-GO** because local evidence was weak.
- **9B KV production smoke: GO** for both regular-9B zero-reference generation
  and three-reference KV edit generation.
