"""Command-line entry point for model evaluation."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from hotcards.evaluation.flux_references import (
    default_reference_output_dir,
    run_flux_reference_evaluation,
)
from hotcards.evaluation.image_prompts import (
    DEFAULT_IMAGE_PROMPT_BENCHMARK,
    ImagePromptBenchmarkSettings,
    default_image_prompt_benchmark_output_dir,
    load_image_prompt_benchmark,
    run_image_prompt_benchmark,
)
from hotcards.evaluation.images import (
    DEFAULT_MFLUX_MODELS,
    ImageEvaluationSettings,
    default_image_output_dir,
    run_image_evaluation,
)
from hotcards.evaluation.inline_references import (
    DEFAULT_INLINE_REFERENCE_EXPERIMENT,
    DEFAULT_INLINE_REFERENCE_SEEDS,
    InlineReferenceSettings,
    default_inline_reference_output_dir,
    load_inline_reference_experiment,
    run_inline_reference_evaluation,
)
from hotcards.evaluation.reports import ReportRenderingError
from hotcards.evaluation.smoke import (
    SmokeSettings,
    default_smoke_output_dir,
    run_smoke,
)
from hotcards.evaluation.style_presets import (
    DEFAULT_STYLE_PRESET_EXPERIMENT,
    DEFAULT_STYLE_PRESET_SEEDS,
    StylePresetSettings,
    default_style_preset_output_dir,
    load_style_preset_experiment,
    run_style_preset_evaluation,
)
from hotcards.evaluation.two_stage_image_prompts import (
    EVIDENCE_GATE_IMAGE_PROMPT_CANDIDATE_ID,
    EVIDENCE_GATE_IMAGE_PROMPT_VERSION,
    TWO_STAGE_IMAGE_PROMPT_CANDIDATE_ID,
    TWO_STAGE_IMAGE_PROMPT_VERSION,
    run_evidence_gate_image_prompt_benchmark,
    run_two_stage_image_prompt_benchmark,
)
from hotcards.generation.errors import GenerationError
from hotcards.generation.ollama_client import DEFAULT_OLLAMA_MODEL


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation command-line parser."""
    parser = argparse.ArgumentParser(
        prog="hotcards-eval",
        description="Run HotCards local-model evaluation suites.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser(
        "smoke",
        help="exercise the production Ollama and MFLUX path",
    )
    smoke.add_argument("--output-dir", type=Path)
    smoke.add_argument("--mflux-model", default="flux2-klein-4b")
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--quantization", type=int)
    images = subparsers.add_parser(
        "images",
        help="compare MFLUX models with deterministic author-controlled prompts",
    )
    images.add_argument("--output-dir", type=Path)
    images.add_argument("--case-dir", type=Path, default=Path("evals/cases/images"))
    images.add_argument(
        "--mflux-model",
        action="append",
        dest="mflux_models",
        default=None,
    )
    images.add_argument("--seed", type=int, default=42)
    images.add_argument("--quantization", type=int)
    image_prompts = subparsers.add_parser(
        "image-prompts",
        help="run the maintained Image Prompt benchmark through production preparation",
    )
    image_prompts.add_argument("--output-dir", type=Path)
    image_prompts.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_IMAGE_PROMPT_BENCHMARK,
    )
    image_prompts.add_argument(
        "--ollama-model",
        action="append",
        dest="ollama_models",
        default=None,
    )
    image_prompts.add_argument("--endpoint", default="http://localhost:11434")
    image_prompts.add_argument(
        "--repetitions",
        type=_positive_int,
        default=1,
    )
    image_prompts.add_argument(
        "--validate-only",
        action="store_true",
        help="validate benchmark structure, assets, and checksums without model calls",
    )
    two_stage_prompts = subparsers.add_parser(
        "image-prompts-two-stage",
        help="run the evaluation-only two-stage Reference-account candidate",
    )
    two_stage_prompts.add_argument("--output-dir", type=Path)
    two_stage_prompts.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_IMAGE_PROMPT_BENCHMARK,
    )
    two_stage_prompts.add_argument(
        "--ollama-model",
        action="append",
        dest="ollama_models",
        default=None,
    )
    two_stage_prompts.add_argument(
        "--endpoint",
        default="http://localhost:11434",
    )
    two_stage_prompts.add_argument(
        "--repetitions",
        type=_positive_int,
        default=1,
    )
    two_stage_prompts.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the frozen benchmark without model calls",
    )
    evidence_gate_prompts = subparsers.add_parser(
        "image-prompts-evidence-gate",
        help="run the target-conditioned Reference evidence-gate candidate",
    )
    evidence_gate_prompts.add_argument("--output-dir", type=Path)
    evidence_gate_prompts.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_IMAGE_PROMPT_BENCHMARK,
    )
    evidence_gate_prompts.add_argument(
        "--ollama-model",
        action="append",
        dest="ollama_models",
        default=None,
    )
    evidence_gate_prompts.add_argument(
        "--endpoint",
        default="http://localhost:11434",
    )
    evidence_gate_prompts.add_argument(
        "--repetitions",
        type=_positive_int,
        default=1,
    )
    evidence_gate_prompts.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the frozen benchmark without model calls",
    )
    references = subparsers.add_parser(
        "flux-references",
        help="measure FLUX.2 Klein multi-reference identity and style behavior",
    )
    references.add_argument("--output-dir", type=Path)
    references.add_argument("--stack", type=Path, required=True)
    references.add_argument(
        "--graphic-style-image",
        type=Path,
        default=Path("evals/cases/references/workshop.png"),
    )
    references.add_argument("--mflux-model", default="flux2-klein-4b")
    references.add_argument("--seed", type=int, default=42)
    references.add_argument("--quantization", type=int)
    references.add_argument("--width", type=_positive_int, default=1024)
    references.add_argument("--height", type=_positive_int, default=768)
    references.add_argument("--steps", type=_positive_int, default=4)
    references.add_argument(
        "--kv-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    inline_references = subparsers.add_parser(
        "inline-references",
        help="compare standalone and inline Reference language through MFLUX",
    )
    inline_references.add_argument("--output-dir", type=Path)
    inline_references.add_argument(
        "--experiment",
        type=Path,
        default=DEFAULT_INLINE_REFERENCE_EXPERIMENT,
    )
    inline_references.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_IMAGE_PROMPT_BENCHMARK,
    )
    inline_references.add_argument(
        "--mflux-model",
        default="flux2-klein-9b-kv",
    )
    inline_references.add_argument(
        "--seed",
        action="append",
        type=int,
        dest="seeds",
        default=None,
    )
    inline_references.add_argument("--quantization", type=int)
    inline_references.add_argument("--width", type=_positive_int, default=1024)
    inline_references.add_argument("--height", type=_positive_int, default=768)
    inline_references.add_argument("--steps", type=_positive_int, default=4)
    inline_references.add_argument("--blinding-seed", type=int)
    inline_references.add_argument(
        "--validate-only",
        action="store_true",
        help="validate experiment, benchmark, and frozen assets without model calls",
    )
    style_presets = subparsers.add_parser(
        "style-presets",
        help="screen proposed deterministic Style suffixes through MFLUX",
    )
    style_presets.add_argument("--output-dir", type=Path)
    style_presets.add_argument(
        "--experiment",
        type=Path,
        default=DEFAULT_STYLE_PRESET_EXPERIMENT,
    )
    style_presets.add_argument(
        "--mflux-model",
        default="flux2-klein-9b",
    )
    style_presets.add_argument(
        "--seed",
        action="append",
        type=int,
        dest="seeds",
        default=None,
    )
    style_presets.add_argument("--quantization", type=int)
    style_presets.add_argument("--width", type=_positive_int, default=1024)
    style_presets.add_argument("--height", type=_positive_int, default=768)
    style_presets.add_argument("--steps", type=_positive_int, default=4)
    style_presets.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the tracked Style matrix without model calls",
    )
    return parser


def run_cli(arguments: Sequence[str] | None = None) -> int:
    """Run the evaluation command-line interface."""
    args = build_parser().parse_args(arguments)
    try:
        if args.command == "smoke":
            result_path = run_smoke(
                SmokeSettings(
                    output_dir=args.output_dir or default_smoke_output_dir(),
                    mflux_model=args.mflux_model,
                    seed=args.seed,
                    quantization=args.quantization,
                )
            )
        elif args.command == "images":
            result_path = run_image_evaluation(
                ImageEvaluationSettings(
                    output_dir=args.output_dir or default_image_output_dir(),
                    case_dir=args.case_dir,
                    mflux_models=tuple(args.mflux_models or DEFAULT_MFLUX_MODELS),
                    seed=args.seed,
                    quantization=args.quantization,
                )
            )
        elif args.command == "image-prompts":
            if args.validate_only:
                load_image_prompt_benchmark(args.benchmark)
                result_path = args.benchmark
            else:
                result_path = run_image_prompt_benchmark(
                    ImagePromptBenchmarkSettings(
                        output_dir=(args.output_dir or default_image_prompt_benchmark_output_dir()),
                        benchmark_path=args.benchmark,
                        ollama_models=tuple(args.ollama_models or (DEFAULT_OLLAMA_MODEL,)),
                        endpoint=args.endpoint,
                        repetitions=args.repetitions,
                    )
                )
        elif args.command == "image-prompts-two-stage":
            if args.validate_only:
                load_image_prompt_benchmark(args.benchmark)
                result_path = args.benchmark
            else:
                result_path = run_two_stage_image_prompt_benchmark(
                    ImagePromptBenchmarkSettings(
                        output_dir=(args.output_dir or default_image_prompt_benchmark_output_dir()),
                        benchmark_path=args.benchmark,
                        ollama_models=tuple(args.ollama_models or (DEFAULT_OLLAMA_MODEL,)),
                        endpoint=args.endpoint,
                        repetitions=args.repetitions,
                        candidate_id=TWO_STAGE_IMAGE_PROMPT_CANDIDATE_ID,
                        candidate_prompt_version=TWO_STAGE_IMAGE_PROMPT_VERSION,
                    )
                )
        elif args.command == "image-prompts-evidence-gate":
            if args.validate_only:
                load_image_prompt_benchmark(args.benchmark)
                result_path = args.benchmark
            else:
                result_path = run_evidence_gate_image_prompt_benchmark(
                    ImagePromptBenchmarkSettings(
                        output_dir=(args.output_dir or default_image_prompt_benchmark_output_dir()),
                        benchmark_path=args.benchmark,
                        ollama_models=tuple(args.ollama_models or (DEFAULT_OLLAMA_MODEL,)),
                        endpoint=args.endpoint,
                        repetitions=args.repetitions,
                        candidate_id=EVIDENCE_GATE_IMAGE_PROMPT_CANDIDATE_ID,
                        candidate_prompt_version=EVIDENCE_GATE_IMAGE_PROMPT_VERSION,
                    )
                )
        elif args.command == "flux-references":
            result_path = run_flux_reference_evaluation(
                output_dir=args.output_dir or default_reference_output_dir(),
                stack_path=args.stack,
                graphic_style_path=args.graphic_style_image,
                model_identifier=args.mflux_model,
                quantization=args.quantization,
                seed=args.seed,
                width=args.width,
                height=args.height,
                step_count=args.steps,
                use_kv_cache=args.kv_cache,
            )
        elif args.command == "inline-references":
            if args.validate_only:
                load_inline_reference_experiment(
                    args.experiment,
                    args.benchmark,
                )
                result_path = args.experiment
            else:
                result_path = run_inline_reference_evaluation(
                    InlineReferenceSettings(
                        output_dir=(args.output_dir or default_inline_reference_output_dir()),
                        experiment_path=args.experiment,
                        benchmark_path=args.benchmark,
                        model_identifier=args.mflux_model,
                        seeds=tuple(args.seeds or DEFAULT_INLINE_REFERENCE_SEEDS),
                        width=args.width,
                        height=args.height,
                        step_count=args.steps,
                        quantization=args.quantization,
                        blinding_seed=args.blinding_seed,
                    )
                )
        elif args.command == "style-presets":
            if args.validate_only:
                load_style_preset_experiment(args.experiment)
                result_path = args.experiment
            else:
                result_path = run_style_preset_evaluation(
                    StylePresetSettings(
                        output_dir=(args.output_dir or default_style_preset_output_dir()),
                        experiment_path=args.experiment,
                        model_identifier=args.mflux_model,
                        seeds=tuple(args.seeds or DEFAULT_STYLE_PRESET_SEEDS),
                        width=args.width,
                        height=args.height,
                        step_count=args.steps,
                        quantization=args.quantization,
                    )
                )
        else:
            build_parser().error(f"unsupported command: {args.command}")
    except (GenerationError, OSError, ValueError) as error:
        print(f"hotcards-eval {args.command} failed: {error}", file=sys.stderr)
        if isinstance(error, ReportRenderingError):
            print(error.result_path)
        return 1
    print(result_path)
    return 0


def main() -> None:
    """Run the console entry point with a meaningful process status."""
    raise SystemExit(run_cli())
