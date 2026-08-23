"""Command-line entry point for model evaluation."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from hypergen.evaluation.flux_references import (
    default_reference_output_dir,
    run_flux_reference_evaluation,
)
from hypergen.evaluation.images import (
    DEFAULT_MFLUX_MODELS,
    ImageEvaluationSettings,
    default_image_output_dir,
    run_image_evaluation,
)
from hypergen.evaluation.reports import ReportRenderingError
from hypergen.evaluation.smoke import (
    SmokeSettings,
    default_smoke_output_dir,
    run_smoke,
)
from hypergen.generation.errors import GenerationError


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation command-line parser."""
    parser = argparse.ArgumentParser(
        prog="hypergen-eval",
        description="Run HyperGen local-model evaluation suites.",
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
        else:
            build_parser().error(f"unsupported command: {args.command}")
    except (GenerationError, OSError, ValueError) as error:
        print(f"hypergen-eval {args.command} failed: {error}", file=sys.stderr)
        if isinstance(error, ReportRenderingError):
            print(error.result_path)
        return 1
    print(result_path)
    return 0


def main() -> None:
    """Run the console entry point with a meaningful process status."""
    raise SystemExit(run_cli())
