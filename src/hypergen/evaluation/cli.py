"""Command-line entry point for model evaluation."""

import argparse
import sys
from collections.abc import Sequence
from math import isfinite
from pathlib import Path

from hypergen.evaluation.images import (
    DEFAULT_MFLUX_MODELS,
    DEFAULT_OLLAMA_MODELS,
    ImageEvaluationSettings,
    default_image_output_dir,
    run_image_evaluation,
)
from hypergen.evaluation.smoke import (
    SmokeSettings,
    default_smoke_fixture,
    default_smoke_output_dir,
    run_smoke,
)
from hypergen.generation.errors import GenerationError


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
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
    smoke.add_argument("--fixture-image", type=Path, default=default_smoke_fixture())
    smoke.add_argument("--ollama-endpoint", default="http://localhost:11434")
    smoke.add_argument("--ollama-model", default="qwen3.5:9b")
    smoke.add_argument("--mflux-model", default="flux2-klein-4b")
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--quantization", type=int)
    smoke.add_argument("--ollama-timeout", type=_positive_float, default=300.0)
    smoke.add_argument("--ollama-num-predict", type=_positive_int, default=2048)
    smoke.add_argument("--ollama-context-length", type=_positive_int, default=8192)
    images = subparsers.add_parser(
        "images",
        help="compare Ollama render prompts and MFLUX image models",
    )
    images.add_argument("--output-dir", type=Path)
    images.add_argument("--case-dir", type=Path, default=Path("evals/cases/images"))
    images.add_argument("--ollama-endpoint", default="http://localhost:11434")
    images.add_argument(
        "--ollama-model",
        action="append",
        dest="ollama_models",
        default=None,
    )
    images.add_argument(
        "--mflux-model",
        action="append",
        dest="mflux_models",
        default=None,
    )
    images.add_argument("--downstream-mflux-model", default="flux2-klein-4b")
    images.add_argument("--seed", type=int, default=42)
    images.add_argument("--quantization", type=int)
    images.add_argument("--ollama-timeout", type=_positive_float, default=300.0)
    images.add_argument("--ollama-num-predict", type=_positive_int, default=2048)
    images.add_argument("--ollama-context-length", type=_positive_int, default=8192)
    return parser


def run_cli(arguments: Sequence[str] | None = None) -> int:
    """Run the evaluation command-line interface."""
    args = build_parser().parse_args(arguments)
    try:
        if args.command == "smoke":
            result_path = run_smoke(
                SmokeSettings(
                    output_dir=args.output_dir or default_smoke_output_dir(),
                    fixture_image=args.fixture_image,
                    ollama_endpoint=args.ollama_endpoint,
                    ollama_model=args.ollama_model,
                    mflux_model=args.mflux_model,
                    seed=args.seed,
                    quantization=args.quantization,
                    ollama_timeout_seconds=args.ollama_timeout,
                    ollama_num_predict=args.ollama_num_predict,
                    ollama_context_length=args.ollama_context_length,
                )
            )
        elif args.command == "images":
            result_path = run_image_evaluation(
                ImageEvaluationSettings(
                    output_dir=args.output_dir or default_image_output_dir(),
                    case_dir=args.case_dir,
                    ollama_endpoint=args.ollama_endpoint,
                    ollama_models=tuple(args.ollama_models or DEFAULT_OLLAMA_MODELS),
                    mflux_models=tuple(args.mflux_models or DEFAULT_MFLUX_MODELS),
                    downstream_mflux_model=args.downstream_mflux_model,
                    seed=args.seed,
                    quantization=args.quantization,
                    ollama_timeout_seconds=args.ollama_timeout,
                    ollama_num_predict=args.ollama_num_predict,
                    ollama_context_length=args.ollama_context_length,
                )
            )
        else:
            build_parser().error(f"unsupported command: {args.command}")
    except (GenerationError, OSError, ValueError) as error:
        print(f"hypergen-eval {args.command} failed: {error}", file=sys.stderr)
        return 1
    print(result_path)
    return 0


def main() -> None:
    """Run the console entry point with a meaningful process status."""
    raise SystemExit(run_cli())
