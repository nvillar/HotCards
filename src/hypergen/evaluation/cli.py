"""Command-line entry point for model evaluation."""

import argparse
import sys
from collections.abc import Sequence
from math import isfinite
from pathlib import Path

from hypergen.evaluation.e2e import (
    E2EEvaluationSettings,
    default_e2e_output_dir,
    run_e2e_evaluation,
)
from hypergen.evaluation.hotspots import (
    DEFAULT_OLLAMA_MODELS,
    HotspotEvaluationSettings,
    default_hotspot_output_dir,
    run_hotspot_evaluation,
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
    hotspots = subparsers.add_parser(
        "hotspots",
        help="compare structured hotspot generation across Ollama models",
    )
    hotspots.add_argument("--output-dir", type=Path)
    hotspots.add_argument("--case-dir", type=Path, default=Path("evals/cases/hotspots"))
    hotspots.add_argument("--fixture-root", type=Path, default=Path("evals/cases"))
    hotspots.add_argument("--ollama-endpoint", default="http://localhost:11434")
    hotspots.add_argument(
        "--ollama-model",
        action="append",
        dest="ollama_models",
        default=None,
    )
    hotspots.add_argument("--ollama-timeout", type=_positive_float, default=120.0)
    hotspots.add_argument("--ollama-num-predict", type=_positive_int, default=1024)
    hotspots.add_argument("--ollama-context-length", type=_positive_int, default=8192)
    hotspots.add_argument("--no-ablations", action="store_true")
    hotspots.add_argument("--ablation-model", default="qwen3.5:9b")
    e2e = subparsers.add_parser(
        "e2e",
        help="compare exact Ollama candidates through one fixed MFLUX model",
    )
    e2e.add_argument("--output-dir", type=Path)
    e2e.add_argument("--case-dir", type=Path, default=Path("evals/cases/e2e"))
    e2e.add_argument("--ollama-endpoint", default="http://localhost:11434")
    e2e.add_argument("--seed", type=int, default=42)
    e2e.add_argument("--quantization", type=int)
    e2e.add_argument("--ollama-timeout", type=_positive_float, default=120.0)
    e2e.add_argument("--ollama-num-predict", type=_positive_int, default=1024)
    e2e.add_argument("--ollama-context-length", type=_positive_int, default=8192)
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
                    mflux_models=tuple(args.mflux_models or DEFAULT_MFLUX_MODELS),
                    seed=args.seed,
                    quantization=args.quantization,
                )
            )
        elif args.command == "hotspots":
            result_path = run_hotspot_evaluation(
                HotspotEvaluationSettings(
                    output_dir=args.output_dir or default_hotspot_output_dir(),
                    case_dir=args.case_dir,
                    fixture_root=args.fixture_root,
                    ollama_endpoint=args.ollama_endpoint,
                    ollama_models=tuple(args.ollama_models or DEFAULT_OLLAMA_MODELS),
                    ollama_timeout_seconds=args.ollama_timeout,
                    ollama_num_predict=args.ollama_num_predict,
                    ollama_context_length=args.ollama_context_length,
                    include_ablations=not args.no_ablations,
                    ablation_model=args.ablation_model,
                )
            )
        elif args.command == "e2e":
            result_path = run_e2e_evaluation(
                E2EEvaluationSettings(
                    output_dir=args.output_dir or default_e2e_output_dir(),
                    case_dir=args.case_dir,
                    ollama_endpoint=args.ollama_endpoint,
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
        if isinstance(error, ReportRenderingError):
            print(error.result_path)
        return 1
    print(result_path)
    return 0


def main() -> None:
    """Run the console entry point with a meaningful process status."""
    raise SystemExit(run_cli())
