"""Command-line entry point for model evaluation."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation command-line parser."""
    return argparse.ArgumentParser(
        prog="hypergen-eval",
        description="Run HyperGen local-model evaluation suites.",
    )


def main() -> None:
    """Run the evaluation command-line interface."""
    build_parser().parse_args()
