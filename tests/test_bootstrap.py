"""Bootstrap tests for package entry points."""

from hypergen.evaluation.cli import build_parser


def test_evaluation_parser_uses_expected_program_name() -> None:
    assert build_parser().prog == "hypergen-eval"
