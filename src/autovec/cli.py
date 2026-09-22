"""Command-line interface.

Supports both:
    python -m autovec <file> [options]
and subcommands:
    autovec analyze|vectorize|verify|benchmark <file> --width N
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import List, Optional

from .diagnostics.reporter import format_report, run_pipeline
from .execution.vector import verify as run_verify
from .ir.parser import ParseError

COMMANDS = ("analyze", "vectorize", "verify", "benchmark")


def detect_input_type(path: str, text: str) -> str:
    if path.endswith(".json"):
        return "json"
    if path.endswith(".ir"):
        return "toy-ir"
    if text.strip().upper().startswith("LOOP"):
        return "toy-ir"
    return "c-like"


def read_source_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def _build_arg_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("file", help="Input source file (.ir toy-ir, .json, or C-like)")
    parser.add_argument("--vector-width", "--width", dest="width", type=int, default=4,
                         help="Vector width: one of 2, 4, 8, 16 (default: 4)")
    parser.add_argument("--input-type", dest="input_type", default=None,
                         choices=["c-like", "toy-ir", "json"],
                         help="Override input-format auto-detection")
    parser.add_argument("--output", "-o", dest="output", default=None,
                         help="Write the report to this file instead of stdout")
    parser.add_argument("--report", action="store_true",
                         help="Emit a structured JSON report instead of the human-readable one")
    parser.add_argument("--verify", action="store_true",
                         help="Run the scalar-vs-vector correctness verification")
    parser.add_argument("--verbose", "--debug", dest="verbose", action="store_true",
                         help="Print pipeline log lines ([Parser], [Analysis], [Legality], ...)")
    return parser


def build_json_report(result, width: int, verify_result) -> dict:
    legality = result.legality
    return {
        "vectorizable": legality.legal,
        "vectorWidth": width,
        "reasons": legality.reasons,
        "warnings": legality.warnings,
        "memoryAccesses": [asdict(m) for m in legality.memory_accesses],
        "dependencies": [asdict(d) for d in legality.dependencies],
        "transformation": {"before": result.before, "after": result.after},
        "metrics": asdict(result.metrics) if result.metrics else None,
        "verify": (
            {"status": verify_result.status, "mismatches": verify_result.mismatches}
            if verify_result is not None else None
        ),
        "log": result.log,
    }


def format_benchmark(result, width: int) -> str:
    lines = [f"Benchmark (vector width = {width})", "-" * 40]
    m = result.metrics
    if m is None:
        lines.append("Not vectorizable or metrics unavailable (symbolic bound).")
        return "\n".join(lines)
    lines.append(f"scalarInstructions   = {m.scalarInstructions}")
    lines.append(f"vectorInstructions   = {m.vectorInstructions}")
    lines.append(f"instructionReduction = {m.instructionReduction:.2f}%")
    lines.append(f"vectorIterations     = {m.vectorIterations}")
    lines.append(f"remainderIterations  = {m.remainderIterations}")
    n = (m.vectorIterations or 0) * width + (m.remainderIterations or 0)
    if n:
        ratio = ((m.vectorIterations or 0) * width) / n
        lines.append(f"vectorizationRatio   = {ratio:.4f}")
    return "\n".join(lines)


def run_command(command: str, args: argparse.Namespace) -> int:
    try:
        text = read_source_file(args.file)
    except OSError as e:
        print(f"Error: could not read {args.file!r}: {e}", file=sys.stderr)
        return 2

    input_type = args.input_type or detect_input_type(args.file, text)

    try:
        result = run_pipeline(text, input_type, args.width)
    except (ParseError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    do_verify = args.verify or command == "verify"
    verify_result = None
    if do_verify and result.legality.legal and result.vec_result is not None:
        verify_result = run_verify(result.loop, result.vec_result)
        result.log.append(f"[Verify] {verify_result.status}")

    if args.verbose:
        for line in result.log:
            print(line)
        print()

    if args.report:
        output_text = json.dumps(build_json_report(result, args.width, verify_result), indent=2)
    elif command == "benchmark":
        output_text = format_benchmark(result, args.width)
    else:
        output_text = format_report(result, args.width, verify_result)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_text + "\n")
    else:
        print(output_text)

    return 0 if result.legality.legal else 1


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)

    if argv and argv[0] in COMMANDS:
        command = argv[0]
        rest = argv[1:]
        prog = f"autovec {command}"
    else:
        command = "analyze"
        rest = argv
        prog = "autovec"

    parser = _build_arg_parser(prog)
    args = parser.parse_args(rest)
    return run_command(command, args)


if __name__ == "__main__":
    sys.exit(main())
