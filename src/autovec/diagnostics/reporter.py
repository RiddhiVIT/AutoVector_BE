"""Pipeline orchestration + human-readable / structured reporting.

`run_pipeline()` is the single entry point used by both the CLI and the
FastAPI app: parse -> analyze memory -> check legality -> vectorize ->
compute metrics, collecting a structured `log` of `[Stage] message` lines
along the way.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..analysis.legality import LegalityResult, VectorizationLegalityChecker
from ..analysis.memory import access_ref_text
from ..execution.vector import VerifyResult
from ..ir import parser as ir_parser
from ..ir.loop import Loop
from ..ir.serializer import instruction_to_text
from ..metrics.performance import Metrics, compute_metrics
from ..vectorize.vectorizer import VALID_WIDTHS, VectorizationResult, vectorize_loop


@dataclass
class PipelineResult:
    loop: Loop
    legality: LegalityResult
    memory_accesses: list = field(default_factory=list)
    vec_result: Optional[VectorizationResult] = None
    metrics: Optional[Metrics] = None
    before: List[str] = field(default_factory=list)
    after: List[str] = field(default_factory=list)
    log: List[str] = field(default_factory=list)


def _parse(source: str, input_type: str) -> Loop:
    if input_type == "c-like":
        return ir_parser.parse_c_like(source)
    if input_type == "toy-ir":
        return ir_parser.parse_toy_ir(source)
    if input_type == "json":
        return ir_parser.parse_json(source)
    raise ir_parser.ParseError(
        f"Unknown inputType {input_type!r}. Must be one of 'c-like', 'toy-ir', 'json'."
    )


def run_pipeline(source: str, input_type: str, width: int, verbose_log: bool = True) -> PipelineResult:
    if width not in VALID_WIDTHS:
        raise ValueError(f"Invalid vector width: {width}. Must be one of {VALID_WIDTHS}.")

    log: List[str] = []
    loop = _parse(source, input_type)
    log.append("[Parser] Loop detected")
    log.append(f"[Parser] Induction variable: {loop.induction_variable}")
    log.append(f"[Parser] Parsed {len(loop.body)} instructions")

    checker = VectorizationLegalityChecker()
    legality = checker.check(loop)

    for ma in legality.memory_accesses:
        kind = "STORE" if ma.is_write else "LOAD"
        log.append(f"[Analysis] {kind} {access_ref_text(ma)}")
    if not legality.memory_accesses and not legality.legal:
        # e.g. unknown access pattern -- extraction failed before any
        # accesses could be collected.
        for r in legality.reasons:
            log.append(f"[Analysis] {r}")

    if legality.dependencies:
        for d in legality.dependencies:
            log.append(
                f"[Dependency] Loop-carried {d.type} dependency: "
                f"{d.target} depends on {d.source}"
            )
    else:
        log.append("[Dependency] No loop-carried dependencies")

    for r in legality.reasons:
        if not r.startswith("Loop-carried"):
            log.append(f"[Legality] {r}")
    log.append("[Legality] Vectorization allowed" if legality.legal else "[Legality] Vectorization rejected")

    before = [instruction_to_text(i) for i in loop.body]
    vec_result: Optional[VectorizationResult] = None
    metrics: Optional[Metrics] = None
    after: List[str] = []

    if legality.legal:
        log.append(f"[Vectorizer] Width = {width}")
        vec_result = vectorize_loop(loop, width)
        for instr in vec_result.vector_body:
            log.append(f"[Vectorizer] Generated {instr.opcode}")
        after = [instruction_to_text(i) for i in vec_result.vector_body]
        if vec_result.remainder:
            log.append(
                f"[Tail] {vec_result.remainder} scalar remainder iteration(s) "
                f"starting at index {vec_result.tail.start_index}"
            )
        metrics = compute_metrics(loop, vec_result, width)
        if metrics:
            log.append(f"[Metrics] Instruction reduction = {metrics.instructionReduction:.0f}%")

    return PipelineResult(
        loop=loop, legality=legality, memory_accesses=legality.memory_accesses,
        vec_result=vec_result, metrics=metrics, before=before, after=after, log=log,
    )


# --------------------------------------------------------------------------
# Human-readable report (used by the CLI)
# --------------------------------------------------------------------------

def format_report(result: PipelineResult, width: int, verify_result: Optional[VerifyResult] = None) -> str:
    loop = result.loop
    legality = result.legality
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("Toy Auto-Vectorization Compiler")
    lines.append("=" * 60)
    lines.append(f"Loop: for {loop.induction_variable} = {loop.start} to {loop.end} step {loop.step}")
    lines.append(f"Vector width: {width}")
    lines.append("")
    lines.append(f"Dependency analysis: {'PASS' if legality.legal else 'FAIL'}")
    lines.append(f"Memory analysis: {len(legality.memory_accesses)} access(es) examined")
    for ma in legality.memory_accesses:
        lines.append(f"  - {'WRITE' if ma.is_write else 'READ '} {access_ref_text(ma)}")
    lines.append(f"Vectorization legality: {'LEGAL' if legality.legal else 'ILLEGAL'}")
    lines.append("")

    if legality.legal and result.vec_result is not None:
        lines.append("Transformation:")
        lines.append("  Before (scalar):")
        for l in result.before:
            lines.append(f"    {l}")
        lines.append("  After (vectorized):")
        for l in result.after:
            lines.append(f"    {l}")
        if result.vec_result.remainder:
            lines.append(f"  Tail (SCALAR_REMAINDER): {result.vec_result.remainder} iteration(s)")
        lines.append("")
        if result.metrics:
            m = result.metrics
            lines.append(f"Scalar instructions:   {m.scalarInstructions}")
            lines.append(f"Vector instructions:   {m.vectorInstructions}")
            lines.append(f"Instruction reduction: {m.instructionReduction:.2f}%")
            lines.append(f"Vector iterations: {m.vectorIterations}, remainder: {m.remainderIterations}")
        else:
            lines.append("Metrics: unavailable (symbolic loop bound)")
        if verify_result is not None:
            lines.append(f"Verification: {verify_result.status}")
            for mm in verify_result.mismatches[:10]:
                lines.append(f"    MISMATCH: {mm}")
        lines.append("")
        lines.append("Status: SUCCESS")
    else:
        lines.append(format_diagnostic_block(result))
        lines.append("")
        lines.append("Status: FAILED")

    return "\n".join(lines)


def format_diagnostic_block(result: PipelineResult) -> str:
    legality = result.legality
    lines = ["Diagnostic:"]
    for r in legality.reasons:
        lines.append(f"  Reason: {r}")
    for d in legality.dependencies:
        lines.append(f"  Instruction: {d.target}")
        lines.append(f"  Conflict: {d.target} vs {d.source}")
        lines.append(f"  Dependency distance: {d.distance}")
        lines.append(f"  Classification: {d.type} (loop-carried dependency)")
    return "\n".join(lines)
