"""Instruction-count metrics for the scalar-vs-vector transformation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..ir.loop import Loop
from ..vectorize.vectorizer import VectorizationResult


@dataclass
class Metrics:
    scalarInstructions: int
    vectorInstructions: int
    instructionReduction: float
    vectorIterations: Optional[int]
    remainderIterations: Optional[int]


def compute_metrics(loop: Loop, vec_result: VectorizationResult, width: int) -> Optional[Metrics]:
    """Returns None when the loop bound is symbolic -- concrete counts
    can't be computed without a known N."""
    if vec_result.symbolic or vec_result.vector_iterations is None:
        return None

    body_len = len(loop.body)
    vector_iterations = vec_result.vector_iterations
    remainder = vec_result.remainder or 0

    n_iters = vector_iterations * width + remainder
    scalar_instructions = n_iters * body_len
    vector_instructions = vector_iterations * body_len + remainder * body_len

    if scalar_instructions > 0:
        reduction = (scalar_instructions - vector_instructions) / scalar_instructions * 100
    else:
        reduction = 0.0

    return Metrics(
        scalarInstructions=scalar_instructions,
        vectorInstructions=vector_instructions,
        instructionReduction=round(reduction, 4),
        vectorIterations=vector_iterations,
        remainderIterations=remainder,
    )
