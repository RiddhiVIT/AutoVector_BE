"""Top-level scalar -> vector transformation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..ir.instructions import Instruction
from ..ir.loop import Loop
from .packing import pack_body
from .tail import TailInfo, build_tail

VALID_WIDTHS = (2, 4, 8, 16)


@dataclass
class VectorizationResult:
    vector_body: List[Instruction]
    width: int
    vector_iterations: Optional[int]
    remainder: Optional[int]
    tail: Optional[TailInfo]
    symbolic: bool


def _iteration_count(loop: Loop) -> Optional[int]:
    """Number of scalar iterations, or None if the bound is symbolic."""
    if isinstance(loop.end, str):
        return None
    if loop.step <= 0:
        return 0
    span = loop.end - loop.start
    if span <= 0:
        return 0
    return (span + loop.step - 1) // loop.step


def vectorize_loop(loop: Loop, width: int) -> VectorizationResult:
    if width not in VALID_WIDTHS:
        raise ValueError(f"Invalid vector width: {width}. Must be one of {VALID_WIDTHS}.")

    vector_body = pack_body(loop.body, width)
    symbolic = isinstance(loop.end, str)

    if symbolic:
        return VectorizationResult(
            vector_body=vector_body, width=width,
            vector_iterations=None, remainder=None, tail=None, symbolic=True,
        )

    n_iters = _iteration_count(loop) or 0
    vector_iterations = n_iters // width
    remainder = n_iters % width
    start_index = loop.start + vector_iterations * width * loop.step
    tail = build_tail(loop.body, start_index, remainder)

    return VectorizationResult(
        vector_body=vector_body, width=width,
        vector_iterations=vector_iterations, remainder=remainder,
        tail=tail, symbolic=False,
    )
