"""Scalar remainder ("tail") handling when N % W != 0.

The main vectorized loop covers `vector_iterations * width` scalar
iterations; whatever's left over runs as a plain scalar loop (the same
scalar body instructions), so that Invariant 5 -- no lost iterations --
holds even when N isn't a multiple of the vector width.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..ir.instructions import Instruction


@dataclass
class TailInfo:
    remainder: int
    start_index: int
    instructions: List[Instruction] = field(default_factory=list)


def build_tail(body: List[Instruction], start_index: int, remainder: int) -> Optional[TailInfo]:
    if remainder <= 0:
        return None
    return TailInfo(remainder=remainder, start_index=start_index, instructions=list(body))
