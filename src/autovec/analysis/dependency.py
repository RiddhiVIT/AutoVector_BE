"""Def-use / RAW-WAR-WAW dependency analysis.

Intra-iteration dependencies (the def-use chain within one pass through the
loop body, e.g. LOAD t1 -> ADD t1,t2) are always safe -- a single scalar
iteration's instructions are already correctly ordered and vectorizing just
replays that same order with wider registers.

Loop-carried dependencies compare every pair of memory accesses to the same
array (at least one of which is a write) across iterations, based on their
integer offsets relative to the induction variable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..ir.instructions import Instruction, MemoryAccess
from .memory import access_ref_text


@dataclass
class Dependency:
    type: str        # "RAW", "WAR", or "WAW"
    distance: int     # abs(offset difference)
    source: str       # textual access this instruction depends on, e.g. "A[i-1]"
    target: str       # textual access that carries the dependency, e.g. "A[i]"
    array: str = ""


def intra_iteration_def_use(body: List[Instruction]) -> List[tuple]:
    """Return (producer_index, consumer_index, register) triples for the
    def-use chain within a single iteration of the loop body. Always safe."""
    edges: List[tuple] = []
    def_index = {}
    for i, instr in enumerate(body):
        for operand in instr.operands:
            if isinstance(operand, str) and operand in def_index:
                edges.append((def_index[operand], i, operand))
        if instr.destination is not None:
            def_index[instr.destination] = i
    return edges


def _classify_pair(a: MemoryAccess, b: MemoryAccess, array: str) -> Optional[Dependency]:
    if a.is_write and b.is_write:
        # WAW: order doesn't matter for distance, but keep it deterministic.
        target, source = (a, b) if a.offset >= b.offset else (b, a)
        distance = abs(a.offset - b.offset)
        return Dependency(
            type="WAW", distance=distance,
            source=access_ref_text(source), target=access_ref_text(target),
            array=array,
        )

    write = a if a.is_write else b
    read = b if a.is_write else a
    diff = write.offset - read.offset
    if diff > 0:
        # write happens at an earlier iteration than the read of the same
        # location -> read-after-write
        return Dependency(
            type="RAW", distance=diff,
            source=access_ref_text(read), target=access_ref_text(write),
            array=array,
        )
    if diff < 0:
        # read happens at an earlier iteration than the write of the same
        # location -> write-after-read (anti-dependency)
        return Dependency(
            type="WAR", distance=-diff,
            source=access_ref_text(read), target=access_ref_text(write),
            array=array,
        )
    return None  # same offset -> intra-iteration, safe


def find_loop_carried_dependencies(memory_accesses: List[MemoryAccess]) -> List[Dependency]:
    by_array: dict = {}
    for ma in memory_accesses:
        by_array.setdefault(ma.array, []).append(ma)

    deps: List[Dependency] = []
    for array, accesses in by_array.items():
        n = len(accesses)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = accesses[i], accesses[j]
                if a.offset == b.offset:
                    continue  # same location within the same iteration: safe
                if not a.is_write and not b.is_write:
                    continue  # read-after-read is never a hazard
                dep = _classify_pair(a, b, array)
                if dep is not None:
                    deps.append(dep)
    return deps
