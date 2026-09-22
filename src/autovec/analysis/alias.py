"""Minimal alias analysis (MVP rule).

The restricted grammar this compiler accepts has no pointer arithmetic and
no aliasing syntax -- every memory operand names a concrete array. So the
MVP rule is simple: distinct array names are assumed to reference disjoint
memory (non-aliasing), and there is nothing resembling an "unknown pointer"
that a parseable program could produce. This module exists to make that
assumption explicit and to provide small helpers used by the legality
checker and diagnostics.
"""
from __future__ import annotations

from typing import Dict, List

from ..ir.instructions import MemoryAccess


def group_by_array(accesses: List[MemoryAccess]) -> Dict[str, List[MemoryAccess]]:
    groups: Dict[str, List[MemoryAccess]] = {}
    for a in accesses:
        groups.setdefault(a.array, []).append(a)
    return groups


def distinct_arrays(accesses: List[MemoryAccess]) -> List[str]:
    seen: List[str] = []
    for a in accesses:
        if a.array not in seen:
            seen.append(a.array)
    return seen


def non_aliasing_warnings(accesses: List[MemoryAccess]) -> List[str]:
    """Different array names are assumed not to alias per the MVP rule.
    Returns an informational warning list (empty unless in the future this
    grows real alias heuristics)."""
    arrays = distinct_arrays(accesses)
    if len(arrays) > 1:
        return [
            f"Assuming arrays {', '.join(arrays)} are non-aliasing "
            "(distinct-name MVP alias rule)."
        ]
    return []
