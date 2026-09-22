"""Vectorization legality checker.

Ties together memory-access extraction and dependency analysis to decide
whether a loop can be safely vectorized, per the MVP rules:

- Non-unit stride (step != 1) or negative step -> illegal.
- Any memory reference whose index isn't exactly i, i+const, or i-const
  -> conservative rejection (can't prove independence).
- Any loop-carried RAW/WAR/WAW dependency (same array, differing offsets,
  at least one write) -> illegal.
- Different array names are assumed non-aliasing (see analysis/alias.py).
- Reduction patterns (scalar accumulation across iterations) are out of
  scope for the parser, since the grammar only supports array-indexed
  statements, so they never reach this checker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from ..ir.instructions import MemoryAccess
from ..ir.loop import Loop
from .dependency import Dependency, find_loop_carried_dependencies
from .memory import UnknownAccessPatternError, extract_memory_accesses


@dataclass
class LegalityResult:
    legal: bool
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    memory_accesses: List[MemoryAccess] = field(default_factory=list)
    dependencies: List[Dependency] = field(default_factory=list)


class VectorizationLegalityChecker:
    def check(self, loop: Loop) -> LegalityResult:
        reasons: List[str] = []
        warnings: List[str] = []

        if loop.step < 0:
            reasons.append("Negative step is not supported for vectorization.")
        elif loop.step != 1:
            reasons.append(
                f"Non-unit stride loop (step={loop.step}) is not supported "
                "for vectorization."
            )

        try:
            accesses = extract_memory_accesses(loop)
        except UnknownAccessPatternError as e:
            reasons.append(str(e))
            return LegalityResult(legal=False, reasons=reasons, warnings=warnings)

        deps = find_loop_carried_dependencies(accesses)
        for d in deps:
            reasons.append(
                f"Loop-carried dependency detected: {d.target} depends on "
                f"{d.source} ({d.type}, distance {d.distance})"
            )

        legal = len(reasons) == 0
        return LegalityResult(
            legal=legal, reasons=reasons, warnings=warnings,
            memory_accesses=accesses, dependencies=deps,
        )
