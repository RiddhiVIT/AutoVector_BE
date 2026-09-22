"""Memory access extraction: parse "A[i]", "A[i-1]", "A[i+1]" style
references relative to the loop's induction variable, and reject anything
that isn't provably an affine i+const pattern.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ..ir.instructions import Instruction, MemoryAccess
from ..ir.loop import Loop

_MEM_REF_RE = re.compile(r"^([A-Za-z_]\w*)\[(.+)\]$")


class UnknownAccessPatternError(Exception):
    """Raised when a memory reference's index expression cannot be proven
    to be an affine i+const relative to the induction variable."""

    def __init__(self, ref: str):
        self.ref = ref
        super().__init__(
            f"Unknown memory access pattern: {ref!r}. "
            "Unable to prove iteration independence."
        )


def split_memory_ref(ref: str) -> Tuple[str, str]:
    """Split "A[i-1]" into ("A", "i-1")."""
    m = _MEM_REF_RE.match(ref.strip())
    if not m:
        raise UnknownAccessPatternError(ref)
    return m.group(1), m.group(2)


def parse_offset(index_expr: str, iv: str) -> Optional[int]:
    """Return the integer offset of index_expr relative to iv, or None if
    index_expr is not exactly iv, iv+<int>, or iv-<int>."""
    s = index_expr.strip()
    if s == iv:
        return 0
    m = re.match(rf"^{re.escape(iv)}\s*\+\s*(\d+)$", s)
    if m:
        return int(m.group(1))
    m = re.match(rf"^{re.escape(iv)}\s*-\s*(\d+)$", s)
    if m:
        return -int(m.group(1))
    return None


def make_memory_access(ref: str, iv: str, is_write: bool) -> MemoryAccess:
    array, index_expr = split_memory_ref(ref)
    offset = parse_offset(index_expr, iv)
    if offset is None:
        raise UnknownAccessPatternError(ref)
    return MemoryAccess(array=array, index=iv, offset=offset, is_write=is_write)


def extract_memory_accesses(loop: Loop) -> List[MemoryAccess]:
    """Extract, in program order, every memory access made by the loop body.
    Raises UnknownAccessPatternError if any reference can't be proven affine."""
    iv = loop.induction_variable
    accesses: List[MemoryAccess] = []
    for instr in loop.body:
        op = instr.opcode
        if op in ("LOAD", "VLOAD"):
            accesses.append(make_memory_access(instr.operands[0], iv, is_write=False))
        elif op in ("STORE", "VSTORE"):
            accesses.append(make_memory_access(instr.operands[0], iv, is_write=True))
    return accesses


def access_ref_text(ma: MemoryAccess) -> str:
    """Render a MemoryAccess back to "A[i]" / "A[i-1]" / "A[i+1]" text."""
    if ma.offset == 0:
        idx = ma.index
    elif ma.offset > 0:
        idx = f"{ma.index}+{ma.offset}"
    else:
        idx = f"{ma.index}{ma.offset}"  # offset already carries the '-'
    return f"{ma.array}[{idx}]"
