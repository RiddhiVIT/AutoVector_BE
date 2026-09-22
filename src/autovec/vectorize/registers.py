"""Scalar -> vector register renaming (t1 -> v1)."""
from __future__ import annotations

import re
from typing import Dict

_SCALAR_RE = re.compile(r"^t(\d+)$")


def scalar_to_vector_reg(reg: str) -> str:
    """Map a scalar register name (t1, t2, ...) to its vector counterpart
    (v1, v2, ...). Any other name is prefixed with 'v_' as a fallback so the
    mapping never collides with a plain integer literal."""
    m = _SCALAR_RE.match(reg)
    if m:
        return f"v{m.group(1)}"
    return f"v_{reg}"


class RegisterMap:
    """Keeps a consistent scalar->vector register mapping across the
    instructions of one vectorized loop body."""

    def __init__(self) -> None:
        self._map: Dict[str, str] = {}

    def get(self, scalar_reg: str) -> str:
        if scalar_reg not in self._map:
            self._map[scalar_reg] = scalar_to_vector_reg(scalar_reg)
        return self._map[scalar_reg]
