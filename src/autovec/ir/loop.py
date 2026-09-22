"""Loop IR node."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from .instructions import Instruction


@dataclass
class Loop:
    """A single-basic-block, count-controlled loop.

    start: integer literal
    end: integer literal (concrete) OR a symbolic name (e.g. "N")
    step: integer literal
    body: the straight-line instruction list executed once per iteration
    """

    induction_variable: str
    start: int
    end: Union[int, str]
    step: int
    body: list = field(default_factory=list)
