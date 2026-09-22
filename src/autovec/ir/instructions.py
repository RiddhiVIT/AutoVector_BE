"""Core IR data model: Instruction and MemoryAccess."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Instruction:
    """A single IR instruction.

    opcode: one of LOAD, STORE, ADD, SUB, MUL, DIV, CONST,
            VLOAD, VSTORE, VADD, VSUB, VMUL, VDIV, VBROADCAST
    destination: destination register name, or None for STORE/VSTORE
    operands: list of operand tokens. For LOAD/VLOAD the single operand is
        the memory reference text (e.g. "A[i]" or "A[i:i+4]"). For
        STORE/VSTORE, operands = [memory_ref, value_register]. For
        arithmetic ops, operands = [lhs, rhs]. For CONST/VBROADCAST,
        operands = [value].
    """

    opcode: str
    destination: Optional[str]
    operands: list = field(default_factory=list)


@dataclass
class MemoryAccess:
    """A single memory access extracted from an instruction.

    array: array name, e.g. "A"
    index: induction variable name, e.g. "i"
    offset: integer offset relative to the induction variable
            (A[i] -> 0, A[i-1] -> -1, A[i+1] -> 1)
    is_write: True for a store, False for a load
    """

    array: str
    index: str
    offset: int
    is_write: bool
