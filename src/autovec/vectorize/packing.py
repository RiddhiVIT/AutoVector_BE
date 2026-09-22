"""Instruction packing: turn each scalar instruction into the vector
instruction that represents it operating across W scalar iterations at
once. Since a Loop's body is already exactly one iteration's straight-line
code, packing is a 1:1, order-preserving rewrite (LOAD->VLOAD, ADD->VADD,
...); the "packing across W iterations" is what the resulting vector op
*means* at runtime, not a structural unrolling in the IR.
"""
from __future__ import annotations

import re
from typing import List

from ..ir.instructions import Instruction
from .registers import scalar_to_vector_reg

_OPCODE_MAP = {
    "LOAD": "VLOAD",
    "STORE": "VSTORE",
    "ADD": "VADD",
    "SUB": "VSUB",
    "MUL": "VMUL",
    "DIV": "VDIV",
    "CONST": "VBROADCAST",
}

_SCALAR_REG_RE = re.compile(r"^t\d+$")
_MEM_REF_RE = re.compile(r"^[A-Za-z_]\w*\[.+\]$")


def _vectorize_ref(ref: str, width: int) -> str:
    from ..analysis.memory import split_memory_ref

    array, index_expr = split_memory_ref(ref)
    idx = index_expr.strip()
    return f"{array}[{idx}:{idx}+{width}]"


def _vectorize_operand(operand, width: int):
    if isinstance(operand, str) and _SCALAR_REG_RE.match(operand):
        return scalar_to_vector_reg(operand)
    if isinstance(operand, str) and _MEM_REF_RE.match(operand):
        return _vectorize_ref(operand, width)
    return operand  # int literal or already-vector register


def pack_instruction(instr: Instruction, width: int) -> Instruction:
    vopcode = _OPCODE_MAP.get(instr.opcode)
    if vopcode is None:
        raise ValueError(f"Cannot vectorize opcode: {instr.opcode}")

    dest = scalar_to_vector_reg(instr.destination) if instr.destination is not None else None
    operands = [_vectorize_operand(o, width) for o in instr.operands]
    return Instruction(opcode=vopcode, destination=dest, operands=operands)


def pack_body(body: List[Instruction], width: int) -> List[Instruction]:
    return [pack_instruction(instr, width) for instr in body]
