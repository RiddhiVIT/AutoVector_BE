"""Scalar interpreter: executes LOAD/STORE/ADD/SUB/MUL/DIV/CONST over
Python lists standing in for arrays."""
from __future__ import annotations

from typing import Any, Dict, List

from ..analysis.memory import parse_offset, split_memory_ref
from ..ir.instructions import Instruction
from ..ir.loop import Loop


def _apply_op(op: str, a: int, b: int) -> int:
    if op == "ADD":
        return a + b
    if op == "SUB":
        return a - b
    if op == "MUL":
        return a * b
    if op == "DIV":
        return a // b if b != 0 else 0
    raise ValueError(f"Unsupported arithmetic op: {op}")


def _resolve_ref(ref: str, iv: str, i_val: int):
    array, index_expr = split_memory_ref(ref)
    offset = parse_offset(index_expr, iv)
    if offset is None:
        raise ValueError(f"Cannot execute unknown access pattern: {ref}")
    return array, i_val + offset


def _resolve_operand(token, regs: Dict[str, int]) -> int:
    if isinstance(token, int):
        return token
    if token in regs:
        return regs[token]
    return int(token)


def exec_instruction(instr: Instruction, regs: Dict[str, int], arrays: Dict[str, List[int]],
                      iv: str, i_val: int) -> None:
    op = instr.opcode
    if op == "LOAD":
        array, idx = _resolve_ref(instr.operands[0], iv, i_val)
        regs[instr.destination] = arrays[array][idx]
    elif op == "STORE":
        array, idx = _resolve_ref(instr.operands[0], iv, i_val)
        arrays[array][idx] = _resolve_operand(instr.operands[1], regs)
    elif op == "CONST":
        regs[instr.destination] = int(instr.operands[0])
    elif op in ("ADD", "SUB", "MUL", "DIV"):
        a = _resolve_operand(instr.operands[0], regs)
        b = _resolve_operand(instr.operands[1], regs)
        regs[instr.destination] = _apply_op(op, a, b)
    else:
        raise ValueError(f"Unsupported scalar opcode: {op}")


def run_scalar(loop: Loop, arrays: Dict[str, List[int]], n_iters: int,
               start_iter: int = 0) -> Dict[str, List[int]]:
    """Execute `loop.body` scalarly for `n_iters` iterations, starting at
    the `start_iter`-th iteration (0-indexed relative to loop.start)."""
    iv = loop.induction_variable
    for it in range(start_iter, start_iter + n_iters):
        i_val = loop.start + it * loop.step
        regs: Dict[str, int] = {}
        for instr in loop.body:
            exec_instruction(instr, regs, arrays, iv, i_val)
    return arrays
