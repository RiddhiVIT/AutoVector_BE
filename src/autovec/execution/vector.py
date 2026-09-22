"""Vector interpreter: simulated SIMD execution of VLOAD/VSTORE/VADD/VSUB/
VMUL/VDIV/VBROADCAST. The "vector engine" simply applies the corresponding
scalar op `width` times per instruction -- this is a toy compiler, not a
real hardware backend.

Also provides `verify()`, which runs both the scalar and vectorized IR
against deterministic generated arrays and compares the results.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..analysis.memory import extract_memory_accesses, parse_offset
from ..ir.loop import Loop
from ..vectorize.vectorizer import VectorizationResult
from .scalar import _apply_op, run_scalar

_VEC_REF_RE = re.compile(r"^([A-Za-z_]\w*)\[([^\]:]+):[^\]]+\]$")


def _split_vector_ref(ref: str):
    m = _VEC_REF_RE.match(ref.strip())
    if not m:
        raise ValueError(f"Malformed vector memory reference: {ref!r}")
    return m.group(1), m.group(2)


def _resolve_vec_operand(token, vregs: Dict[str, List[int]], width: int) -> List[int]:
    if isinstance(token, str) and token in vregs:
        return vregs[token]
    val = int(token)
    return [val] * width


def exec_vector_instruction(instr, vregs: Dict[str, List[int]], arrays: Dict[str, List[int]],
                             iv: str, i_val: int, width: int) -> None:
    op = instr.opcode
    if op == "VLOAD":
        array, idx_expr = _split_vector_ref(instr.operands[0])
        offset = parse_offset(idx_expr, iv)
        if offset is None:
            raise ValueError(f"Cannot resolve vector reference offset: {instr.operands[0]}")
        vregs[instr.destination] = [arrays[array][i_val + offset + k] for k in range(width)]
    elif op == "VSTORE":
        array, idx_expr = _split_vector_ref(instr.operands[0])
        offset = parse_offset(idx_expr, iv)
        if offset is None:
            raise ValueError(f"Cannot resolve vector reference offset: {instr.operands[0]}")
        vals = _resolve_vec_operand(instr.operands[1], vregs, width)
        for k in range(width):
            arrays[array][i_val + offset + k] = vals[k]
    elif op == "VBROADCAST":
        val = int(instr.operands[0])
        vregs[instr.destination] = [val] * width
    elif op in ("VADD", "VSUB", "VMUL", "VDIV"):
        a = _resolve_vec_operand(instr.operands[0], vregs, width)
        b = _resolve_vec_operand(instr.operands[1], vregs, width)
        scalar_op = op[1:]  # VADD -> ADD, etc.
        vregs[instr.destination] = [_apply_op(scalar_op, a[k], b[k]) for k in range(width)]
    else:
        raise ValueError(f"Unsupported vector opcode: {op}")


def run_vector(loop: Loop, vec_result: VectorizationResult, arrays: Dict[str, List[int]]) -> Dict[str, List[int]]:
    iv = loop.induction_variable
    width = vec_result.width

    for vit in range(vec_result.vector_iterations or 0):
        i_val = loop.start + vit * width * loop.step
        vregs: Dict[str, List[int]] = {}
        for instr in vec_result.vector_body:
            exec_vector_instruction(instr, vregs, arrays, iv, i_val, width)

    if vec_result.tail and vec_result.remainder:
        start_iter = vec_result.vector_iterations or 0
        run_scalar(loop, arrays, vec_result.remainder, start_iter=start_iter * width)

    return arrays


@dataclass
class VerifyResult:
    status: str  # "CORRECT" or "MISMATCH"
    mismatches: List[str]


def generate_test_arrays(loop: Loop, n_iters: int) -> Dict[str, List[int]]:
    """Deterministically seed one array per distinct array name referenced
    by the loop, sized to safely cover every offset the loop touches."""
    try:
        accesses = extract_memory_accesses(loop)
    except Exception:
        accesses = []

    names: List[str] = []
    min_offset = 0
    max_offset = 0
    for a in accesses:
        if a.array not in names:
            names.append(a.array)
        min_offset = min(min_offset, a.offset)
        max_offset = max(max_offset, a.offset)

    size = max(n_iters, 1) + loop.start + max(max_offset, 0) + abs(min(min_offset, 0)) + 8
    arrays: Dict[str, List[int]] = {}
    for idx, name in enumerate(names, start=1):
        arrays[name] = [k * idx for k in range(size)]
    return arrays


def verify(loop: Loop, vec_result: VectorizationResult, n_iters: Optional[int] = None) -> VerifyResult:
    """Run the scalar and vectorized IR against identical seeded arrays and
    compare the results array-by-array."""
    if n_iters is None:
        n_iters = (vec_result.vector_iterations or 0) * vec_result.width + (vec_result.remainder or 0)

    base_arrays = generate_test_arrays(loop, n_iters)

    scalar_arrays = copy.deepcopy(base_arrays)
    run_scalar(loop, scalar_arrays, n_iters)

    vector_arrays = copy.deepcopy(base_arrays)
    run_vector(loop, vec_result, vector_arrays)

    mismatches: List[str] = []
    for name in base_arrays:
        s_vals, v_vals = scalar_arrays[name], vector_arrays[name]
        for i in range(len(s_vals)):
            if s_vals[i] != v_vals[i]:
                mismatches.append(f"{name}[{i}]: scalar={s_vals[i]} vector={v_vals[i]}")

    status = "MISMATCH" if mismatches else "CORRECT"
    return VerifyResult(status=status, mismatches=mismatches)
