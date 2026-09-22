"""Serialize IR back to toy-ir text or JSON."""
from __future__ import annotations

from typing import Any, Dict, List

from .instructions import Instruction
from .loop import Loop


def instruction_to_text(instr: Instruction) -> str:
    """Render a single instruction as toy-ir text, e.g. 'LOAD t1, A[i]'."""
    parts: List[str] = []
    if instr.destination is not None:
        parts.append(str(instr.destination))
    parts.extend(str(o) for o in instr.operands)
    return f"{instr.opcode} " + ", ".join(parts)


def body_to_text(body: List[Instruction]) -> List[str]:
    return [instruction_to_text(i) for i in body]


def loop_to_text(loop: Loop) -> str:
    lines = [f"LOOP {loop.induction_variable} {loop.start} {loop.end} {loop.step}"]
    lines.extend(instruction_to_text(i) for i in loop.body)
    lines.append("ENDLOOP")
    return "\n".join(lines)


def loop_to_json(loop: Loop) -> Dict[str, Any]:
    return {
        "loop": {
            "variable": loop.induction_variable,
            "start": loop.start,
            "end": loop.end,
            "step": loop.step,
        },
        "body": [
            {"opcode": i.opcode, "destination": i.destination, "operands": list(i.operands)}
            for i in loop.body
        ],
    }
