"""Parsers that turn textual/JSON source into the IR.

Three input formats are supported:

1. Toy IR text format (see README):
       LOOP i 0 N 1
       LOAD t1, A[i]
       ...
       ENDLOOP

2. Raw JSON format:
       {"loop": {"variable": "i", "start": 0, "end": "N", "step": 1},
        "body": [{"op": "load", "dst": "t1", "array": "A", "index": "i"}, ...]}

3. Restricted C-like format:
       for (int i = 0; i < 100; i++) { C[i] = A[i] + B[i]; }

All parsers raise ParseError with a human-readable message on invalid input.
"""
from __future__ import annotations

import json
import re
from typing import Any, List

from .instructions import Instruction
from .loop import Loop


class ParseError(Exception):
    """Raised when source text cannot be parsed into valid IR."""


def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------
# Toy IR text parser
# --------------------------------------------------------------------------

_LOOP_RE = re.compile(r"^LOOP\s+(\w+)\s+(\S+)\s+(\S+)\s+(\S+)$", re.IGNORECASE)

_LOAD_OPS = {"LOAD", "VLOAD"}
_STORE_OPS = {"STORE", "VSTORE"}
_CONST_OPS = {"CONST", "VBROADCAST"}
_ARITH_OPS = {"ADD", "SUB", "MUL", "DIV", "VADD", "VSUB", "VMUL", "VDIV"}
_ALL_OPS = _LOAD_OPS | _STORE_OPS | _CONST_OPS | _ARITH_OPS


def parse_toy_ir(text: str) -> Loop:
    if not isinstance(text, str) or not text.strip():
        raise ParseError("Empty toy-ir source.")

    raw_lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in raw_lines if ln and not ln.startswith("#")]
    if not lines:
        raise ParseError("Empty toy-ir source.")

    header = lines[0]
    m = _LOOP_RE.match(header)
    if not m:
        raise ParseError(
            f"Invalid LOOP header: {header!r}. "
            "Expected 'LOOP <var> <start> <end> <step>'."
        )
    var, start_s, end_s, step_s = m.groups()
    if not _is_int(start_s):
        raise ParseError(f"Loop start must be an integer literal, got {start_s!r}.")
    if not _is_int(step_s):
        raise ParseError(f"Loop step must be an integer literal, got {step_s!r}.")
    start = int(start_s)
    step = int(step_s)
    end: Any = int(end_s) if _is_int(end_s) else end_s

    body: List[Instruction] = []
    saw_endloop = False
    for ln in lines[1:]:
        if ln.upper() == "ENDLOOP":
            saw_endloop = True
            break
        body.append(_parse_toy_instruction(ln))

    if not saw_endloop:
        raise ParseError("Missing ENDLOOP terminator.")
    if not body:
        raise ParseError("Loop body is empty.")

    return Loop(induction_variable=var, start=start, end=end, step=step, body=body)


def _parse_toy_instruction(line: str) -> Instruction:
    tokens = [t for t in line.replace(",", " ").split() if t]
    if not tokens:
        raise ParseError(f"Empty instruction line: {line!r}")

    opcode = tokens[0].upper()
    rest = tokens[1:]

    if opcode not in _ALL_OPS:
        raise ParseError(f"Unknown opcode {tokens[0]!r} in line: {line!r}")

    if opcode in _LOAD_OPS or opcode in _CONST_OPS:
        if len(rest) != 2:
            raise ParseError(f"Malformed {opcode} instruction: {line!r}")
        dest, ref = rest
        return Instruction(opcode=opcode, destination=dest, operands=[ref])

    if opcode in _STORE_OPS:
        if len(rest) != 2:
            raise ParseError(f"Malformed {opcode} instruction: {line!r}")
        ref, val = rest
        return Instruction(opcode=opcode, destination=None, operands=[ref, val])

    if opcode in _ARITH_OPS:
        if len(rest) != 3:
            raise ParseError(f"Malformed {opcode} instruction: {line!r}")
        dest, a, b = rest
        return Instruction(opcode=opcode, destination=dest, operands=[a, b])

    raise ParseError(f"Unhandled opcode {opcode!r} in line: {line!r}")


# --------------------------------------------------------------------------
# JSON parser
# --------------------------------------------------------------------------

def parse_json(source: Any) -> Loop:
    if isinstance(source, (str, bytes)):
        try:
            data = json.loads(source)
        except json.JSONDecodeError as e:
            raise ParseError(f"Invalid JSON: {e}") from e
    elif isinstance(source, dict):
        data = source
    else:
        raise ParseError("JSON source must be a string or object.")

    if not isinstance(data, dict) or "loop" not in data:
        raise ParseError("JSON source must contain a top-level 'loop' object.")

    loop_info = data["loop"]
    try:
        var = loop_info["variable"]
        start = int(loop_info["start"])
        end_raw = loop_info["end"]
        end: Any = int(end_raw) if _is_int(str(end_raw)) else end_raw
        step = int(loop_info.get("step", 1))
    except (KeyError, TypeError, ValueError) as e:
        raise ParseError(f"Invalid 'loop' section: {e}") from e

    body_items = data.get("body", [])
    if not isinstance(body_items, list) or not body_items:
        raise ParseError("JSON 'body' must be a non-empty list of instructions.")

    body = [_parse_json_instruction(item, i) for i, item in enumerate(body_items)]
    return Loop(induction_variable=var, start=start, end=end, step=step, body=body)


def _parse_json_instruction(item: Any, idx: int) -> Instruction:
    if not isinstance(item, dict) or "op" not in item:
        raise ParseError(f"Body item {idx} must be an object with an 'op' field.")
    op = str(item["op"]).lower()
    try:
        if op == "load":
            ref = f"{item['array']}[{item['index']}]"
            return Instruction(opcode="LOAD", destination=item["dst"], operands=[ref])
        if op == "store":
            ref = f"{item['array']}[{item['index']}]"
            return Instruction(opcode="STORE", destination=None, operands=[ref, item["value"]])
        if op in ("add", "sub", "mul", "div"):
            return Instruction(
                opcode=op.upper(), destination=item["dst"],
                operands=[item["lhs"], item["rhs"]],
            )
        if op == "const":
            return Instruction(opcode="CONST", destination=item["dst"], operands=[item["value"]])
    except KeyError as e:
        raise ParseError(f"Body item {idx} ({op}) missing field {e}.") from e

    raise ParseError(f"Unknown op {op!r} in body item {idx}.")


# --------------------------------------------------------------------------
# Restricted C-like parser
# --------------------------------------------------------------------------

_FOR_RE = re.compile(
    r"for\s*\(\s*int\s+(\w+)\s*=\s*(-?\d+)\s*;\s*\1\s*<\s*([A-Za-z_]\w*|\d+)\s*;"
    r"\s*\1\s*\+\+\s*\)\s*\{(.*)\}\s*$",
    re.DOTALL,
)

_ASSIGN_RE = re.compile(r"^([A-Za-z_]\w*)\[([^\]]+)\]\s*=\s*(.+)$")

_TOKEN_RE = re.compile(r"[A-Za-z_]\w*\[[^\]]*\]|\d+|[+\-*/]")

_INDEX_RE_TMPL = r"^{iv}\s*[+-]\s*\d+$"

def parse_c_like(text: str) -> Loop:
    if not isinstance(text, str) or not text.strip():
        raise ParseError("Empty C-like source.")

    src = text.strip()
    m = _FOR_RE.search(src)
    if not m:
        raise ParseError(
            "Could not parse C-like source. Expected the form: "
            "'for (int i = <start>; i < <end>; i++) { <ArrayName>[<index>] = <expr>; ... }'."
        )
    var, start_s, end_s, body_src = m.groups()
    start = int(start_s)
    end: Any = int(end_s) if _is_int(end_s) else end_s

    stmts = [s.strip() for s in body_src.split(";")]
    stmts = [s for s in stmts if s]
    if not stmts:
        raise ParseError("Loop body is empty.")

    body: List[Instruction] = []
    counter = [0]
    for stmt in stmts:
        body.extend(_lower_c_statement(stmt, var, counter))

    return Loop(induction_variable=var, start=start, end=end, step=1, body=body)


def _fresh_temp(counter: List[int]) -> str:
    counter[0] += 1
    return f"t{counter[0]}"


def _validate_index_expr(idx_expr: str, iv: str, stmt: str) -> None:
    s = idx_expr.strip()
    if s == iv or re.match(_INDEX_RE_TMPL.format(iv=re.escape(iv)), s):
        return
    raise ParseError(
        f"Unsupported index expression {idx_expr!r} in statement {stmt!r}. "
        f"Only '{iv}', '{iv}+<int>', or '{iv}-<int>' are allowed."
    )


def _lower_c_statement(stmt: str, iv: str, counter: List[int]) -> List[Instruction]:
    m = _ASSIGN_RE.match(stmt)
    if not m:
        raise ParseError(f"Invalid statement: {stmt!r}. Expected 'Array[index] = expr;'.")
    lhs_array, lhs_index, rhs = m.groups()
    _validate_index_expr(lhs_index, iv, stmt)

    tokens = _TOKEN_RE.findall(rhs)
    stripped_rhs = re.sub(r"\s+", "", rhs)
    if "".join(tokens) != stripped_rhs:
        raise ParseError(f"Invalid expression: {rhs!r} in statement {stmt!r}.")
    if not tokens:
        raise ParseError(f"Empty expression in statement: {stmt!r}")

    instrs: List[Instruction] = []
    running_reg = _emit_term(tokens[0], iv, stmt, counter, instrs)

    pos = 1
    while pos < len(tokens):
        op_tok = tokens[pos]
        if op_tok not in ("+", "-", "*", "/"):
            raise ParseError(f"Expected an operator in expression: {rhs!r}")
        if pos + 1 >= len(tokens):
            raise ParseError(f"Dangling operator in expression: {rhs!r}")
        term_tok = tokens[pos + 1]
        term_reg = _emit_term(term_tok, iv, stmt, counter, instrs)
        opcode = {"+": "ADD", "-": "SUB", "*": "MUL", "/": "DIV"}[op_tok]
        dest = _fresh_temp(counter)
        instrs.append(Instruction(opcode=opcode, destination=dest, operands=[running_reg, term_reg]))
        running_reg = dest
        pos += 2

    ref = f"{lhs_array}[{lhs_index.strip()}]"
    instrs.append(Instruction(opcode="STORE", destination=None, operands=[ref, running_reg]))
    return instrs


def _emit_term(tok: str, iv: str, stmt: str, counter: List[int], instrs: List[Instruction]) -> str:
    if _is_int(tok):
        dest = _fresh_temp(counter)
        instrs.append(Instruction(opcode="CONST", destination=dest, operands=[int(tok)]))
        return dest

    am = re.match(r"^([A-Za-z_]\w*)\[([^\]]*)\]$", tok)
    if not am:
        raise ParseError(f"Invalid term {tok!r} in statement {stmt!r}.")
    array, idx = am.groups()
    _validate_index_expr(idx, iv, stmt)
    dest = _fresh_temp(counter)
    instrs.append(Instruction(opcode="LOAD", destination=dest, operands=[f"{array}[{idx.strip()}]"]))
    return dest
