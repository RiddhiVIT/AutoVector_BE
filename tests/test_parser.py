import json

import pytest

from autovec.ir.instructions import Instruction
from autovec.ir.loop import Loop
from autovec.ir.parser import ParseError, parse_c_like, parse_json, parse_toy_ir
from autovec.ir.serializer import instruction_to_text, loop_to_text


TOY_IR_SRC = """LOOP i 0 N 1
LOAD t1, A[i]
LOAD t2, B[i]
ADD t3, t1, t2
STORE C[i], t3
ENDLOOP
"""


def test_parse_toy_ir_basic():
    loop = parse_toy_ir(TOY_IR_SRC)
    assert loop.induction_variable == "i"
    assert loop.start == 0
    assert loop.end == "N"  # symbolic
    assert loop.step == 1
    assert len(loop.body) == 4
    assert loop.body[0] == Instruction("LOAD", "t1", ["A[i]"])
    assert loop.body[3] == Instruction("STORE", None, ["C[i]", "t3"])


def test_parse_toy_ir_no_commas():
    src = "LOOP i 0 100 1\nLOAD t1 A[i]\nSTORE C[i] t1\nENDLOOP"
    loop = parse_toy_ir(src)
    assert loop.end == 100
    assert loop.body[0].operands == ["A[i]"]


def test_toy_ir_round_trip():
    loop = parse_toy_ir(TOY_IR_SRC)
    text = loop_to_text(loop)
    loop2 = parse_toy_ir(text)
    assert loop2.induction_variable == loop.induction_variable
    assert loop2.start == loop.start
    assert loop2.end == loop.end
    assert loop2.step == loop.step
    assert [instruction_to_text(i) for i in loop2.body] == [instruction_to_text(i) for i in loop.body]


def test_toy_ir_missing_endloop_raises():
    with pytest.raises(ParseError):
        parse_toy_ir("LOOP i 0 100 1\nLOAD t1, A[i]\n")


def test_toy_ir_bad_header_raises():
    with pytest.raises(ParseError):
        parse_toy_ir("LOOP i 0\nENDLOOP")


def test_toy_ir_empty_source_raises():
    with pytest.raises(ParseError):
        parse_toy_ir("")


def test_parse_c_like_basic():
    loop = parse_c_like("for (int i = 0; i < 100; i++) { C[i] = A[i] + B[i]; }")
    assert loop.induction_variable == "i"
    assert loop.start == 0
    assert loop.end == 100
    assert loop.step == 1
    assert [instruction_to_text(i) for i in loop.body] == [
        "LOAD t1, A[i]", "LOAD t2, B[i]", "ADD t3, t1, t2", "STORE C[i], t3",
    ]


def test_parse_c_like_symbolic_end():
    loop = parse_c_like("for (int i = 0; i < N; i++) { C[i] = A[i] + B[i]; }")
    assert loop.end == "N"


def test_parse_c_like_three_term_expr():
    loop = parse_c_like("for (int i = 0; i < 100; i++) { D[i] = A[i] + B[i] - C[i]; }")
    assert [instruction_to_text(i) for i in loop.body] == [
        "LOAD t1, A[i]", "LOAD t2, B[i]", "ADD t3, t1, t2",
        "LOAD t4, C[i]", "SUB t5, t3, t4", "STORE D[i], t5",
    ]


def test_parse_c_like_offset_and_const():
    loop = parse_c_like("for (int i = 1; i < 100; i++) { A[i] = A[i-1] + 1; }")
    assert [instruction_to_text(i) for i in loop.body] == [
        "LOAD t1, A[i-1]", "CONST t2, 1", "ADD t3, t1, t2", "STORE A[i], t3",
    ]


def test_parse_c_like_invalid_index_rejected():
    with pytest.raises(ParseError):
        parse_c_like("for (int i = 0; i < 100; i++) { C[i] = A[j]; }")


def test_parse_c_like_malformed_raises():
    with pytest.raises(ParseError):
        parse_c_like("not a loop at all")


def test_parse_c_like_dangling_operator_raises():
    with pytest.raises(ParseError):
        parse_c_like("for (int i = 0; i < 100; i++) { C[i] = A[i] +; }")


def test_parse_json_basic():
    data = {
        "loop": {"variable": "i", "start": 0, "end": "N", "step": 1},
        "body": [
            {"op": "load", "dst": "t1", "array": "A", "index": "i"},
            {"op": "load", "dst": "t2", "array": "B", "index": "i"},
            {"op": "add", "dst": "t3", "lhs": "t1", "rhs": "t2"},
            {"op": "store", "array": "C", "index": "i", "value": "t3"},
        ],
    }
    loop = parse_json(json.dumps(data))
    assert loop.induction_variable == "i"
    assert loop.end == "N"
    assert [instruction_to_text(i) for i in loop.body] == [
        "LOAD t1, A[i]", "LOAD t2, B[i]", "ADD t3, t1, t2", "STORE C[i], t3",
    ]


def test_parse_json_accepts_dict_directly():
    data = {
        "loop": {"variable": "i", "start": 0, "end": 10, "step": 1},
        "body": [{"op": "const", "dst": "t1", "value": 5}, {"op": "store", "array": "A", "index": "i", "value": "t1"}],
    }
    loop = parse_json(data)
    assert loop.end == 10


def test_parse_json_invalid_raises():
    with pytest.raises(ParseError):
        parse_json("{not valid json")


def test_parse_json_missing_loop_raises():
    with pytest.raises(ParseError):
        parse_json(json.dumps({"body": []}))


def test_parse_json_empty_body_raises():
    with pytest.raises(ParseError):
        parse_json(json.dumps({"loop": {"variable": "i", "start": 0, "end": 10, "step": 1}, "body": []}))
