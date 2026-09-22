import pytest

from autovec.analysis.memory import (
    UnknownAccessPatternError,
    access_ref_text,
    extract_memory_accesses,
    parse_offset,
    split_memory_ref,
)
from autovec.ir.instructions import MemoryAccess
from autovec.ir.parser import parse_c_like, parse_toy_ir


def test_split_memory_ref():
    assert split_memory_ref("A[i]") == ("A", "i")
    assert split_memory_ref("A[i-1]") == ("A", "i-1")
    assert split_memory_ref("A[i+2]") == ("A", "i+2")


def test_split_memory_ref_invalid_raises():
    with pytest.raises(UnknownAccessPatternError):
        split_memory_ref("not_an_array_ref")


@pytest.mark.parametrize("expr,expected", [("i", 0), ("i+1", 1), ("i-1", -1), ("i+10", 10)])
def test_parse_offset_known_patterns(expr, expected):
    assert parse_offset(expr, "i") == expected


@pytest.mark.parametrize("expr", ["j", "2*i", "i*4", "i+i"])
def test_parse_offset_unknown_patterns(expr):
    assert parse_offset(expr, "i") is None


def test_extract_memory_accesses_basic_add():
    loop = parse_c_like("for (int i = 0; i < 100; i++) { C[i] = A[i] + B[i]; }")
    accesses = extract_memory_accesses(loop)
    assert accesses == [
        MemoryAccess("A", "i", 0, False),
        MemoryAccess("B", "i", 0, False),
        MemoryAccess("C", "i", 0, True),
    ]


def test_extract_memory_accesses_with_offsets():
    loop = parse_c_like("for (int i = 1; i < 100; i++) { A[i] = A[i-1] + 1; }")
    accesses = extract_memory_accesses(loop)
    assert accesses == [
        MemoryAccess("A", "i", -1, False),
        MemoryAccess("A", "i", 0, True),
    ]


def test_extract_memory_accesses_unknown_pattern_raises():
    src = "LOOP i 0 100 1\nLOAD t1, A[j]\nSTORE C[i], t1\nENDLOOP"
    loop = parse_toy_ir(src)
    with pytest.raises(UnknownAccessPatternError):
        extract_memory_accesses(loop)


def test_extract_memory_accesses_rejects_2i():
    src = "LOOP i 0 100 1\nLOAD t1, A[2*i]\nSTORE C[i], t1\nENDLOOP"
    loop = parse_toy_ir(src)
    with pytest.raises(UnknownAccessPatternError):
        extract_memory_accesses(loop)


def test_access_ref_text():
    assert access_ref_text(MemoryAccess("A", "i", 0, False)) == "A[i]"
    assert access_ref_text(MemoryAccess("A", "i", -1, False)) == "A[i-1]"
    assert access_ref_text(MemoryAccess("A", "i", 1, False)) == "A[i+1]"
