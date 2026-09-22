from autovec.analysis.legality import VectorizationLegalityChecker
from autovec.ir.parser import parse_c_like, parse_toy_ir


def test_no_dependency_for_independent_arrays():
    loop = parse_c_like("for (int i = 0; i < 100; i++) { C[i] = A[i] + B[i]; }")
    result = VectorizationLegalityChecker().check(loop)
    assert result.legal
    assert result.dependencies == []
    assert result.reasons == []


def test_raw_dependency_detected():
    loop = parse_c_like("for (int i = 1; i < 100; i++) { A[i] = A[i-1] + 1; }")
    result = VectorizationLegalityChecker().check(loop)
    assert not result.legal
    assert len(result.dependencies) == 1
    dep = result.dependencies[0]
    assert dep.type == "RAW"
    assert dep.distance == 1
    assert dep.source == "A[i-1]"
    assert dep.target == "A[i]"
    assert any("RAW" in r and "distance 1" in r for r in result.reasons)


def test_war_dependency_for_array_shift():
    loop = parse_c_like("for (int i = 0; i < 100; i++) { A[i] = A[i+1]; }")
    result = VectorizationLegalityChecker().check(loop)
    assert not result.legal
    assert len(result.dependencies) == 1
    dep = result.dependencies[0]
    assert dep.type == "WAR"
    assert dep.distance == 1
    assert dep.source == "A[i+1]"
    assert dep.target == "A[i]"


def test_same_offset_read_write_is_safe():
    # Same iteration, same location: A is read and written at offset 0 only.
    loop = parse_c_like("for (int i = 0; i < 100; i++) { A[i] = A[i]; }")
    result = VectorizationLegalityChecker().check(loop)
    assert result.legal
    assert result.dependencies == []


def test_different_arrays_never_conflict():
    loop = parse_c_like("for (int i = 0; i < 100; i++) { D[i] = A[i] + B[i] - C[i]; }")
    result = VectorizationLegalityChecker().check(loop)
    assert result.legal
    assert result.dependencies == []


def test_unknown_access_pattern_rejected():
    src = "LOOP i 0 100 1\nLOAD t1, A[j]\nSTORE C[i], t1\nENDLOOP"
    loop = parse_toy_ir(src)
    result = VectorizationLegalityChecker().check(loop)
    assert not result.legal
    assert any("Unknown memory access pattern" in r for r in result.reasons)


def test_non_unit_stride_rejected():
    src = "LOOP i 0 100 2\nLOAD t1, A[i]\nSTORE C[i], t1\nENDLOOP"
    loop = parse_toy_ir(src)
    result = VectorizationLegalityChecker().check(loop)
    assert not result.legal
    assert any("stride" in r.lower() for r in result.reasons)


def test_negative_step_rejected():
    src = "LOOP i 99 0 -1\nLOAD t1, A[i]\nSTORE C[i], t1\nENDLOOP"
    loop = parse_toy_ir(src)
    result = VectorizationLegalityChecker().check(loop)
    assert not result.legal
    assert any("negative" in r.lower() for r in result.reasons)
