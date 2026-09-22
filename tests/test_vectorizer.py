import pytest

from autovec.ir.parser import parse_c_like
from autovec.ir.serializer import instruction_to_text
from autovec.metrics.performance import compute_metrics
from autovec.vectorize.registers import scalar_to_vector_reg
from autovec.vectorize.vectorizer import VALID_WIDTHS, vectorize_loop


def test_scalar_to_vector_reg():
    assert scalar_to_vector_reg("t1") == "v1"
    assert scalar_to_vector_reg("t42") == "v42"


def test_vectorize_basic_add_matches_spec_example():
    loop = parse_c_like("for (int i = 0; i < 100; i++) { C[i] = A[i] + B[i]; }")
    result = vectorize_loop(loop, 4)
    text = [instruction_to_text(i) for i in result.vector_body]
    assert text == [
        "VLOAD v1, A[i:i+4]", "VLOAD v2, B[i:i+4]", "VADD v3, v1, v2", "VSTORE C[i:i+4], v3",
    ]
    assert result.vector_iterations == 25
    assert result.remainder == 0
    assert not result.symbolic


def test_vectorize_preserves_order_and_shape():
    loop = parse_c_like("for (int i = 0; i < 100; i++) { D[i] = A[i] + B[i] - C[i]; }")
    result = vectorize_loop(loop, 4)
    opcodes = [i.opcode for i in result.vector_body]
    assert opcodes == ["VLOAD", "VLOAD", "VADD", "VLOAD", "VSUB", "VSTORE"]


def test_vectorize_invalid_width_raises():
    loop = parse_c_like("for (int i = 0; i < 100; i++) { C[i] = A[i] + B[i]; }")
    with pytest.raises(ValueError):
        vectorize_loop(loop, 3)


@pytest.mark.parametrize("width", VALID_WIDTHS)
def test_vectorize_accepts_all_valid_widths(width):
    loop = parse_c_like("for (int i = 0; i < 100; i++) { C[i] = A[i] + B[i]; }")
    result = vectorize_loop(loop, width)
    assert result.width == width


def test_metrics_example_1_exact():
    loop = parse_c_like("for (int i = 0; i < 100; i++) { C[i] = A[i] + B[i]; }")
    vec_result = vectorize_loop(loop, 4)
    metrics = compute_metrics(loop, vec_result, 4)
    assert metrics.scalarInstructions == 400
    assert metrics.vectorInstructions == 100
    assert metrics.instructionReduction == 75.0
    assert metrics.vectorIterations == 25
    assert metrics.remainderIterations == 0


def test_symbolic_bound_metrics_are_none():
    loop = parse_c_like("for (int i = 0; i < N; i++) { C[i] = A[i] + B[i]; }")
    vec_result = vectorize_loop(loop, 4)
    assert vec_result.symbolic
    assert compute_metrics(loop, vec_result, 4) is None


def test_zero_iterations():
    loop = parse_c_like("for (int i = 0; i < 0; i++) { C[i] = A[i] + B[i]; }")
    vec_result = vectorize_loop(loop, 4)
    assert vec_result.vector_iterations == 0
    assert vec_result.remainder == 0
    metrics = compute_metrics(loop, vec_result, 4)
    assert metrics.scalarInstructions == 0
    assert metrics.vectorInstructions == 0


def test_n_smaller_than_width_entirely_scalar():
    loop = parse_c_like("for (int i = 0; i < 3; i++) { C[i] = A[i] + B[i]; }")
    vec_result = vectorize_loop(loop, 4)
    assert vec_result.vector_iterations == 0
    assert vec_result.remainder == 3


def test_exact_multiple_no_remainder():
    loop = parse_c_like("for (int i = 0; i < 16; i++) { C[i] = A[i] + B[i]; }")
    vec_result = vectorize_loop(loop, 4)
    assert vec_result.vector_iterations == 4
    assert vec_result.remainder == 0


def test_remainder_case():
    loop = parse_c_like("for (int i = 0; i < 18; i++) { C[i] = A[i] + B[i]; }")
    vec_result = vectorize_loop(loop, 4)
    assert vec_result.vector_iterations == 4
    assert vec_result.remainder == 2
