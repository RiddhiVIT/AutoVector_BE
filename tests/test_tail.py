from autovec.ir.parser import parse_c_like
from autovec.vectorize.tail import build_tail
from autovec.vectorize.vectorizer import vectorize_loop


def test_build_tail_none_when_no_remainder():
    loop = parse_c_like("for (int i = 0; i < 16; i++) { C[i] = A[i] + B[i]; }")
    assert build_tail(loop.body, start_index=16, remainder=0) is None


def test_build_tail_carries_scalar_body():
    loop = parse_c_like("for (int i = 0; i < 10; i++) { C[i] = A[i] + B[i]; }")
    tail = build_tail(loop.body, start_index=8, remainder=2)
    assert tail is not None
    assert tail.remainder == 2
    assert tail.start_index == 8
    assert tail.instructions == loop.body


def test_vectorizer_wires_tail_for_example_6():
    loop = parse_c_like("for (int i = 0; i < 10; i++) { C[i] = A[i] + B[i]; }")
    result = vectorize_loop(loop, 4)
    assert result.vector_iterations == 2
    assert result.remainder == 2
    assert result.tail is not None
    assert result.tail.remainder == 2
    assert result.tail.start_index == 8  # 2 vector iterations * width 4


def test_no_tail_for_exact_multiple():
    loop = parse_c_like("for (int i = 0; i < 100; i++) { C[i] = A[i] + B[i]; }")
    result = vectorize_loop(loop, 4)
    assert result.tail is None
