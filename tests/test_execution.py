import pytest

from autovec.execution.scalar import run_scalar
from autovec.execution.vector import generate_test_arrays, run_vector, verify
from autovec.ir.parser import parse_c_like
from autovec.vectorize.vectorizer import vectorize_loop

EXAMPLES = [
    "for (int i = 0; i < 100; i++) { C[i] = A[i] + B[i]; }",
    "for (int i = 0; i < 100; i++) { C[i] = A[i] * B[i]; }",
    "for (int i = 0; i < 100; i++) { D[i] = A[i] + B[i] - C[i]; }",
    "for (int i = 0; i < 10; i++) { C[i] = A[i] + B[i]; }",
]


@pytest.mark.parametrize("src", EXAMPLES)
def test_scalar_and_vector_interpreters_agree(src):
    loop = parse_c_like(src)
    vec_result = vectorize_loop(loop, 4)
    n_iters = vec_result.vector_iterations * 4 + vec_result.remainder

    base = generate_test_arrays(loop, n_iters)

    import copy
    scalar_arrays = copy.deepcopy(base)
    run_scalar(loop, scalar_arrays, n_iters)

    vector_arrays = copy.deepcopy(base)
    run_vector(loop, vec_result, vector_arrays)

    for name in base:
        assert scalar_arrays[name][:n_iters + loop.start] == vector_arrays[name][:n_iters + loop.start]


@pytest.mark.parametrize("src", EXAMPLES)
def test_verify_reports_correct(src):
    loop = parse_c_like(src)
    vec_result = vectorize_loop(loop, 4)
    result = verify(loop, vec_result)
    assert result.status == "CORRECT"
    assert result.mismatches == []


def test_scalar_interpreter_addition():
    loop = parse_c_like("for (int i = 0; i < 5; i++) { C[i] = A[i] + B[i]; }")
    arrays = {"A": [1, 2, 3, 4, 5, 0], "B": [10, 20, 30, 40, 50, 0], "C": [0] * 6}
    run_scalar(loop, arrays, 5)
    assert arrays["C"][:5] == [11, 22, 33, 44, 55]


def test_scalar_interpreter_dependency_chain():
    # A[i] = A[i-1] + 1, starting A[0]=0 -> A should become 0,1,2,3,4
    loop = parse_c_like("for (int i = 1; i < 5; i++) { A[i] = A[i-1] + 1; }")
    arrays = {"A": [0, 0, 0, 0, 0]}
    run_scalar(loop, arrays, 4)
    assert arrays["A"] == [0, 1, 2, 3, 4]
