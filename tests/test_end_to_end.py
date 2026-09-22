from fastapi.testclient import TestClient

from autovec.api import app

client = TestClient(app)


def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_analyze_example_1_exact_metrics():
    resp = client.post("/api/analyze", json={
        "source": "for (int i = 0; i < 100; i++) { C[i] = A[i] + B[i]; }",
        "vectorWidth": 4,
        "inputType": "c-like",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["vectorizable"] is True
    assert data["vectorWidth"] == 4
    assert data["analysis"]["dependencies"] == []
    assert data["analysis"]["memoryAccesses"] == [
        {"array": "A", "index": "i", "offset": 0, "isWrite": False},
        {"array": "B", "index": "i", "offset": 0, "isWrite": False},
        {"array": "C", "index": "i", "offset": 0, "isWrite": True},
    ]
    assert data["transformation"]["before"] == [
        "LOAD t1, A[i]", "LOAD t2, B[i]", "ADD t3, t1, t2", "STORE C[i], t3",
    ]
    assert data["transformation"]["after"] == [
        "VLOAD v1, A[i:i+4]", "VLOAD v2, B[i:i+4]", "VADD v3, v1, v2", "VSTORE C[i:i+4], v3",
    ]
    m = data["metrics"]
    assert m["scalarInstructions"] == 400
    assert m["vectorInstructions"] == 100
    assert m["instructionReduction"] == 75.0
    assert m["vectorIterations"] == 25
    assert m["remainderIterations"] == 0
    assert any("[Parser]" in l for l in data["log"])
    assert any("[Legality]" in l for l in data["log"])


def test_analyze_example_2_multiplication():
    resp = client.post("/api/analyze", json={
        "source": "for (int i = 0; i < 100; i++) { C[i] = A[i] * B[i]; }",
        "vectorWidth": 4,
        "inputType": "c-like",
    })
    data = resp.json()
    assert data["vectorizable"] is True
    assert "VMUL v3, v1, v2" in data["transformation"]["after"]


def test_analyze_example_3_independent_arrays():
    resp = client.post("/api/analyze", json={
        "source": "for (int i = 0; i < 100; i++) { D[i] = A[i] + B[i] - C[i]; }",
        "vectorWidth": 4,
        "inputType": "c-like",
    })
    data = resp.json()
    assert data["vectorizable"] is True
    assert data["metrics"]["scalarInstructions"] == 600
    assert data["metrics"]["vectorInstructions"] == 150


def test_analyze_example_4_rejected_raw_dependency():
    resp = client.post("/api/analyze", json={
        "source": "for (int i = 1; i < 100; i++) { A[i] = A[i-1] + 1; }",
        "vectorWidth": 4,
        "inputType": "c-like",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["vectorizable"] is False
    assert data["metrics"] is None
    assert data["transformation"]["after"] == []
    deps = data["analysis"]["dependencies"]
    assert len(deps) == 1
    assert deps[0] == {"type": "RAW", "distance": 1, "source": "A[i-1]", "target": "A[i]"}


def test_analyze_example_5_rejected_array_shift():
    resp = client.post("/api/analyze", json={
        "source": "for (int i = 0; i < 100; i++) { A[i] = A[i+1]; }",
        "vectorWidth": 4,
        "inputType": "c-like",
    })
    data = resp.json()
    assert data["vectorizable"] is False
    assert len(data["analysis"]["dependencies"]) == 1


def test_analyze_example_6_remainder_tail():
    resp = client.post("/api/analyze", json={
        "source": "for (int i = 0; i < 10; i++) { C[i] = A[i] + B[i]; }",
        "vectorWidth": 4,
        "inputType": "c-like",
    })
    data = resp.json()
    assert data["vectorizable"] is True
    assert data["metrics"]["vectorIterations"] == 2
    assert data["metrics"]["remainderIterations"] == 2


def test_analyze_invalid_source_returns_success_false_not_500():
    resp = client.post("/api/analyze", json={
        "source": "this is not valid syntax at all {{{",
        "vectorWidth": 4,
        "inputType": "c-like",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert "error" in data


def test_analyze_invalid_width_returns_success_false():
    resp = client.post("/api/analyze", json={
        "source": "for (int i = 0; i < 100; i++) { C[i] = A[i] + B[i]; }",
        "vectorWidth": 3,
        "inputType": "c-like",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False


def test_analyze_toy_ir_input_type():
    src = "LOOP i 0 100 1\nLOAD t1, A[i]\nLOAD t2, B[i]\nADD t3, t1, t2\nSTORE C[i], t3\nENDLOOP"
    resp = client.post("/api/analyze", json={
        "source": src, "vectorWidth": 4, "inputType": "toy-ir",
    })
    data = resp.json()
    assert data["vectorizable"] is True
    assert data["metrics"]["scalarInstructions"] == 400


def test_analyze_json_input_type():
    import json
    payload = {
        "loop": {"variable": "i", "start": 0, "end": 100, "step": 1},
        "body": [
            {"op": "load", "dst": "t1", "array": "A", "index": "i"},
            {"op": "load", "dst": "t2", "array": "B", "index": "i"},
            {"op": "add", "dst": "t3", "lhs": "t1", "rhs": "t2"},
            {"op": "store", "array": "C", "index": "i", "value": "t3"},
        ],
    }
    resp = client.post("/api/analyze", json={
        "source": json.dumps(payload), "vectorWidth": 4, "inputType": "json",
    })
    data = resp.json()
    assert data["vectorizable"] is True
    assert data["metrics"]["scalarInstructions"] == 400
