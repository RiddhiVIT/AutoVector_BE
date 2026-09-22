# Toy Auto-Vectorization Pass

A small, from-scratch compiler that takes a restricted, count-controlled
single-basic-block loop and determines whether it can be legally
transformed into SIMD-style vector instructions -- plus a FastAPI HTTP API
wrapping it for a Next.js frontend.

```
Scalar                              Vectorized (width 4)
for i = 0 to N-1:                   for i = 0 to N-4 step 4:
    t1 = load A[i]                      v1 = vload A[i:i+4]
    t2 = load B[i]                      v2 = vload B[i:i+4]
    t3 = add t1, t2                     v3 = vadd v1, v2
    store C[i], t3                      vstore C[i:i+4], v3
```

## Project layout

```
backend/
├── src/autovec/
│   ├── ir/            parser (toy-ir / JSON / C-like), IR data model, serializer
│   ├── analysis/       memory-access extraction, dependency analysis, legality checker
│   ├── vectorize/      scalar -> vector transformation, register renaming, tail handling
│   ├── execution/      scalar and vector interpreters, correctness verification
│   ├── metrics/         instruction-count metrics
│   ├── diagnostics/    pipeline orchestration + human-readable reporting
│   ├── cli.py           `autovec` / `python -m autovec` command line
│   └── api.py            FastAPI app (`/api/health`, `/api/analyze`)
├── examples/            the six example .ir programs
└── tests/                pytest suite
```

## IR data model

```python
@dataclass
class Instruction:
    opcode: str                 # LOAD, STORE, ADD, SUB, MUL, DIV, CONST,
                                 # VLOAD, VSTORE, VADD, VSUB, VMUL, VDIV, VBROADCAST
    destination: str | None
    operands: list

@dataclass
class MemoryAccess:
    array: str
    index: str          # the induction variable name, e.g. "i"
    offset: int          # A[i] -> 0, A[i-1] -> -1, A[i+1] -> 1
    is_write: bool

@dataclass
class Loop:
    induction_variable: str
    start: int
    end: int | str        # int literal (concrete) OR symbolic name
    step: int
    body: list[Instruction]
```

## Input formats

### 1. Toy IR text

```
LOOP i 0 N 1
LOAD t1, A[i]
LOAD t2, B[i]
ADD t3, t1, t2
STORE C[i], t3
ENDLOOP
```

`LOOP <var> <start> <end> <step>` where `<end>` may be an integer literal or
a symbolic identifier. Commas after the first token on each instruction
line are optional (`LOAD t1, A[i]` and `LOAD t1 A[i]` both parse).

### 2. Restricted C-like

```c
for (int i = 0; i < 100; i++) {
    C[i] = A[i] + B[i];
}
```

One statement per line, `<ArrayName>[<index-expr>] = <expr>;`, where
`<index-expr>` is `i`, `i+<int>`, or `i-<int>`, and `<expr>` is a
left-to-right chain of `<ArrayName>[<index-expr>]` or integer-literal terms
separated by `+ - * /` (no operator precedence -- this is a toy language).
`<end>` may be an integer literal or a bare symbolic identifier (e.g. `N`).

### 3. Raw JSON

```json
{
  "loop": {"variable": "i", "start": 0, "end": "N", "step": 1},
  "body": [
    {"op": "load", "dst": "t1", "array": "A", "index": "i"},
    {"op": "add", "dst": "t3", "lhs": "t1", "rhs": "t2"},
    {"op": "store", "array": "C", "index": "i", "value": "t3"}
  ]
}
```

## Architecture pipeline

```
source text
    │  ir/parser.py  (toy-ir | c-like | json)
    ▼
Loop IR
    │  analysis/memory.py     -> MemoryAccess list (or UnknownAccessPatternError)
    │  analysis/dependency.py -> loop-carried RAW/WAR/WAW dependencies
    │  analysis/legality.py   -> LegalityResult(legal, reasons, warnings)
    ▼
legal? ──no──> diagnostics/reporter.py renders a diagnostic block, stop
    │yes
    ▼
vectorize/vectorizer.py  (packing.py + registers.py + tail.py)
    ▼
VectorizationResult (vector_body, vector_iterations, remainder, tail)
    │  metrics/performance.py -> Metrics(scalarInstructions, vectorInstructions, ...)
    │  execution/scalar.py + execution/vector.py -> verify() CORRECT/MISMATCH
    ▼
diagnostics/reporter.py -> human-readable report / structured log
```

`diagnostics/reporter.run_pipeline()` is the single entry point used by both
the CLI (`cli.py`) and the FastAPI app (`api.py`), so both surfaces stay in
sync automatically.

## Legality rules (implemented exactly)

- Non-unit stride (`step != 1`) or negative step -> illegal.
- Any memory reference whose index isn't exactly `i`, `i+const`, or
  `i-const` -> conservative rejection ("Unknown memory access pattern.
  Unable to prove iteration independence.").
- Same array, differing offsets, at least one write -> loop-carried
  RAW/WAR/WAW dependency, illegal. Reported with the dependency distance
  and a human-readable reason, e.g.
  `"Loop-carried dependency detected: A[i] depends on A[i-1] (RAW, distance 1)"`.
- Same array, same offset (including offset 0 read+write) -> intra-iteration,
  safe.
- Different array names -> assumed non-aliasing (MVP alias rule,
  `analysis/alias.py`).

## Install

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # installs autovec + fastapi + uvicorn
pip install -r requirements.txt   # also pulls in pytest/httpx for the test suite
```

## CLI usage

```bash
# bare form: python -m autovec <file> [options]
python -m autovec examples/basic_add.ir --vector-width 4 --verify --verbose

# subcommands: autovec analyze|vectorize|verify|benchmark <file> --width N
autovec analyze examples/basic_add.ir --width 4
autovec vectorize examples/basic_add.ir --width 4 --output out.txt
autovec verify examples/basic_add.ir --width 4
autovec benchmark examples/remainder.ir --width 4
```

Options: `--vector-width/--width` (default 4, must be 2/4/8/16),
`--input-type` (`c-like`/`toy-ir`/`json`, auto-detected from the file
extension/content by default), `--output <file>`, `--report` (emit JSON
instead of the human-readable report), `--verify` (run the scalar-vs-vector
correctness check), `--verbose`/`--debug` (print `[Parser]`, `[Analysis]`,
`[Dependency]`, `[Legality]`, `[Vectorizer]`, `[Tail]`, `[Verify]`,
`[Metrics]` pipeline log lines). Exit code is `0` on success, `1` when the
loop is rejected, `2` on a parse/IO error.

## API server

```bash
uvicorn autovec.api:app --reload --port 8000
```

- `GET /api/health` -> `{"status": "ok"}`
- `POST /api/analyze` -> `{source, vectorWidth, inputType}` request,
  see `src/autovec/api.py` for the exact response contract (dependencies,
  memory accesses, before/after instruction listings, metrics, and a
  structured pipeline log). CORS is enabled for `http://localhost:3000`.
  Malformed input never 500s -- it returns HTTP 200 with
  `{"success": false, "error": "..."}`.

## Tests

```bash
pytest
```

80 tests covering: parser round-trips (toy-ir, C-like, JSON), the full
vectorizability matrix (all six example programs, N=0, N<W, N==k*W,
N%W!=0, unknown-index rejection), tail/remainder handling, scalar/vector
interpreter correctness (`verify()` == CORRECT on every vectorizable
example), and FastAPI end-to-end tests asserting the exact contract
numbers for example #1 (400/100/75.0/25/0) and example #6 (tail=2).

## The six example programs (`examples/`)

| File | Program | Result |
|---|---|---|
| `basic_add.ir` | `C[i] = A[i] + B[i]` | VECTORIZABLE |
| `multiplication.ir` | `C[i] = A[i] * B[i]` | VECTORIZABLE |
| `independent_arrays.ir` | `D[i] = A[i] + B[i] - C[i]` | VECTORIZABLE |
| `dependency.ir` | `A[i] = A[i-1] + 1` | REJECTED (RAW, distance 1) |
| `array_shift.ir` | `A[i] = A[i+1]` | REJECTED (WAR, distance 1) |
| `remainder.ir` | `C[i] = A[i] + B[i]`, N=10, W=4 | VECTORIZABLE, tail=2 |

## Non-goals / limitations

- No nested loops, no function calls, no pointer arithmetic.
- No full alias analysis -- arrays are assumed non-aliasing purely by
  having distinct names; there is no pointer syntax to alias in the first
  place.
- No real machine code generation -- vector "instructions" are a toy IR;
  the vector interpreter simulates SIMD by applying the scalar op `width`
  times, not real hardware vector instructions.
- Unit stride only (`step == 1`); non-unit and negative strides are
  rejected outright.
- No reduction support (scalar accumulation across iterations) -- out of
  scope for the parser, since the grammar only supports array-indexed
  statements.
- Symbolic loop bounds (`end` is an identifier like `N`) still produce
  structurally correct vector IR, but concrete metrics (instruction counts,
  vector/remainder iteration counts) are omitted since they require a
  known N.

## Future work

- Reduction idioms (`sum += A[i]`) via horizontal-add lowering.
- Masked/predicated tails instead of a separate scalar remainder loop.
- A real cost model (latency/throughput per opcode) instead of raw
  instruction counts.
- Interprocedural / multi-basic-block alias analysis for real pointer
  inputs.
