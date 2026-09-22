"""FastAPI app exposing the auto-vectorization pipeline to the Next.js
frontend. See README.md for the full API contract.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .diagnostics.reporter import run_pipeline
from .ir.parser import ParseError

app = FastAPI(title="Toy Auto-Vectorization API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    source: str
    vectorWidth: int = 4
    inputType: Literal["c-like", "toy-ir", "json"] = "c-like"


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    try:
        result = run_pipeline(req.source, req.inputType, req.vectorWidth)
    except (ParseError, ValueError) as e:
        return {"success": False, "error": str(e)}
    except Exception as e:  # pragma: no cover - genuine server bug guard
        return {"success": False, "error": f"Internal error: {e}"}

    legality = result.legality

    if legality.legal:
        reason = "No loop-carried dependencies detected."
    elif legality.dependencies:
        reason = "Loop-carried dependency detected."
    elif legality.reasons:
        reason = legality.reasons[0]
    else:
        reason = "Vectorization rejected."

    dependencies = [
        {"type": d.type, "distance": d.distance, "source": d.source, "target": d.target}
        for d in legality.dependencies
    ]
    memory_accesses = [
        {"array": m.array, "index": m.index, "offset": m.offset, "isWrite": m.is_write}
        for m in result.memory_accesses
    ]

    metrics = None
    if result.metrics is not None:
        m = result.metrics
        metrics = {
            "scalarInstructions": m.scalarInstructions,
            "vectorInstructions": m.vectorInstructions,
            "instructionReduction": m.instructionReduction,
            "vectorIterations": m.vectorIterations,
            "remainderIterations": m.remainderIterations,
        }

    return {
        "success": True,
        "vectorizable": legality.legal,
        "vectorWidth": req.vectorWidth,
        "analysis": {
            "dependencies": dependencies,
            "memoryAccesses": memory_accesses,
            "reason": reason,
        },
        "transformation": {
            "before": result.before,
            "after": result.after,
        },
        "metrics": metrics,
        "log": result.log,
    }
