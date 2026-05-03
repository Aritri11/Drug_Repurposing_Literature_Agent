import sys, os, time, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Optional, AsyncGenerator
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pipeline import build_pipeline
from shared.config import warmup_models, free_reasoning_model
import asyncio, queue, threading

app = FastAPI(title="Drug Repurposing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RunRequest(BaseModel):
    disease: str
    max_results: Optional[int] = 300

@app.get("/")
def health():
    return {"status": "ok", "message": "Drug Repurposing API running"}

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

@app.post("/run")
async def run_pipeline(req: RunRequest):
    """
    Streams pipeline progress as Server-Sent Events (SSE).
    The frontend receives real-time log lines and the final report.
    """
    log_queue: queue.Queue = queue.Queue()
    result_container = {}

    class QueueStream:
        """Redirect stdout into the queue so pipeline prints become SSE events."""
        def __init__(self, original):
            self.original = original
        def write(self, text):
            if text.strip():
                log_queue.put({"type": "log", "text": text.rstrip()})
            self.original.write(text)
        def flush(self):
            self.original.flush()

    def run_in_thread():
        old_stdout = sys.stdout
        sys.stdout = QueueStream(old_stdout)
        try:
            warmup_models()
            free_reasoning_model()
            pipeline = build_pipeline()
            final_state = pipeline.invoke({
                "disease":         req.disease,
                "max_results":     req.max_results,
                "abstracts":       [],
                "disease_genes":   [],
                "drug_candidates": [],
                "kg_status":       "pending",
                "final_report":    ""
            })
            result_container["state"] = final_state
        except Exception as e:
            result_container["error"] = str(e)
        finally:
            sys.stdout = old_stdout
            log_queue.put({"type": "done"})

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()

    async def event_generator() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        while True:
            try:
                item = await loop.run_in_executor(None, lambda: log_queue.get(timeout=500))
            except Exception:
                yield f"data: {json.dumps({'type':'error','text':'Timeout waiting for pipeline'})}\n\n"
                break

            if item["type"] == "done":
                # Send final result
                if "error" in result_container:
                    yield f"data: {json.dumps({'type':'error','text':result_container['error']})}\n\n"
                else:
                    state = result_container.get("state", {})
                    payload = {
                        "type": "result",
                        "disease": req.disease,
                        "abstracts": len(state.get("abstracts", [])),
                        "disease_genes": len(state.get("disease_genes", [])),
                        "drug_candidates": len(state.get("drug_candidates", [])),
                        "kg_status": state.get("kg_status", "unknown"),
                        "final_report": state.get("final_report", ""),
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                break
            else:
                yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )