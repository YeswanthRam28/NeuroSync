# server.py
import asyncio
import threading
import traceback
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import requests

# -----------------------------------------
# Local modules
# -----------------------------------------
from llm.router import route_event
from llm.memory import load_memory, save_memory, add_memory, retrieve_memories
from llm.loader import llm


# ============================================================
# FASTAPI INITIALIZATION
# ============================================================
app = FastAPI(title="NeuroSync LLM Server — Stable v4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Pydantic models
# ============================================================
class EventPayload(BaseModel):
    event_type: str
    data: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class ChatPayload(BaseModel):
    message: str
    max_tokens: Optional[int] = 200


class MemoryQuery(BaseModel):
    query: str
    top_k: Optional[int] = 3


# ============================================================
# DASHBOARD PUSH BRIDGE
# ============================================================
def push_dashboard(msg: str):
    """Forward event text to dashboard queue."""
    try:
        requests.post(
            "http://127.0.0.1:8501/dashboard_event",
            json={"msg": msg},
            timeout=1
        )
    except:
        pass


# ============================================================
# HELPERS
# ============================================================
def compress_payload(data: Dict[str, Any]):
    """Compact focus/emotion payloads."""
    compact = {}
    arr = data.get("focus_trend")
    if isinstance(arr, list) and arr:
        arr = arr[-120:]
        compact["focus_summary"] = {
            "avg": sum(arr) / len(arr),
            "max": max(arr),
            "min": min(arr),
        }

    arr2 = data.get("fatigue_curve")
    if isinstance(arr2, list) and arr2:
        arr2 = arr2[-120:]
        compact["fatigue_summary"] = {
            "avg": sum(arr2) / len(arr2),
            "max": max(arr2),
        }

    for k in ["focus", "fatigue", "blink", "gaze", "gaze_off", "head_angle",
              "emotion", "trend", "drifts"]:
        if k in data:
            compact[k] = data[k]

    return compact


# ============================================================
# WEBSOCKET MANAGER
# ============================================================
class WSManager:
    def __init__(self):
        self.clients = {}
        self._lock = threading.Lock()

    async def connect(self, cid: str, ws: WebSocket):
        await ws.accept()
        with self._lock:
            self.clients[cid] = ws

    def disconnect(self, cid: str):
        with self._lock:
            self.clients.pop(cid, None)

    async def send(self, cid: str, text: str):
        with self._lock:
            ws = self.clients.get(cid)
        if not ws:
            return
        try:
            await ws.send_text(text)
        except:
            pass


ws_manager = WSManager()


# ============================================================
# THREADSAFE LLM EXECUTION
# ============================================================
async def run_blocking(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


# ============================================================
# HEALTH CHECK
# ============================================================
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": True}


# ============================================================
# EVENT ENDPOINT
# ============================================================
@app.post("/event")
async def receive_event(payload: EventPayload):
    event_type = payload.event_type
    client_id = payload.session_id or payload.user_id
    compact = compress_payload(payload.data or {})

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    def worker():
        try:
            # Actual reasoning
            result = route_event(event_type, compact)

            # Memory
            try:
                add_memory(str(payload.dict()), result)
            except:
                pass

            # Push to dashboard
            push_dashboard(result)

            # Push to websocket if connected
            if client_id and loop:
                asyncio.run_coroutine_threadsafe(
                    ws_manager.send(client_id, result),
                    loop
                )

        except Exception:
            traceback.print_exc()

    threading.Thread(target=worker, daemon=True).start()

    return {
        "status": "queued",
        "type": "final",
        "text": f"Event received: {event_type}",
        "event_type": event_type
    }


# ============================================================
# CHAT ENDPOINT (GEMMA FIX APPLIED)
# ============================================================
def extract_text(out: dict) -> str:
    """
    Safe extractor that supports:
        - Gemma chat_format
        - Llama text responses
        - Any fallback format
    """
    try:
        choice = out["choices"][0]

        # Gemma chat format
        if "message" in choice and "content" in choice["message"]:
            return choice["message"]["content"]

        # Llama style
        if "text" in choice and choice["text"].strip():
            return choice["text"]

        return str(choice)
    except:
        return ""


@app.post("/chat")
async def chat(payload: ChatPayload):

    # Proper Gemma conversation format
    prompt = (
        "<start_of_turn>user\n"
        + payload.message +
        "\n<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

    def sync_call():
        out = llm(
            prompt,
            max_tokens=payload.max_tokens or 200,
            temperature=0.7
        )
        reply = extract_text(out)
        return reply.strip()

    reply = await run_blocking(sync_call)

    if not reply:
        reply = "I'm here, but I couldn't generate a reply."

    try:
        add_memory(payload.message, reply)
    except:
        pass

    return {"reply": reply}


# ============================================================
# MEMORY ENDPOINTS
# ============================================================
@app.post("/memory/query")
async def memory_query(payload: MemoryQuery):
    return {"results": retrieve_memories(payload.query, top_k=payload.top_k or 3)}


@app.get("/memory")
async def memory_dump():
    mem = load_memory()
    return {"count": len(mem), "memory": mem}


@app.post("/memory/clear")
async def memory_clear():
    save_memory([])
    return {"status": "cleared"}


# ============================================================
# WEBSOCKET ENDPOINT
# ============================================================
@app.websocket("/ws/{cid}")
async def ws_endpoint(ws: WebSocket, cid: str):
    await ws_manager.connect(cid, ws)

    try:
        while True:
            msg = await ws.receive_text()
            await ws_manager.send(cid, f"pong: {msg}")

    except WebSocketDisconnect:
        ws_manager.disconnect(cid)
    except Exception:
        ws_manager.disconnect(cid)


# ============================================================
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8765, reload=False)
