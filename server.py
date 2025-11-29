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
app = FastAPI(title="NeuroSync LLM Server — Stable v3")

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
    """
    Dashboard listens on: POST /dashboard_event
    So we forward event-text → dashboard queue.
    """
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
    """Compact focus+emotion payloads to avoid context explosion."""
    compact = {}

    # Focus trend
    arr = data.get("focus_trend")
    if isinstance(arr, list) and len(arr) > 0:
        arr = arr[-120:]
        avg = sum(arr) / len(arr)
        compact["focus_summary"] = {
            "avg": avg,
            "max": max(arr),
            "min": min(arr),
        }

    # Fatigue
    arr2 = data.get("fatigue_curve")
    if isinstance(arr2, list) and len(arr2) > 0:
        arr2 = arr2[-120:]
        compact["fatigue_summary"] = {
            "avg": sum(arr2) / len(arr2),
            "max": max(arr2),
        }

    # direct fields
    for k in ["focus", "fatigue", "blink", "gaze", "gaze_off", "head_angle", "emotion", "trend", "drifts"]:
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
            if cid in self.clients:
                try:
                    del self.clients[cid]
                except:
                    pass

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
# HEALTH
# ============================================================
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": True}


# ============================================================
#  EVENT HANDLER (FINAL)
# ============================================================
@app.post("/event")
async def receive_event(payload: EventPayload):
    """
    Handles:
        - focus_drop
        - distraction
        - emotion_shift
    """

    event_type = payload.event_type
    client_id = payload.session_id or payload.user_id
    compact = compress_payload(payload.data or {})

    # capture event-loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None  # extremely rare fallback

    # -----------------------------
    # Worker thread (NO async inside)
    # -----------------------------
    def worker():
        try:
            # Main LLM logic
            result = route_event(event_type, compact)

            # Store memory
            try:
                add_memory(str(payload.dict()), result)
            except:
                pass

            # Push to dashboard (HTTP → queue)
            push_dashboard(result)

            # If websocket client exists → schedule send
            if client_id and loop:
                asyncio.run_coroutine_threadsafe(ws_manager.send(client_id, result), loop)

        except Exception:
            traceback.print_exc()

    # Start thread (non-blocking)
    threading.Thread(target=worker, daemon=True).start()

    return {"status": "queued", "event_type": event_type}


# ============================================================
# CHAT
# ============================================================
@app.post("/chat")
async def chat(payload: ChatPayload):
    prompt = f"User: {payload.message}\nReply briefly."

    def sync_call():
        out = llm(prompt, max_tokens=payload.max_tokens or 200, temperature=0.7)
        if "choices" in out:
            c = out["choices"][0]
            return c.get("text") or c.get("message", {}).get("content", "")
        return ""

    reply = await run_blocking(sync_call)
    reply = reply.strip()

    try: add_memory(payload.message, reply)
    except: pass

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
# WEBSOCKET
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
    except:
        ws_manager.disconnect(cid)


# ============================================================
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8765, reload=False)
