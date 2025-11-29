# llm/memory.py
"""
NeuroSync memory module (fast, persistent, semantic).

Features:
- Stores memories as {"id","user","assistant","embedding","ts"} in memory.json
- Uses sentence-transformers/all-MiniLM-L6-v2 for fast embeddings (lazy)
- Thread-safe add/retrieve/clear operations
- Automatic pruning to MAX_MEMORIES
- Safe atomic writes and directory creation
- Backwards-compatible with older memory entries
"""

from __future__ import annotations

import os
import json
import time
import threading
import uuid
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from llm.config import MEMORY_FILE

# lazy import of SentenceTransformer (only when embeddings are used)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # handled when embedding requested

# --------------------
# CONFIG
# --------------------
MEMORY_FILE = os.environ.get("NEUROSYNC_MEMORY_FILE", MEMORY_FILE)
EMBED_MODEL = os.environ.get("NEUROSYNC_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_MEMORIES = int(os.environ.get("NEUROSYNC_MAX_MEMORIES", "800"))
# expected embedding dim for all-MiniLM-L6-v2 (used only for validation)
EXPECTED_EMBED_DIM = 384

# --------------------
# INTERNAL STATE
# --------------------
_lock = threading.RLock()
_embedder = None  # lazy initialized SentenceTransformer instance
_mem_cache: List[Dict[str, Any]] = []
_loaded = False


# --------------------
# UTILITIES
# --------------------
def _now_ts() -> float:
    return time.time()


def _ensure_dir_for(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            # best-effort: if this fails, writes will raise later
            pass


def _atomic_write(path: str, data: str):
    _ensure_dir_for(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            # some platforms may not support fsync on temp files - ignore
            pass
    os.replace(tmp, path)


def _load_file() -> List[Dict[str, Any]]:
    """
    Read memory file; return list. Non-raising (returns empty on error).
    Also performs light migration/validation.
    """
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            arr = json.load(f)
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    for item in arr if isinstance(arr, list) else []:
        if not isinstance(item, dict):
            continue
        # ensure id
        if "id" not in item:
            item["id"] = str(uuid.uuid4())
        # normalize embedding
        emb = item.get("embedding")
        if emb is None:
            item["embedding"] = None
        else:
            try:
                # convert to list of floats or None if shape mismatch
                item["embedding"] = [float(x) for x in emb]
            except Exception:
                item["embedding"] = None
        # ensure ts
        if "ts" not in item:
            item["ts"] = float(item.get("timestamp", _now_ts()))
        out.append(item)
    return out


def _save_file(memories: List[Dict[str, Any]]):
    try:
        _atomic_write(MEMORY_FILE, json.dumps(memories, ensure_ascii=False, indent=2))
    except Exception:
        # fallback best-effort
        try:
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(memories, f, ensure_ascii=False, indent=2)
        except Exception:
            # swallow - persistence failed
            pass


def _prune_if_needed(memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(memories) <= MAX_MEMORIES:
        return memories
    # prune oldest by ts, keep most recent MAX_MEMORIES
    sorted_by_ts = sorted(memories, key=lambda x: x.get("ts", 0.0))
    return sorted_by_ts[-MAX_MEMORIES:]


def _ensure_embedder():
    global _embedder
    if _embedder is None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not available. Install it with `pip install sentence-transformers`")
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _ensure_loaded():
    global _loaded, _mem_cache
    with _lock:
        if _loaded:
            return
        _mem_cache = _load_file()
        # ensure pruning just in case file is huge
        _mem_cache = _prune_if_needed(_mem_cache)
        _save_file(_mem_cache)
        _loaded = True


# --------------------
# PUBLIC API
# --------------------
def load_memory() -> List[Dict[str, Any]]:
    """
    Return a shallow copy of memory list (so callers cannot accidentally mutate internal cache).
    """
    _ensure_loaded()
    with _lock:
        return [dict(m) for m in _mem_cache]


def save_memory(memories: List[Dict[str, Any]]):
    """
    Replace memory store with provided list. This writes atomically and updates in-memory cache.
    """
    global _mem_cache
    with _lock:
        mems = _prune_if_needed([dict(m) for m in memories])
        _mem_cache[:] = mems
        _save_file(_mem_cache)


def add_memory(user_msg: str, assistant_msg: str, metadata: Optional[Dict[str, Any]] = None, embedding: Optional[List[float]] = None):
    """
    Add a memory entry. If embedding is None, attempts to compute it synchronously (best-effort).
    Thread-safe and persists immediately.
    """
    global _mem_cache
    _ensure_loaded()
    with _lock:
        entry: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "user": str(user_msg) if user_msg is not None else "",
            "assistant": str(assistant_msg) if assistant_msg is not None else "",
            "ts": _now_ts(),
            "embedding": None
        }
        # attach metadata if provided (non-sensitive)
        if metadata:
            entry["meta"] = metadata

        if embedding is not None:
            try:
                entry["embedding"] = [float(x) for x in embedding]
            except Exception:
                entry["embedding"] = None
        else:
            # compute embedding if embedder available; otherwise leave None
            try:
                emb_model = _ensure_embedder()
                vec = emb_model.encode([entry["user"]], show_progress_bar=False)[0]
                # convert to list of floats
                arr = np.array(vec, dtype=np.float32)
                entry["embedding"] = [float(x) for x in arr.tolist()]
            except Exception:
                entry["embedding"] = None

        _mem_cache.append(entry)
        # prune + persist
        _mem_cache = _prune_if_needed(_mem_cache)
        _save_file(_mem_cache)


def retrieve_memories(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Semantic search over memories. Returns list of memory dicts (copy) with added 'score' key.
    If embedding model is unavailable, falls back to simple substring match scoring.
    """
    _ensure_loaded()
    with _lock:
        mems = [dict(m) for m in _mem_cache]  # copy

    if not mems:
        return []

    # Try semantic route
    try:
        emb_model = _ensure_embedder()
        q_vec = emb_model.encode([query], show_progress_bar=False)[0]
        q_vec = np.array(q_vec, dtype=np.float32)
    except Exception:
        # fallback keyword match
        ql = query.lower()
        scored = []
        for m in mems:
            text = (m.get("user", "") + " " + m.get("assistant", "")).lower()
            score = 1.0 if ql in text else 0.0
            scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [dict(item[1], score=item[0]) for item in scored[:top_k]]

    # Ensure missing embeddings are computed in a best-effort batch
    missing_idx = [i for i, m in enumerate(mems) if not m.get("embedding")]
    if missing_idx:
        try:
            texts = [mems[i].get("user", "") for i in missing_idx]
            vecs = emb_model.encode(texts, show_progress_bar=False)
            for idx, v in zip(missing_idx, vecs):
                mems[idx]["embedding"] = [float(x) for x in np.array(v, dtype=np.float32).tolist()]
            # persist updated embeddings back to file/cache
            with _lock:
                # update global cache entries (match by id to be safe)
                id_to_idx = {m["id"]: i for i, m in enumerate(_mem_cache)}
                for updated in mems:
                    mid = updated.get("id")
                    if mid in id_to_idx:
                        _mem_cache[id_to_idx[mid]]["embedding"] = updated.get("embedding")
                _mem_cache = _prune_if_needed(_mem_cache)
                _save_file(_mem_cache)
        except Exception:
            # if fill fails, continue with what we have
            pass

    # Build vector matrix from available embeddings
    emb_list = []
    valid_indices = []
    for i, m in enumerate(mems):
        emb = m.get("embedding")
        if emb and isinstance(emb, list) and len(emb) > 0:
            emb_list.append(np.array(emb, dtype=np.float32))
            valid_indices.append(i)

    if not emb_list:
        return []

    mem_matrix = np.vstack(emb_list)
    # normalize
    mem_norms = np.linalg.norm(mem_matrix, axis=1, keepdims=True)
    mem_matrix = mem_matrix / (mem_norms + 1e-8)

    q_norm = np.linalg.norm(q_vec) + 1e-8
    q_vec_n = q_vec / q_norm

    sims = (mem_matrix @ q_vec_n).reshape(-1)  # cosine similarities

    scored: List[Dict[str, Any]] = []
    for sim_val, idx in zip(sims.tolist(), valid_indices):
        item = dict(mems[idx])  # copy
        item["score"] = float(sim_val)
        scored.append(item)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def clear_memory():
    """
    Erase all memory (in-memory + file).
    """
    global _mem_cache
    with _lock:
        _mem_cache = []
        try:
            if os.path.exists(MEMORY_FILE):
                os.remove(MEMORY_FILE)
        except Exception:
            pass


# Convenience: allow bulk import of (user, assistant) pairs
def import_simple_pairs(pairs: List[Tuple[str, str]]):
    for u, a in pairs:
        add_memory(u, a)


# Load cache on module import (lazy)
_ensure_loaded()
