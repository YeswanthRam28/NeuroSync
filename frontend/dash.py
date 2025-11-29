# frontend/dash.py
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import numpy as np
import joblib
import os
import traceback
import time
import threading
import queue
import json
from typing import Optional
from websocket import WebSocketApp

BACKEND_BASE = "http://127.0.0.1:8000"
LLM_BASE = "http://127.0.0.1:8765"
WS_URL = "ws://127.0.0.1:8765/ws/dashboard"

st.set_page_config(page_title="🧠 NeuroSync Dashboard", layout="wide")

def safe_json(resp):
    try:
        return resp.json()
    except Exception:
        return {}

tabs = st.tabs(["📊 Dashboard", "🤖 Assistant Chat"])
tab_dashboard, tab_chat = tabs

MODEL_PATH = os.path.join("..", "ml", "model.pkl")
LABEL_PATH = os.path.join("..", "ml", "label_encoder.pkl")

@st.cache_resource
def load_ml_artifacts():
    clf = None
    le = None
    err = None
    try:
        if os.path.exists(MODEL_PATH):
            clf = joblib.load(MODEL_PATH)
        else:
            err = f"Model file not found at {MODEL_PATH}"
        if os.path.exists(LABEL_PATH):
            le = joblib.load(LABEL_PATH)
    except Exception as e:
        err = f"Error loading ML artifacts: {e}\n{traceback.format_exc()}"
    return clf, le, err

clf, le, ml_load_err = load_ml_artifacts()
ml_ready = clf is not None and le is not None

defaults = {
    "running": False,
    "data_cache": pd.DataFrame(columns=[
        "timestamp", "focus_percent", "blink_per_min", "gaze_x", "gaze_y", "yaw", "pitch", "drift_count"
    ]),
    "drift_count": 0,
    "last_drift_time": None,
    "last_emotion_time": 0.0,
    "prev_emotion": None,
    "llm_messages": [],
    "last_llm_event": {},
    "llm_queue": None,
    "typing": False,
    "current_stream": "",
    "chat_input_value": "",
    "chat_clear_flag": False,
    "ws_running": False,
    "ws_last_ping": 0.0
}

for k, v in defaults.items():
    if k not in st.session_state:
        if k == "llm_queue":
            st.session_state["llm_queue"] = queue.Queue()
        else:
            st.session_state[k] = v

def compress_payload(data):
    compact = {}
    arr = data.get("focus_trend")
    if isinstance(arr, (list, tuple)) and arr:
        arr = list(map(float, arr[-120:]))
        compact["focus_summary"] = {
            "avg": float(np.mean(arr)),
            "max": float(np.max(arr)),
            "min": float(np.min(arr)),
            "sd": float(np.std(arr)),
        }
    arr2 = data.get("fatigue_curve")
    if isinstance(arr2, (list, tuple)) and arr2:
        arr2 = list(map(float, arr2[-120:]))
        compact["fatigue_summary"] = {
            "avg": float(np.mean(arr2)),
            "max": float(np.max(arr2)),
        }
    for k in ["focus", "fatigue", "blink", "gaze", "emotion", "trend", "drifts", "gaze_off", "head_angle"]:
        if k in data:
            compact[k] = data[k]
    return compact

def start_stream():
    try:
        r = requests.get(f"{BACKEND_BASE}/focus/metrics", timeout=3)

        if r.status_code == 200:
            st.session_state["running"] = True
            st.toast("🎥 Stream started!", icon="📡")
        else:
            st.toast("❌ Cannot reach backend", icon="❌")

    except Exception as e:
        st.toast(f"❌ Backend unreachable: {e}", icon="❌")

def stop_stream():
    st.session_state["running"] = False
    st.toast("🛑 Stream stopped", icon="🛑")

def fetch_snapshot():
    try:
        r = requests.get(f"{BACKEND_BASE}/focus/snapshot", timeout=3)
        return r.content if r.status_code == 200 else None
    except:
        return None

def fetch_metrics():
    try:
        r = requests.get(f"{BACKEND_BASE}/focus/metrics", timeout=3)
        r.raise_for_status()
        j = safe_json(r)
        if isinstance(j, dict) and "metrics" in j:
            j = j["metrics"]
        return pd.DataFrame(j) if isinstance(j, list) else pd.DataFrame()
    except:
        return pd.DataFrame()

def fetch_emotion():
    snapshot = fetch_snapshot()
    if not snapshot:
        return None
    files = {"file": ("frame.jpg", snapshot, "image/jpeg")}
    try:
        r = requests.post(f"{BACKEND_BASE}/emotion/analyze", files=files, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def compute_cognitive_insights(df):
    if df.empty:
        return None, None, None
    focus_vals = df["focus_percent"].tail(30).astype(float).values
    blink_vals = df["blink_per_min"].tail(30).astype(float).values
    gaze_x = float(df["gaze_x"].iloc[-1])
    gaze_y = float(df["gaze_y"].iloc[-1])
    stability = max(0, 100 - np.std(focus_vals) * 2)
    drift_detected = False
    if abs(gaze_x) > 0.3 or abs(gaze_y) > 0.3:
        now = time.time()
        if not st.session_state["last_drift_time"] or now - st.session_state["last_drift_time"] > 5:
            st.session_state["drift_count"] += 1
            st.session_state["last_drift_time"] = now
            drift_detected = True
    fatigue_score = min(100, (np.mean(blink_vals) * 0.4) + (np.var(focus_vals) * 0.6))
    fatigue_status = ("🟢 Fresh" if fatigue_score < 40 else "🟡 Mild Fatigue" if fatigue_score < 70 else "🔴 Fatigued")
    return stability, drift_detected, fatigue_status

def _enqueue_llm_message(msg):
    try:
        if isinstance(msg, str):
            msg = {"type": "final", "text": msg}
        st.session_state["llm_queue"].put_nowait(msg)
    except Exception:
        st.session_state["llm_queue"] = queue.Queue()
        if isinstance(msg, str):
            msg = {"type": "final", "text": msg}
        st.session_state["llm_queue"].put_nowait(msg)

def send_llm_event(event_type, data):
    now = time.time()
    last = st.session_state["last_llm_event"].get(event_type, 0)
    if now - last < 10:
        return
    st.session_state["last_llm_event"][event_type] = now
    payload = {"event_type": event_type, "data": compress_payload(data)}
    def _worker():
        try:
            r = requests.post(f"{LLM_BASE}/event", json=payload, timeout=10)
            j = safe_json(r) or {}
            msg_text = j.get("text") or j.get("summary") or j.get("result") or str(j)
            msg_type = j.get("type", "final")
            _enqueue_llm_message({"type": msg_type, "text": msg_text})
        except Exception as e:
            _enqueue_llm_message({"type": "final", "text": f"Error: {e}"})
    threading.Thread(target=_worker, daemon=True).start()

def llm_chat_async(message):
    payload = {"message": message}
    def _worker():
        _enqueue_llm_message({"type": "typing_start"})
        try:
            r = requests.post(f"{LLM_BASE}/chat", json=payload, timeout=30)
            j = safe_json(r) or {}
            reply = j.get("reply") or j.get("result") or str(j)
            chunk_size = 20
            for i in range(0, len(reply), chunk_size):
                _enqueue_llm_message({"type": "chunk", "text": reply[: i + chunk_size]})
                time.sleep(0.03)
            _enqueue_llm_message({"type": "final", "text": reply})
        except Exception as e:
            _enqueue_llm_message({"type": "final", "text": f"Error: {e}"})
        finally:
            _enqueue_llm_message({"type": "typing_end"})
    threading.Thread(target=_worker, daemon=True).start()

def ml_predict_state(row):
    try:
        f_focus = float(row.get("focus_percent", 0) or 0)
        f_blink = float(row.get("blink_per_min", 0) or 0)
        f_gx = float(row.get("gaze_x", 0) or 0)
        f_gy = float(row.get("gaze_y", 0) or 0)
        f_yaw = float(row.get("yaw", 0) or 0)
        f_pitch = float(row.get("pitch", 0) or 0)
        f_drift = float(st.session_state.get("drift_count", 0) or 0)
        features = [f_focus, f_blink, f_gx, f_gy, f_yaw, f_pitch, f_drift]
        pred = clf.predict([features])[0]
        label = le.inverse_transform([pred])[0] if le is not None else str(pred)
        prob = None
        try:
            if hasattr(clf, "predict_proba"):
                p = clf.predict_proba([features])[0]
                prob = float(np.max(p))
        except Exception:
            prob = None
        return label, prob
    except Exception as e:
        return f"err: {e}", None

def _start_ws():
    if st.session_state.get("ws_running"):
        return
    st.session_state["ws_running"] = True
    def on_message(ws, message):
        try:
            obj = json.loads(message)
        except Exception:
            obj = {"type": "final", "text": message}
        _enqueue_llm_message(obj)
        st.session_state["ws_last_ping"] = time.time()
    def on_error(ws, error):
        _enqueue_llm_message({"type": "final", "text": f"WS error: {error}"})
    def on_close(ws, close_status_code, close_msg):
        st.session_state["ws_running"] = False
        _enqueue_llm_message({"type": "final", "text": "WS closed"})
    def on_open(ws):
        try:
            pkt = json.dumps({"join": "dashboard"})
            ws.send(pkt)
        except Exception:
            pass
    def run_ws():
        while True:
            try:
                ws = WebSocketApp(WS_URL, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open)
                ws.run_forever()
            except Exception:
                _enqueue_llm_message({"type": "final", "text": "WS reconnecting..."})
                time.sleep(2)
            if not st.session_state.get("ws_running"):
                break
    t = threading.Thread(target=run_ws, daemon=True)
    t.start()

def _stop_ws():
    st.session_state["ws_running"] = False

with tab_dashboard:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🎥 Live Camera Feed")
        placeholder_image = np.zeros((480, 640, 3), np.uint8)
        image_slot = st.image(placeholder_image, use_container_width=True)
        st.subheader("📊 Focus Log")
        chart = st.empty()
    with col2:
        st.subheader("🔍 Realtime Metrics")
        focus_text = st.empty()
        blink_text = st.empty()
        gaze_text = st.empty()
        head_text = st.empty()
        status_text = st.empty()
        st.markdown("---")
        st.subheader("🧠 Cognitive Insights + ML")
        stability_text = st.empty()
        drift_text = st.empty()
        fatigue_text = st.empty()
        ml_state_text = st.empty()
        st.markdown("---")
        st.subheader("🧠 Assistant Events")
        llm_msg_slot = st.empty()
        st.markdown("---")
        run_button = st.button("🎬 Start / Restart Stream")
        stop_button = st.button("🛑 Stop Stream")
        ws_button = st.checkbox("Connect to LLM WebSocket", value=False)
        refresh_rate = st.slider("Refresh interval (s)", 0.5, 5.0, 1.0, 0.5)
    if run_button:
        start_stream()
    if stop_button:
        stop_stream()
    if ws_button and not st.session_state.get("ws_running"):
        _start_ws()
    if not ws_button and st.session_state.get("ws_running"):
        _stop_ws()
    try:
        from streamlit_autorefresh import st_autorefresh
        if st.session_state["running"]:
            st_autorefresh(interval=int(refresh_rate * 1000), key="autorefresh_dash")
    except Exception:
        pass
    if st.session_state["running"]:
        df_new = fetch_metrics()
        if not df_new.empty:
            for col in ["focus_percent", "blink_per_min", "gaze_x", "gaze_y", "yaw", "pitch", "drift_count", "timestamp"]:
                if col not in df_new:
                    df_new[col] = np.nan
                else:
                    df_new[col] = pd.to_numeric(df_new[col], errors="coerce")
            st.session_state["data_cache"] = pd.concat([st.session_state["data_cache"], df_new], ignore_index=True).drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
            df = st.session_state["data_cache"].tail(240)
            if not df.empty:
                df["ts"] = df["timestamp"].apply(lambda t: datetime.fromtimestamp(float(t)))
                chart.line_chart(df.set_index("ts")["focus_percent"], use_container_width=True)
                latest = df.iloc[-1]
                focus_text.markdown(f"**🧩 Focus:** {float(latest['focus_percent']):.1f}%")
                blink_text.markdown(f"**👁️ Blinks/min:** {float(latest['blink_per_min']):.1f}")
                gaze_text.markdown(f"**🎯 Gaze:** {float(latest['gaze_x']):.2f}, {float(latest['gaze_y']):.2f}")
                head_text.markdown(f"**🤖 Yaw/Pitch:** {float(latest['yaw']):.1f}, {float(latest['pitch']):.1f}")
                status_text.markdown(f"Updated: {df['ts'].iloc[-1].strftime('%H:%M:%S')}")
                stability, drift_detected, fatigue_status = compute_cognitive_insights(df)
                stability_text.markdown(f"**📊 Stability:** {stability:.1f}")
                drift_text.markdown(f"**🚨 Drift Count:** {st.session_state['drift_count']}")
                fatigue_text.markdown(f"**💤 Fatigue:** {fatigue_status}")
                try:
                    avg_recent = float(df["focus_percent"].tail(30).mean())
                    if float(latest["focus_percent"]) < max(50, avg_recent - 18):
                        send_llm_event("focus_drop", {"focus": float(latest["focus_percent"]), "fatigue": fatigue_status, "blink": float(latest["blink_per_min"])})
                    if drift_detected:
                        send_llm_event("distraction", {"gaze_off": 5, "head_angle": float(latest["yaw"])})
                except Exception:
                    pass
                if ml_ready:
                    ml_label, ml_prob = ml_predict_state(latest)
                else:
                    ml_label, ml_prob = "Model not loaded", None
                if isinstance(ml_label, str) and ml_label.startswith("err:"):
                    ml_state_text.markdown(f"**🧠 ML Error:** {ml_label}")
                else:
                    if ml_prob is not None:
                        ml_state_text.markdown(f"**🧠 ML Predicted State:** {ml_label} (conf {ml_prob:.2f})")
                    else:
                        ml_state_text.markdown(f"**🧠 ML Predicted State:** {ml_label}")
                    if ml_label == "Fatigued":
                        send_llm_event("ml_fatigue", {"state": ml_label})
        snapshot = fetch_snapshot()
        if snapshot:
            image_slot.image(snapshot, use_container_width=True)
        now_ts = time.time()
        if now_ts - st.session_state["last_emotion_time"] > 2:
            st.session_state["last_emotion_time"] = now_ts
            emo = fetch_emotion()
            if emo:
                dominant = emo.get("emotion", "unknown")
                if dominant != st.session_state["prev_emotion"]:
                    st.session_state["prev_emotion"] = dominant
                    send_llm_event("emotion_shift", {"emotion": dominant})
    try:
        q = st.session_state["llm_queue"]
        drained = []
        while not q.empty() and len(drained) < 24:
            drained.append(q.get_nowait())
        for tok in drained:
            if not isinstance(tok, dict):
                continue
            t = tok.get("type")
            if t == "typing_start":
                st.session_state["typing"] = True
            elif t == "typing_end":
                st.session_state["typing"] = False
                st.session_state["current_stream"] = ""
            elif t == "chunk":
                st.session_state["current_stream"] = tok.get("text", "")
            elif t == "final":
                final_text = tok.get("text", "").strip()
                st.session_state["current_stream"] = ""
                st.session_state["typing"] = False
                st.session_state["llm_messages"].append({"ts": datetime.utcnow().strftime("%H:%M:%S"), "text": final_text, "source": "LLM"})
            elif t == "event":
                event_text = tok.get("text", "")
                st.session_state["llm_messages"].append({"ts": datetime.utcnow().strftime("%H:%M:%S"), "text": event_text, "source": "LLM"})
    except Exception:
        st.session_state["llm_queue"] = queue.Queue()

# ============================================================================================
# TAB 2 — CHAT INTERFACE (thread-safe streaming, fixed)
# ============================================================================================
with tab_chat:
    st.title("🤖 NeuroSync Assistant")

    # -------------------------
    # Controls column
    # -------------------------
    ctrl_col, chat_col = st.columns([2, 8])
    with ctrl_col:
        st.markdown("### Controls")
        local_memory_mode = st.checkbox(
            "Keep history local (client-side only)",
            value=False,
            help="When checked, assistant replies are also stored in local browser session memory."
        )

        if st.button("Clear server memory"):
            try:
                r = requests.post(f"{LLM_BASE}/memory/clear", timeout=6)
                if r.status_code == 200:
                    st.success("Server memory cleared.")
                else:
                    st.error(f"Server responded: {r.status_code}")
            except Exception as e:
                st.error(f"Error clearing server memory: {e}")

        if st.button("Export local chat"):
            export = {"chat": st.session_state.get("llm_messages", [])}
            st.download_button(
                "Download JSON",
                data=str(export),
                file_name="neurosync_chat.json",
                mime="application/json"
            )

    # -------------------------
    # Ensure session state keys exist (main thread only)
    # -------------------------
    if "llm_messages" not in st.session_state:
        st.session_state["llm_messages"] = []  # list of {"ts","text","source"}

    if "llm_queue" not in st.session_state or st.session_state["llm_queue"] is None:
        st.session_state["llm_queue"] = queue.Queue()

    if "typing" not in st.session_state:
        st.session_state["typing"] = False

    if "current_stream" not in st.session_state:
        st.session_state["current_stream"] = ""

    if "chat_input_value" not in st.session_state:
        st.session_state["chat_input_value"] = ""

    # flag used to clear the input safely on next rerun
    if "chat_clear_flag" not in st.session_state:
        st.session_state["chat_clear_flag"] = False

    # local reference to avoid repeated session_state access inside loops
    local_queue = st.session_state["llm_queue"]

    # -------------------------
    # safe_json helper
    # -------------------------
    def safe_json(resp):
        try:
            return resp.json()
        except Exception:
            return {}

    # -------------------------
    # styling
    # -------------------------
    st.markdown(
        """
        <style>
        .chat-box { background:#0b1220; color:#e6eef9; padding:12px; border-radius:10px;
                   max-height:520px; overflow:auto; }
        .msg-row { display:flex; margin-bottom:10px; align-items:flex-end; }
        .msg-user { margin-left:auto; background:#dbeafe; color:#061122; padding:10px 14px;
                   border-radius:14px; max-width:72%; word-wrap:break-word; }
        .msg-assistant { margin-right:auto; background:#0f1724; color:#e6eef9; padding:10px 14px;
                         border-radius:14px; max-width:72%; word-wrap:break-word; border:1px solid rgba(255,255,255,0.04);}
        .avatar { width:36px; height:36px; border-radius:50%; margin-right:8px; }
        .typing { opacity:0.85; font-style:italic; color:#9fb3d7; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------
    # Drain queue (MAIN THREAD only)
    # -------------------------
    try:
        drained = []
        q = local_queue
        # limit how many tokens we process each tick
        while not q.empty() and len(drained) < 16:
            drained.append(q.get_nowait())

        for tok in drained:
            # ignore accidental non-dict values (defensive)
            if not isinstance(tok, dict):
                continue

            tok_type = tok.get("type")
            if tok_type == "typing_start":
                st.session_state["typing"] = True

            elif tok_type == "typing_end":
                st.session_state["typing"] = False
                st.session_state["current_stream"] = ""

            elif tok_type == "chunk":
                # display the streaming partial (do NOT append to history)
                st.session_state["current_stream"] = tok.get("text", "")

            elif tok_type == "final":
                # final assistant text -> add to chat history only here
                final_text = tok.get("text", "")
                st.session_state["current_stream"] = ""
                st.session_state["llm_messages"].append({
                    "ts": datetime.utcnow().strftime("%H:%M:%S"),
                    "text": final_text,
                    "source": "LLM"
                })
                if local_memory_mode:
                    st.session_state.setdefault("chat_memory", []).append({
                        "role": "assistant",
                        "text": final_text,
                        "ts": datetime.utcnow().isoformat()
                    })
    except Exception:
        # recover queue if broken
        st.session_state["llm_queue"] = queue.Queue()
        local_queue = st.session_state["llm_queue"]

    # -------------------------
    # Render chat HTML (history + streaming + typing)
    # -------------------------
    def build_chat_html(messages, streaming_text, typing_flag):
        html = "<div class='chat-box'>"
        for m in messages[-80:]:
            if m.get("source") == "USER":
                html += (
                    "<div class='msg-row' style='justify-content:flex-end;'>"
                    f"<div class='msg-user'>{m.get('text')}</div>"
                    "<img class='avatar' src='https://cdn-icons-png.flaticon.com/512/847/847969.png'>"
                    "</div>"
                )
            else:
                html += (
                    "<div class='msg-row'>"
                    "<img class='avatar' src='https://cdn-icons-png.flaticon.com/512/4712/4712107.png'>"
                    f"<div class='msg-assistant'>{m.get('text')}</div>"
                    "</div>"
                )

        # streaming in-progress (assistant)
        if streaming_text:
            html += (
                "<div class='msg-row'>"
                "<img class='avatar' src='https://cdn-icons-png.flaticon.com/512/4712/4712107.png'>"
                f"<div class='msg-assistant typing'>{streaming_text}</div>"
                "</div>"
            )
        elif typing_flag:
            html += (
                "<div class='msg-row'>"
                "<img class='avatar' src='https://cdn-icons-png.flaticon.com/512/4712/4712107.png'>"
                "<div class='msg-assistant typing'>Typing <span style='margin-left:8px;'>● ● ●</span></div>"
                "</div>"
            )

        html += "</div>"
        return html

    chat_html = build_chat_html(
        st.session_state["llm_messages"],
        st.session_state.get("current_stream", ""),
        st.session_state.get("typing", False),
    )

    chat_box = st.empty()
    chat_box.markdown(chat_html, unsafe_allow_html=True)

    # -------------------------
    # Input row (safe clearing pattern)
    # -------------------------
    st.markdown("---")

    # default value: clear if flag set, otherwise keep last
    default_text = "" if st.session_state.get("chat_clear_flag", False) else st.session_state.get("chat_input_value", "")
    # reset flag immediately (so widget renders with cleared value exactly once)
    st.session_state["chat_clear_flag"] = False

    user_input = st.text_input(
        "Your message:",
        key="chat_input_value",
        value=default_text,
        placeholder="Type your question and press Send..."
    )

    send_clicked = st.button("Send", key="chat_send_btn")

    # -------------------------
    # Worker (background) - NEVER call Streamlit inside this thread
    # -------------------------
    def chat_worker(prompt, q_obj):
        # control tokens only, no streamlit ops
        q_obj.put({"type": "typing_start"})
        try:
            r = requests.post(f"{LLM_BASE}/chat", json={"message": prompt}, timeout=30)
            j = safe_json(r)
            reply = j.get("reply") or j.get("summary") or j.get("result") or str(j)
        except Exception as e:
            reply = f"Error: {e}"

        # stream reply in small chunks for nicer UX
        # but do NOT append control tokens to history
        chunk_size = 20
        for i in range(0, len(reply), chunk_size):
            q_obj.put({"type": "chunk", "text": reply[: i + chunk_size]})
            # short sleep to emulate streaming; keep tiny to feel responsive
            time.sleep(0.04)

        # final token
        q_obj.put({"type": "final", "text": reply})
        q_obj.put({"type": "typing_end"})

    # -------------------------
    # When user sends
    # -------------------------
    if send_clicked and user_input and user_input.strip():
        # add user bubble to UI immediately (main thread)
        st.session_state["llm_messages"].append({
            "ts": datetime.utcnow().strftime("%H:%M:%S"),
            "text": user_input,
            "source": "USER"
        })

        # local memory if enabled
        if local_memory_mode:
            st.session_state.setdefault("chat_memory", []).append({
                "role": "user",
                "text": user_input,
                "ts": datetime.utcnow().isoformat()
            })

        # tell main thread to clear input on next render
        st.session_state["chat_clear_flag"] = True

        # start worker thread with local queue reference
        threading.Thread(target=chat_worker, args=(user_input, local_queue), daemon=True).start()

        # trigger a rerun so UI updates immediately (input clears and typing shows)
        st.rerun()

with st.expander("Debug Info"):
    st.write("Backend:", BACKEND_BASE)
    st.write("LLM:", LLM_BASE)
    st.write("Running:", st.session_state["running"])
    st.write("WebSocket connected:", st.session_state.get("ws_running", False))
    st.write("WS last ping:", datetime.fromtimestamp(st.session_state.get("ws_last_ping", 0)).isoformat() if st.session_state.get("ws_last_ping", 0) else "never")
    st.write("Datapoints:", len(st.session_state["data_cache"]))
    st.write("LLM Messages:", st.session_state["llm_messages"][-12:])
    if ml_load_err:
        st.code(ml_load_err)
