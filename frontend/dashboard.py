# frontend/dashboard.py
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
import streamlit.components.v1 as components

# ======================================================
# CONFIG
# ======================================================
BACKEND_BASE = "http://127.0.0.1:8000"
LLM_BASE = "http://127.0.0.1:8765"

st.set_page_config(page_title="🧠 NeuroSync Dashboard", layout="wide")

# --------------------------
# TABS
# --------------------------
tabs = st.tabs(["📊 Dashboard", "🤖 Assistant Chat"])
tab_dashboard, tab_chat = tabs

with tab_dashboard:
    st.title("🧠 NeuroSync — Cognitive Focus Dashboard")

# ======================================================
# ML MODEL LOAD
# ======================================================
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
ml_ready = clf is not None


# ======================================================
# SESSION STATE INIT
# ======================================================
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
}

for k, v in defaults.items():
    if k not in st.session_state:
        if k == "llm_queue":
            st.session_state["llm_queue"] = queue.Queue()
        else:
            st.session_state[k] = v


# ======================================================
# UTILITY HELPERS
# ======================================================
def _safe_get_json(r):
    try:
        return r.json()
    except:
        return None


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


# ======================================================
# BACKEND INTERACTIONS
# ======================================================
def start_stream():
    try:
        r = requests.get(f"{BACKEND_BASE}/focus/metrics", timeout=3)
        if r.status_code == 200:
            st.session_state["running"] = True
            st.info("🎥 Stream started!")
        else:
            st.error("Cannot reach backend.")
    except Exception as e:
        st.error(f"Backend unreachable: {e}")


def stop_stream():
    st.session_state["running"] = False
    st.info("🛑 Stream stopped")


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
        j = _safe_get_json(r)
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


# ======================================================
# INSIGHTS
# ======================================================
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
    fatigue_status = (
        "🟢 Fresh" if fatigue_score < 40 else
        "🟡 Mild Fatigue" if fatigue_score < 70 else
        "🔴 Fatigued"
    )

    return stability, drift_detected, fatigue_status


# ======================================================
# LLM WORKERS
# ======================================================
def _enqueue_llm_message(msg):
    try:
        st.session_state["llm_queue"].put_nowait(msg)
    except:
        st.session_state["llm_queue"] = queue.Queue()
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
            r = requests.post(f"{LLM_BASE}/event", json=payload, timeout=8)
            j = _safe_get_json(r) or {}
            msg = j.get("summary") or j.get("result") or j.get("status") or str(j)
            _enqueue_llm_message(msg)
        except Exception as e:
            _enqueue_llm_message(f"Error: {e}")

    threading.Thread(target=_worker, daemon=True).start()


def llm_chat_async(message):
    payload = {"message": message}

    def _worker():
        try:
            r = requests.post(f"{LLM_BASE}/chat", json=payload, timeout=15)
            j = _safe_get_json(r) or {}
            reply = j.get("reply") or str(j)
            _enqueue_llm_message(reply)
        except Exception as e:
            _enqueue_llm_message(f"Error: {e}")

    threading.Thread(target=_worker, daemon=True).start()


# ============================================================================================
# TAB 1 — DASHBOARD VIEW
# ============================================================================================
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

        st.subheader("🧠 Cognitive Insights")
        stability_text = st.empty()
        drift_text = st.empty()
        fatigue_text = st.empty()
        st.markdown("---")

        st.subheader("🧠 Assistant Events")
        llm_msg_slot = st.empty()
        st.markdown("---")

        run_button = st.button("🎬 Start / Restart Stream")
        stop_button = st.button("🛑 Stop Stream")
        refresh_rate = st.slider("Refresh interval", 0.5, 5.0, 1.0, 0.5)

    if run_button:
        start_stream()
    if stop_button:
        stop_stream()

    # Auto-refresh
    try:
        from streamlit_autorefresh import st_autorefresh
        if st.session_state["running"]:
            st_autorefresh(interval=int(refresh_rate * 1000), key="autorefresh2")
    except:
        pass

    # ======================================================
    # MAIN LOOP
    # ======================================================
    if st.session_state["running"]:
        df_new = fetch_metrics()
        if not df_new.empty:
            for col in [
                "focus_percent", "blink_per_min",
                "gaze_x", "gaze_y", "yaw", "pitch",
                "drift_count", "timestamp"
            ]:
                if col not in df_new:
                    df_new[col] = np.nan
                else:
                    df_new[col] = pd.to_numeric(df_new[col], errors="coerce")

            st.session_state["data_cache"] = (
                pd.concat([st.session_state["data_cache"], df_new], ignore_index=True)
                .drop_duplicates(subset=["timestamp"], keep="last")
                .reset_index(drop=True)
            )
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

                # Trigger LLM
                try:
                    avg_recent = float(df["focus_percent"].tail(30).mean())
                    if float(latest["focus_percent"]) < max(50, avg_recent - 18):
                        send_llm_event("focus_drop", {
                            "focus": float(latest["focus_percent"]),
                            "fatigue": fatigue_status,
                            "blink": float(latest["blink_per_min"]),
                        })

                    if drift_detected:
                        send_llm_event("distraction", {
                            "gaze_off": 5,
                            "head_angle": float(latest["yaw"])
                        })
                except:
                    pass

        # show camera
        snapshot = fetch_snapshot()
        if snapshot:
            image_slot.image(snapshot, use_container_width=True)

        # EMOTION
        now_ts = time.time()
        if now_ts - st.session_state["last_emotion_time"] > 2:
            st.session_state["last_emotion_time"] = now_ts
            emo = fetch_emotion()
            if emo:
                dominant = emo.get("emotion", "unknown")
                if dominant != st.session_state["prev_emotion"]:
                    st.session_state["prev_emotion"] = dominant
                    send_llm_event("emotion_shift", {"emotion": dominant})


    # ======================================================
    # PROCESS LLM QUEUE (FIXED — filters chunk/tokens properly)
    # ======================================================
    try:
        q = st.session_state["llm_queue"]
        drained = []

        # Pull up to 16 tokens per refresh
        while not q.empty() and len(drained) < 16:
            drained.append(q.get_nowait())

        for tok in drained:

            # Skip accidental non-dict messages
            if not isinstance(tok, dict):
                continue

            t = tok.get("type")

            # -----------------------------
            # START TYPING
            # -----------------------------
            if t == "typing_start":
                st.session_state["typing"] = True

            # -----------------------------
            # STOP TYPING
            # -----------------------------
            elif t == "typing_end":
                st.session_state["typing"] = False
                st.session_state["current_stream"] = ""

            # -----------------------------
            # STREAMING CHUNK UPDATE
            # -----------------------------
            elif t == "chunk":
                # Show partial response (live text)
                st.session_state["current_stream"] = tok.get("text", "")

            # -----------------------------
            # FINAL RESPONSE
            # -----------------------------
            elif t == "final":
                final_text = tok.get("text", "").strip()

                # Clear streaming state
                st.session_state["current_stream"] = ""
                st.session_state["typing"] = False

                # Append ONLY the final assistant message
                st.session_state["llm_messages"].append({
                    "ts": datetime.utcnow().strftime("%H:%M:%S"),
                    "text": final_text,
                    "source": "LLM"
                })

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

# ============================================================================================
# DEBUG PANEL
# ============================================================================================
with st.expander("Debug Info"):
    st.write("Backend:", BACKEND_BASE)
    st.write("LLM:", LLM_BASE)
    st.write("Running:", st.session_state["running"])
    st.write("Datapoints:", len(st.session_state["data_cache"]))
    st.write("LLM Messages:", st.session_state["llm_messages"][-10:])
    if ml_load_err:
        st.code(ml_load_err)
