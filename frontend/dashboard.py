# frontend/dashboard.py
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import numpy as np
import joblib

# ----------------- Config / Secrets -----------------
try:
    BACKEND_BASE = st.secrets["backend_base"]
except Exception:
    BACKEND_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="🧠 NeuroSync Dashboard", layout="wide")
st.title("🧠 NeuroSync — Cognitive Focus Dashboard")

# ----------------- Load ML Model -----------------
try:
    clf = joblib.load("../ml/model.pkl")
    le = joblib.load("../ml/label_encoder.pkl")
    ml_ready = True
except Exception as e:
    st.warning(f"ML model not loaded: {e}")
    ml_ready = False

# ----------------- UI Layout -----------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎥 Live Camera Feed")
    placeholder_image = np.zeros((480, 640, 3), dtype=np.uint8)
    image_slot = st.image(placeholder_image, channels="BGR", width="stretch")

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
    st.write("⚙️ Controls")
    run_button = st.button("🎬 Start / Restart Stream")
    stop_button = st.button("🛑 Stop Stream")
    refresh_rate = st.slider("Refresh interval (seconds)", 0.5, 5.0, 1.0, 0.5)
    fetch_button = st.button("🔄 Fetch Latest Metrics")

# ----------------- Session State Init -----------------
if "running" not in st.session_state:
    st.session_state["running"] = False
if "data_cache" not in st.session_state:
    st.session_state["data_cache"] = pd.DataFrame(
        columns=["timestamp", "focus_percent", "blink_per_min", "gaze_x", "gaze_y", "yaw", "pitch", "drift_count"]
    )
if "drift_count" not in st.session_state:
    st.session_state["drift_count"] = 0
if "last_drift_time" not in st.session_state:
    st.session_state["last_drift_time"] = None

# ----------------- Helper Functions -----------------
def start_stream():
    try:
        r = requests.get(f"{BACKEND_BASE}/focus/metrics", timeout=3)
        if r.status_code == 200:
            st.session_state["running"] = True
            st.toast("🎥 Stream started successfully!", icon="✅")
        else:
            st.error(f"⚠️ Failed to start stream — status {r.status_code}")
    except Exception as e:
        st.error(f"❌ Backend not reachable: {e}")

def stop_stream():
    st.session_state["running"] = False
    st.toast("🛑 Stream stopped", icon="🧩")

def fetch_snapshot():
    try:
        r = requests.get(f"{BACKEND_BASE}/focus/snapshot", timeout=2.5)
        if r.status_code == 200:
            return r.content
    except Exception:
        return None
    return None

def fetch_metrics():
    try:
        r = requests.get(f"{BACKEND_BASE}/focus/metrics", timeout=2.5)
        r.raise_for_status()
        j = r.json()
        data = j.get("metrics") if isinstance(j, dict) else j if isinstance(j, list) else []
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def compute_cognitive_insights(df):
    if df.empty:
        return None, None, None

    focus_vals = df["focus_percent"].tail(30).values
    blink_vals = df["blink_per_min"].tail(30).values
    gaze_x, gaze_y = df["gaze_x"].iloc[-1], df["gaze_y"].iloc[-1]

    stability = max(0, 100 - np.std(focus_vals) * 2)
    drift_detected = False

    if abs(gaze_x) > 0.3 or abs(gaze_y) > 0.3:
        now = datetime.now().timestamp()
        if st.session_state["last_drift_time"] is None or now - st.session_state["last_drift_time"] > 5:
            st.session_state["drift_count"] += 1
            st.session_state["last_drift_time"] = now
            drift_detected = True

    focus_var = np.var(focus_vals)
    blink_rate = np.mean(blink_vals) if len(blink_vals) > 0 else 0
    fatigue_score = min(100, (blink_rate * 0.4) + (focus_var * 0.6))

    fatigue_status = (
        "🟢 Fresh" if fatigue_score < 40 else
        "🟡 Mild Fatigue" if fatigue_score < 70 else
        "🔴 Fatigued"
    )

    return stability, drift_detected, fatigue_status

def render_session_summary(df):
    if df.empty:
        st.info("No session data yet.")
        return

    st.markdown("## 🧾 Session Summary")
    duration = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) if len(df) > 1 else 0
    avg_focus = df["focus_percent"].mean()
    max_focus = df["focus_percent"].max()
    min_focus = df["focus_percent"].min()

    st.write(f"**⏱️ Duration:** {duration:.1f} sec")
    st.write(f"**📈 Avg Focus:** {avg_focus:.1f}%")
    st.write(f"**🔺 Max Focus:** {max_focus:.1f}%")
    st.write(f"**🔻 Min Focus:** {min_focus:.1f}%")
    st.write(f"**👁️ Drift Count:** {st.session_state['drift_count']}")
    verdict = (
        "💪 Great Session!" if avg_focus > 80 else
        "🙂 Decent Focus" if avg_focus > 60 else
        "😴 Distracted Session"
    )
    st.write(f"**Verdict:** {verdict}")

# ----------------- Button Actions -----------------
if run_button:
    start_stream()
if stop_button:
    stop_stream()
if fetch_button:
    fetch_metrics()

# ----------------- Auto-Refresh -----------------
try:
    from streamlit_autorefresh import st_autorefresh
    if st.session_state["running"]:
        st_autorefresh(interval=int(refresh_rate * 1000), key="autorefresh")
except Exception:
    if st.session_state["running"]:
        st.experimental_set_query_params(_refresh=int(datetime.utcnow().timestamp() * 1000) % 1000000)

# ----------------- Main Loop -----------------
if st.session_state["running"]:
    df_new = fetch_metrics()
    if not df_new.empty:
        st.session_state["data_cache"] = pd.concat([st.session_state["data_cache"], df_new]).drop_duplicates(subset=["timestamp"], keep="last")
        df = st.session_state["data_cache"].copy()

        if "timestamp" in df.columns:
            df["ts"] = df["timestamp"].apply(lambda t: datetime.fromtimestamp(t))
            df = df.sort_values("ts").tail(240)
            chart.line_chart(df.set_index("ts")["focus_percent"])

            latest = df.iloc[-1]
            focus_text.markdown(f"**🧩 Focus:** {latest['focus_percent']}%")
            blink_text.markdown(f"**👁️ Blinks/min:** {latest['blink_per_min']}")
            gaze_text.markdown(f"**🎯 Gaze (x,y):** {latest['gaze_x']:.2f}, {latest['gaze_y']:.2f}")
            head_text.markdown(f"**🤖 Yaw/Pitch:** {latest['yaw']:.1f}, {latest['pitch']:.1f}")
            status_text.markdown(f"**🕓 Last updated:** {latest['ts'].isoformat()}")

            # --- Cognitive Insights ---
            stability, drift_detected, fatigue_status = compute_cognitive_insights(df)
            if stability is not None:
                stability_text.markdown(f"**📊 Focus Stability:** {stability:.1f} / 100")
                drift_text.markdown(f"**🚨 Drift Count:** {st.session_state['drift_count']} ({'⚠️ Drift!' if drift_detected else '✅ Stable'})")
                fatigue_text.markdown(f"**💤 Fatigue Level:** {fatigue_status}")

            # --- ML Prediction ---
            if ml_ready:
                features = [[
                    latest['focus_percent'],
                    latest['blink_per_min'],
                    latest['gaze_x'],
                    latest['gaze_y'],
                    latest['yaw'],
                    latest['pitch'],
                    latest.get('drift_count', 0)
                ]]
                pred_label = clf.predict(features)
                pred_state = le.inverse_transform(pred_label)[0]
                fatigue_text.markdown(f"**💤 Fatigue Level (Rule vs ML):** {fatigue_status} | {pred_state}")

    snapshot = fetch_snapshot()
    image_slot.image(snapshot if snapshot else placeholder_image, channels="BGR", width="stretch")
else:
    st.markdown("---")
    render_session_summary(st.session_state["data_cache"])

# ----------------- Debug Panel -----------------
with st.expander("Backend & Status", expanded=False):
    st.write("Backend base URL:", BACKEND_BASE)
    st.write("Streaming running:", st.session_state["running"])
    st.write("Cached datapoints:", len(st.session_state["data_cache"]))
    st.write("Drift Count:", st.session_state["drift_count"])
