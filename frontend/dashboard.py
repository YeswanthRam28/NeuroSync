# frontend/dashboard.py
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import numpy as np
import joblib
import os
import traceback

# ----------------- Config / Secrets -----------------
try:
    BACKEND_BASE = st.secrets["backend_base"]
except Exception:
    BACKEND_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="🧠 NeuroSync Dashboard", layout="wide")
st.title("🧠 NeuroSync — Cognitive Focus Dashboard")

# ----------------- Load ML Model (cached) -----------------
MODEL_PATH = os.path.join("ml", "model.pkl")
LABEL_PATH = os.path.join("ml", "label_encoder.pkl")


@st.cache_resource
def load_ml_artifacts():
    """Return (clf, label_encoder, error_str)."""
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
        # else label encoder can be None (we'll just show raw labels)
    except Exception as e:
        err = f"Error loading ML artifacts: {e}\n{traceback.format_exc()}"
    return clf, le, err


clf, le, ml_load_err = load_ml_artifacts()
ml_ready = clf is not None

if not ml_ready:
    st.warning("ML model not loaded." + (f" Details: {ml_load_err}" if ml_load_err else ""))

# ----------------- UI Layout -----------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎥 Live Camera Feed")
    placeholder_image = np.zeros((480, 640, 3), dtype=np.uint8)
    image_slot = st.image(placeholder_image, width="stretch")

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
    analyze_button = st.button("😊 Analyze Emotion from Snapshot")

# ----------------- Session State Init -----------------
if "running" not in st.session_state:
    st.session_state["running"] = False
if "data_cache" not in st.session_state:
    st.session_state["data_cache"] = pd.DataFrame(
        columns=[
            "timestamp",
            "focus_percent",
            "blink_per_min",
            "gaze_x",
            "gaze_y",
            "yaw",
            "pitch",
            "drift_count",
        ]
    )
if "drift_count" not in st.session_state:
    st.session_state["drift_count"] = 0
if "last_drift_time" not in st.session_state:
    st.session_state["last_drift_time"] = None

# ----------------- Helper Functions -----------------


def _safe_get_json(r):
    try:
        return r.json()
    except Exception:
        return None


def start_stream():
    """Start stream by verifying backend metrics endpoint and toggling running state."""
    try:
        # ping /focus/metrics (non-streaming) to verify backend alive
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
    """
    Fetch snapshot bytes from backend.
    Backend expected to return raw jpeg bytes (content-type image/jpeg).
    """
    try:
        r = requests.get(f"{BACKEND_BASE}/focus/snapshot", timeout=3)
        if r.status_code == 200 and r.content:
            return r.content
    except Exception:
        pass
    return None


def fetch_metrics():
    """
    Fetch metrics. Backend may return either:
      - {"metrics": [ ... ] } or
      - [ ... ]
    We return a DataFrame (possibly empty).
    """
    try:
        r = requests.get(f"{BACKEND_BASE}/focus/metrics", timeout=3)
        r.raise_for_status()
        j = _safe_get_json(r)
        if j is None:
            return pd.DataFrame()
        if isinstance(j, dict) and "metrics" in j:
            data = j.get("metrics", [])
        elif isinstance(j, list):
            data = j
        else:
            # sometimes backend returns single latest metric dict
            data = [j] if isinstance(j, dict) else []
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        return df
    except Exception:
        return pd.DataFrame()


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
        now = datetime.now().timestamp()
        if st.session_state["last_drift_time"] is None or now - st.session_state["last_drift_time"] > 5:
            st.session_state["drift_count"] += 1
            st.session_state["last_drift_time"] = now
            drift_detected = True

    focus_var = float(np.var(focus_vals))
    blink_rate = float(np.mean(blink_vals)) if len(blink_vals) > 0 else 0.0
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


def analyze_emotion():
    """
    Sends a snapshot to a backend emotion analyser.
    backend expects 'image' file. We send multipart properly with filename and content-type.
    """
    snapshot = fetch_snapshot()
    if not snapshot:
        st.error("No snapshot available to analyze.")
        return None

    files = {"image": ("snapshot.jpg", snapshot, "image/jpeg")}
    try:
        r = requests.post(f"{BACKEND_BASE}/analyze/emotion", files=files, timeout=5)
        if r.status_code == 200:
            # backend could return json or bytes; try json first
            try:
                return r.json()
            except Exception:
                return r.content
        else:
            st.error(f"Emotion analysis failed: {r.status_code}")
    except Exception as e:
        st.error(f"Emotion analysis error: {e}")
    return None


# ----------------- Button Actions -----------------
if run_button:
    start_stream()
if stop_button:
    stop_stream()
if fetch_button:
    _ = fetch_metrics()  # quick manual poll
if analyze_button:
    emo = analyze_emotion()
    if emo is not None:
        st.write("Emotion analysis result:")
        st.json(emo)


# ----------------- Auto-Refresh -----------------
try:
    from streamlit_autorefresh import st_autorefresh
    if st.session_state["running"]:
        st_autorefresh(interval=int(refresh_rate * 1000), key="autorefresh")
except Exception:
    if st.session_state["running"]:
        st.experimental_set_query_params(_refresh=int(datetime.utcnow().timestamp() * 1000) % 1000000)


# ----------------- Safe ML prediction helper -----------------
def safe_ml_predict(latest_row):
    """
    latest_row: pandas Series or dict containing feature values.
    Returns (pred_label, prob, debug_str)
    """
    if not ml_ready or clf is None:
        return None, None, "ML not available"

    # define expected feature order (must match training)
    feature_names = ["focus_percent", "blink_per_min", "gaze_x", "gaze_y", "yaw", "pitch", "drift_count"]
    vals = []
    debug = []
    for f in feature_names:
        try:
            v = latest_row.get(f) if hasattr(latest_row, "get") else latest_row[f]
        except Exception:
            try:
                v = float(latest_row[f])
            except Exception:
                v = None
        if v is None or (isinstance(v, float) and np.isnan(v)):
            debug.append(f"feature {f} missing/NaN -> filling 0.0")
            v = 0.0
        vals.append(float(v))
    X = np.array(vals, dtype=float).reshape(1, -1)

    # check n_features_in_ if available for warning
    try:
        expected = getattr(clf, "n_features_in_", None)
        if expected is None and hasattr(clf, "steps"):
            # pipeline case - try last estimator
            last = clf.steps[-1][1]
            expected = getattr(last, "n_features_in_", None)
        if expected is not None and X.shape[1] != expected:
            debug.append(f"WARNING: X.shape={X.shape}, model expects {expected}")
    except Exception as e:
        debug.append(f"n_features_in_ check error: {e}")

    # predict
    try:
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X)
            idx = int(np.argmax(probs, axis=1)[0])
            prob_val = float(np.max(probs, axis=1)[0])
            classes = getattr(clf, "classes_", None)
            pred_raw = classes[idx] if classes is not None else idx
            if le is not None:
                try:
                    pred_label = le.inverse_transform([pred_raw])[0]
                except Exception:
                    pred_label = str(pred_raw)
            else:
                pred_label = str(pred_raw)
            debug.append(f"predict_proba ok, prob={prob_val:.3f}")
            return pred_label, prob_val, "\n".join(debug)
        else:
            pred = clf.predict(X)
            pred0 = pred[0]
            if le is not None:
                try:
                    pred_label = le.inverse_transform([pred0])[0]
                except Exception:
                    pred_label = str(pred0)
            else:
                pred_label = str(pred0)
            debug.append("predict() ok")
            return pred_label, None, "\n".join(debug)
    except Exception as e:
        debug.append("Prediction error: " + str(e))
        debug.append(traceback.format_exc())
        return None, None, "\n".join(debug)


# ----------------- Main Loop -----------------
if st.session_state["running"]:
    df_new = fetch_metrics()
    if not df_new.empty:
        # ensure expected numeric columns exist and coerce types
        for col in ["focus_percent", "blink_per_min", "gaze_x", "gaze_y", "yaw", "pitch", "drift_count", "timestamp"]:
            if col not in df_new.columns:
                df_new[col] = np.nan
        # coerce numeric
        num_cols = ["focus_percent", "blink_per_min", "gaze_x", "gaze_y", "yaw", "pitch", "drift_count"]
        df_new[num_cols] = df_new[num_cols].astype(float, errors="ignore")
        st.session_state["data_cache"] = (
            pd.concat([st.session_state["data_cache"], df_new], ignore_index=True)
            .drop_duplicates(subset=["timestamp"], keep="last")
            .reset_index(drop=True)
        )
        df = st.session_state["data_cache"].copy()

        if "timestamp" in df.columns and not df.empty:
            # convert timestamp -> ts and sort
            df["ts"] = df["timestamp"].apply(lambda t: datetime.fromtimestamp(float(t)))
            df = df.sort_values("ts").tail(240)
            chart.line_chart(df.set_index("ts")["focus_percent"])

            latest = df.iloc[-1]
            focus_text.markdown(f"**🧩 Focus:** {float(latest['focus_percent']):.1f}%")
            blink_text.markdown(f"**👁️ Blinks/min:** {float(latest['blink_per_min']):.1f}")
            gaze_text.markdown(f"**🎯 Gaze (x,y):** {float(latest['gaze_x']):.2f}, {float(latest['gaze_y']):.2f}")
            head_text.markdown(f"**🤖 Yaw/Pitch:** {float(latest['yaw']):.1f}, {float(latest['pitch']):.1f}")
            status_text.markdown(f"**🕓 Last updated:** {df['ts'].iloc[-1].isoformat()}")

            # --- Cognitive Insights ---
            stability, drift_detected, fatigue_status = compute_cognitive_insights(df)
            if stability is not None:
                stability_text.markdown(f"**📊 Focus Stability:** {stability:.1f} / 100")
                drift_text.markdown(
                    f"**🚨 Drift Count:** {st.session_state['drift_count']} ({'⚠️ Drift!' if drift_detected else '✅ Stable'})"
                )
                # ML prediction (safe)
                if ml_ready:
                    pred_label, prob, debug_info = safe_ml_predict(latest)
                    if pred_label:
                        if prob is not None:
                            fatigue_text.markdown(f"**💤 Fatigue (Rule vs ML):** {fatigue_status} | {pred_label} ({prob*100:.1f}%)")
                        else:
                            fatigue_text.markdown(f"**💤 Fatigue (Rule vs ML):** {fatigue_status} | {pred_label}")
                    else:
                        fatigue_text.markdown(f"**💤 Fatigue (Rule):** {fatigue_status}")
                        # debug info visible for troubleshooting
                        st.code(debug_info)
                else:
                    fatigue_text.markdown(f"**💤 Fatigue Level (Rule):** {fatigue_status}")

    # snapshot display
    snapshot = fetch_snapshot()
    if snapshot:
        # pass raw bytes to st.image (Streamlit will detect image)
        try:
            image_slot.image(snapshot, width="stretch")
        except Exception:
            # fallback to placeholder if display fails
            image_slot.image(placeholder_image, width="stretch")
    else:
        image_slot.image(placeholder_image, width="stretch")

else:
    st.markdown("---")
    render_session_summary(st.session_state["data_cache"])

# ----------------- Debug Panel -----------------
with st.expander("Backend & Status", expanded=False):
    st.write("Backend base URL:", BACKEND_BASE)
    st.write("Streaming running:", st.session_state["running"])
    st.write("Cached datapoints:", len(st.session_state["data_cache"]))
    st.write("Drift Count:", st.session_state["drift_count"])
    st.write("ML ready:", ml_ready)
    if ml_load_err:
        st.write("ML load error (if any):")
        st.code(ml_load_err)
