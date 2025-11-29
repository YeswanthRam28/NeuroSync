# llm/router.py
from llm.loader import generate

# ============================================================
# SYSTEM INSTRUCTIONS (optimized for Gemma)
# ============================================================
COACH = (
    "You are NeuroSync, a real-time cognitive coach. "
    "Speak in one short, actionable sentence. "
    "Tone: calm, supportive, direct."
)

EMO_COACH = (
    "You are NeuroSync, an emotional adaptation coach. "
    "Reply with one supportive and focus-oriented sentence."
)

SUMMARY = (
    "You are NeuroSync, an AI that creates short study-session summaries. "
    "Write up to four concise, helpful sentences. "
    "Keep it warm, realistic, and focused on improvement."
)

CHAT = (
    "You are NeuroSync, a helpful assistant. "
    "Reply clearly and briefly in under five sentences."
)

# ============================================================
# EVENT HANDLERS (Gemma-friendly prompts)
# ============================================================

def handle_focus_drop(data):
    focus = data.get("focus")
    fatigue = data.get("fatigue")
    blink = data.get("blink")

    prompt = (
        f"{COACH}\n"
        f"Focus dropped to {focus}%. Fatigue: {fatigue}. Blink rate: {blink}.\n"
        f"Give one short refocus cue."
    )
    return generate(prompt, max_tokens=60)


def handle_distraction(data):
    gaze_off = data.get("gaze_off")
    head = data.get("head_angle")

    prompt = (
        f"{COACH}\n"
        f"Gaze drift detected for {gaze_off} seconds. Head angle: {head}.\n"
        f"Give one direct refocus reminder."
    )
    return generate(prompt, max_tokens=60)


def handle_emotion_shift(data):
    emotion = data.get("emotion")

    prompt = (
        f"{EMO_COACH}\n"
        f"Detected emotion shift: {emotion}.\n"
        f"Give one supportive sentence to stabilize focus."
    )
    return generate(prompt, max_tokens=60)


def handle_session_end(data):
    f = data.get("focus_summary", {})
    fat = data.get("fatigue_summary", {})
    drifts = data.get("drifts", 0)

    prompt = (
        f"{SUMMARY}\n"
        f"Focus avg {f.get('avg')}, max {f.get('max')}, min {f.get('min')}, "
        f"stability SD {f.get('sd')}.\n"
        f"Fatigue avg {fat.get('avg')}, max {fat.get('max')}.\n"
        f"Drifts detected: {drifts}.\n"
        f"Write a 3–4 sentence summary."
    )
    return generate(prompt, max_tokens=200)


def handle_chat(message):
    prompt = (
        f"{CHAT}\n"
        f"User: {message}\n"
        f"Assistant:"
    )
    return generate(prompt, max_tokens=200)


# ============================================================
# EVENT ROUTER TABLE
# ============================================================
ROUTE_TABLE = {
    "focus_drop": handle_focus_drop,
    "distraction": handle_distraction,
    "emotion_shift": handle_emotion_shift,
    "session_end": handle_session_end,
}


def route_event(event_type, data):
    handler = ROUTE_TABLE.get(event_type)
    if handler:
        return handler(data)
    return "Event not recognized."


# ============================================================
# Direct chat API for server
# ============================================================
def route_chat(message):
    return handle_chat(message)
