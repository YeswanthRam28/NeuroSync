# llm/router.py
from llm.loader import generate

# ============================================================
# SYSTEM INSTRUCTIONS (short, efficient)
# ============================================================
SYSTEM_COACH = (
    "You are NeuroSync, a cognitive state coach. "
    "Reply in ONE short sentence. "
    "Tone: encouraging, calm, direct. "
    "DO NOT explain or ramble. "
    "Always give actionable advice."
)

SYSTEM_SUMMARY = (
    "You are NeuroSync, an AI that generates brief study session summaries. "
    "Keep summary concise, friendly, and focused on improvement. "
    "No paragraphs longer than 2 sentences."
)

SYSTEM_CHAT = (
    "You are NeuroSync, an assistant who answers clearly and briefly. "
    "Avoid long paragraphs. Be helpful and direct."
)

# ============================================================
# EVENT HANDLERS
# ============================================================

def handle_focus_drop(data):
    focus = data.get("focus")
    fatigue = data.get("fatigue")
    blink = data.get("blink")

    prompt = f"""
{SYSTEM_COACH}
Focus dropped to {focus}%. Fatigue: {fatigue}. Blink rate: {blink}.
Give one short encouragement or adjustment reminder.
"""
    return generate(prompt, max_tokens=60)


def handle_distraction(data):
    gaze_off = data.get("gaze_off")
    head = data.get("head_angle")

    prompt = f"""
{SYSTEM_COACH}
User gaze drift detected (off {gaze_off} sec). Head angle {head}.
Give one short refocus cue.
"""
    return generate(prompt, max_tokens=60)


def handle_emotion_shift(data):
    emo = data.get("emotion")

    prompt = f"""
{SYSTEM_COACH}
Emotion changed to {emo}.
Give one adaptive emotional response to support focus.
"""
    return generate(prompt, max_tokens=60)


def handle_session_end(data):
    focus = data.get("focus_summary", {})
    fatigue = data.get("fatigue_summary", {})
    drifts = data.get("drifts", 0)

    prompt = f"""
{SYSTEM_SUMMARY}

Focus: avg {focus.get('avg')}, max {focus.get('max')}, min {focus.get('min')}, stability {focus.get('sd')}.
Fatigue curve avg {fatigue.get('avg')}, max {fatigue.get('max')}.
Total drifts: {drifts}.

Write a clear ~4 sentence summary.
"""
    return generate(prompt, max_tokens=200)


def handle_chat(message):
    prompt = f"""
{SYSTEM_CHAT}
User: {message}
Assistant:
"""
    return generate(prompt, max_tokens=200)

# ============================================================
# EVENT ROUTER
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


#===============================================================
# Direct chat API used by server
#===============================================================
def route_chat(message):
    return handle_chat(message)
# ============================================================