from ..loader import llm
from ..memory import retrieve_memories, add_memory
from ..config import TEMP, TOP_K, TOP_P

def generate_coach_advice(metrics):
    prompt = f"""
You are a realtime cognitive coach. Metrics:
Focus: {metrics['focus']}
Fatigue: {metrics['fatigue']}
Blink Rate: {metrics['blink']}
Gaze Stability: {metrics['gaze']}

Give ONE short actionable sentence of advice.
"""

    out = llm(
        prompt,
        max_tokens=80,
        temperature=TEMP,
        top_k=TOP_K,
        top_p=TOP_P,
    )

    reply = out["choices"][0]["text"].strip()
    add_memory(str(metrics), reply)
    return reply
