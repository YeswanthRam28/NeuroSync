from ..loader import llm
from ..memory import add_memory

def generate_emotion_feedback(data):
    prompt = f"""
Current emotion: {data['emotion']}
Trend: {data['trend']}

Give a soft, motivational 1-sentence message.
"""

    out = llm(prompt, max_tokens=60)
    reply = out["choices"][0]["text"].strip()
    add_memory(str(data), reply)
    return reply
