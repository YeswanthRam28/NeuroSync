from ..loader import llm
from ..memory import add_memory

def generate_distraction_msg(data):
    prompt = f"""
The user got distracted.

Gaze off-screen: {data['gaze_off']} seconds
Head angle: {data['head_angle']}

Give a friendly 1-sentence explanation or question.
"""

    out = llm(prompt, max_tokens=60)
    reply = out["choices"][0]["text"].strip()
    add_memory(str(data), reply)
    return reply
