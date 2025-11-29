from ..loader import llm
from ..memory import add_memory

def generate_summary(data):
    prompt = f"""
Session Summary Request:

Focus trend: {data['focus_trend']}
Distractions: {data['drifts']}
Fatigue curve: {data['fatigue_curve']}

Create a 5-line session summary with:
- What went well
- What went wrong
- Insights
- Advice
- One improvement tip
"""

    out = llm(prompt, max_tokens=200)
    reply = out["choices"][0]["text"].strip()
    add_memory("session_summary", reply)
    return reply
