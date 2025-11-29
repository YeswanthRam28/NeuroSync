from ..loader import llm
from ..memory import add_memory

def generate_study_plan(data):
    prompt = f"""
You are a personalized study planner.

Best focus times: {data['best_hours']}
Fatigue pattern: {data['fatigue']}
Tomorrow free slots: {data['schedule']}

Create a smart study timetable in bullet points.
"""

    out = llm(prompt, max_tokens=200)
    reply = out["choices"][0]["text"].strip()
    add_memory("study_plan", reply)
    return reply
