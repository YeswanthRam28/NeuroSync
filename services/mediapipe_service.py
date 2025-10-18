import subprocess
import json

def run_focus(input_data):
    cmd = [
        "D:/Projects/NeuroSync/mediapipe_env/Scripts/python.exe",
        "backend/routes/focus.py",
        json.dumps(input_data)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout)
