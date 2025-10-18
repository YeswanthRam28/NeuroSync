import subprocess
import json

def run_emotion(image_path):
    cmd = [
        "D:/Projects/NeuroSync/deepface_env/Scripts/python.exe",
        "backend/routes/emotion.py",
        image_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout)
