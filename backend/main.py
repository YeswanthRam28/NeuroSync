from fastapi import FastAPI
from backend.routes import emotion, focus

app = FastAPI(title="NeuroSync API", version="1.0")

app.include_router(emotion.router, prefix="/emotion", tags=["Emotion Tracking"])
app.include_router(focus.router, prefix="/focus", tags=["Focus Tracking"])

@app.get("/")
async def root():
    return {"message": "Welcome to NeuroSync API"}
