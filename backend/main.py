from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # Import the middleware
from backend.routes import emotion, focus

app = FastAPI(title="NeuroSync API", version="1.0")

# --- CORS Configuration ---
# Allows all origins, methods, and headers. This is necessary because
# the frontend (dashboard.html) is running from a different origin
# (usually 'null' or 'file://') than the backend (http://127.0.0.1:8000).
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --------------------------

app.include_router(emotion.router, prefix="/emotion", tags=["Emotion Tracking"])
app.include_router(focus.router, prefix="/focus", tags=["Focus Tracking"])

@app.get("/")
async def root():
    return {"message": "Welcome to NeuroSync API"}
