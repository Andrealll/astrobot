from fastapi import FastAPI
from app_kb import kb_router

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok", "message": "AstroBot online 🪐"}

app.include_router(kb_router)
