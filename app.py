# app.py (root del repo)
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok", "message": "AstroBot online 🪐"}
