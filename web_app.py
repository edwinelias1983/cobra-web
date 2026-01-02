from fastapi import FastAPI, HTTPException
from app import call_model_with_retry
import os
import uvicorn

app = FastAPI()

@app.post("/cobra/run")
def run_cobra(payload: dict):
    return {"status": "ok"}


