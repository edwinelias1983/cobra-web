from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
# from app import call_model_with_retry
import os
import uvicorn

app = FastAPI()

@app.get("/")
def root():
    return FileResponse("index.html")

@app.post("/cobra/run")
def run_cobra(payload: dict):
    from app import call_model_with_retry
    return {"status": "ok"}




