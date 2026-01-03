from fastapi import FastAPI
from fastapi.responses import FileResponse
from app import call_model_with_retry   # import your real logic

app = FastAPI()

@app.get("/")
def root():
    return FileResponse("index.html")

@app.post("/cobra/run")
def run_cobra(payload: dict):
    return call_model_with_retry(
        prompt=payload["prompt"],
        expected_domain=payload["expected_domain"],
        expected_phase=payload["expected_phase"],
        symbol_universe=payload["symbol_universe"],
        strict_schema=True,
    )



