from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from app import call_model_with_retry

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/cobra/run")
def run_cobra(payload: dict):
    return call_model_with_retry(
        prompt=payload["prompt"],
        expected_domain=payload["expected_domain"],
        expected_phase=payload["expected_phase"],
        symbol_universe=payload.get("symbol_universe"),
        strict_schema=True,
    )



