from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "alive"}

@app.post("/cobra/run")
def run_cobra(payload: dict):
    return {"status": "ok"}




