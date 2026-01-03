from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Serve static files (css/js/images if you have them)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ROOT → serve your existing index.html
@app.get("/")
def root():
    return FileResponse("index.html")

# API stays exactly as-is
@app.post("/cobra/run")
def run_cobra(payload: dict):
    return {"status": "ok"}




