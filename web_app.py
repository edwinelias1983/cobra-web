from fastapi import FastAPI, HTTPException
from app import call_model_with_retry
import os
import uvicorn

app = FastAPI()

@app.post("/cobra/run")
def run_cobra(payload: dict):
    try:
        result = call_model_with_retry(
            prompt=payload["prompt"],
            expected_domain=payload["expected_domain"],
            expected_phase=payload["expected_phase"],
            symbol_universe=payload["symbol_universe"],
            strict_schema=True,
        )
        return result

    except HTTPException:
        raise

    except Exception as e:
        print("COBRA ERROR:", repr(e))
        raise HTTPException(status_code=400, detail=str(e))


# ----------------------------
# Railway port fix for deployment
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Use Railway-assigned port
    uvicorn.run("web_app:app", host="0.0.0.0", port=port)


