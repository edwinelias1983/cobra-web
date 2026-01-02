from fastapi import FastAPI, HTTPException
from app import call_model_with_retry

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
