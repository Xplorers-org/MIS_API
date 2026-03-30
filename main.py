from fastapi import FastAPI

app = FastAPI(
    title="Motor Impairment Score API",
    description="API for predicting Parkinson's motor impairment from drawings."
)

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Welcome to the Motor Impairment Score API"
    }