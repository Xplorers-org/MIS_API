from fastapi import FastAPI
from services import predictor
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):

    # Load models on startup

    print("\n--- SYSTEM BOOT ---")
    print("Pre-loading deep learning models from the Hub into RAM...\n")

    predictor._get_wave_model()
    predictor._get_spiral_model()    


    print("Models successfully cached! Server is now ready to accept network traffic.\n")

    yield
    # Everything before 'yield' happens BEFORE the server accepts traffic.

    print("\n--- SYSTEM SHUTDOWN ---")
    print("Releasing memory and shutting down...")




app = FastAPI(
    title="Motor Impairment Score API",
    description="API for predicting Parkinson's motor impairment from drawings.",
    lifespan=lifespan
)

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Welcome to the Motor Impairment Score API"
    }