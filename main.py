import logging

from src.models import SimpleMNISTClassifier

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="INFO:     %(message)s")
app = FastAPI()

### --- GLOBAL SETUP --- ###

# List of allowed origins
origins = [
    "http://localhost:3000"
]

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate the model
model = SimpleMNISTClassifier(force_gpu=True)

### --- ROUTES --- ###
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/train")
async def train():
    model.load_data()
    model.train()
    return {"message": "Training complete"}

@app.get("/evaluate")
async def evaluate():
    accuracy = model.evaluate()
    return {"accuracy": accuracy}

@app.get("/run")
async def run():
    model.run()
    return {"message": model.eval_history}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=5000, timeout_graceful_shutdown=10)