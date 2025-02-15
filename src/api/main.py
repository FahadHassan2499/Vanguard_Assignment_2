from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import logging

app = FastAPI()
logger = logging.getLogger("uvicorn")

# Pydantic models
class TrainingConfig(BaseModel):
    model_type: str = "logistic_regression"
    param_space: dict = {"C": [0.1, 1, 10]}

class TextRequest(BaseModel):
    text: str

# Load model from MLflow
def load_model():
    try:
        return mlflow.pyfunc.load_model(f"models:/prod_model/production")
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Model loading failed")

@app.post("/train")
async def train_model(config: TrainingConfig):
    try:
        # Trigger training process
        # Implement your training logic here
        return {"message": "Training started", "run_id": "some_run_id"}
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/best_params")
async def get_best_params():
    try:
        # Retrieve best params from MLflow
        client = mlflow.tracking.MlflowClient()
        production_model = client.get_latest_versions("prod_model", stages=["Production"])[0]
        return {"best_params": production_model.run_id}
    except Exception as e:
        logger.error(f"Param retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(request: TextRequest):
    try:
        model = load_model()
        prediction = model.predict([request.text])
        return {"text": request.text, "sentiment": prediction[0]}
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))