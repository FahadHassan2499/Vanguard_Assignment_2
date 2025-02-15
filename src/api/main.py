from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import mlflow.pyfunc
import logging

from src.data.data_loader import load_and_preprocess_data
from src.models.train_model import train_model

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
async def train_model_endpoint(config: TrainingConfig, background_tasks: BackgroundTasks):
    try:
        X_train, _, y_train, _ = load_and_preprocess_data()

        def train_task():
            train_model(X_train, y_train, config.model_type, config.param_space)
        background_tasks.add_task(train_task) # Run training in the background

        return {"message": "Training started in the background"}
    except Exception as e:
         # ... (improved error handling)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


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