import os, asyncio
import json
import numpy as np
from pydantic import Field
import mlflow
import torch
import bentoml
from bentoml.exceptions import BentoMLException, InternalServerError
import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

class MyCustomException(BentoMLException):
    error_code = 501

# Image/Enviornment for the service.
my_image = bentoml.images.Image(base_image="python:3.11-slim", distro="debian").requirements_file("requirements.txt")

# Deploy the ML model as async endpoint
@bentoml.service(name = "ModelServer", image=my_image, traffic={
        "concurrency": 200,
    }
)
class DurationModel:
    def __init__(self):
        required_vars = ["MODEL_NAME", "DATABRICKS_HOST", "DATABRICKS_TOKEN"]
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        if missing_vars:
            raise MyCustomException(f"Missing required environment variables: {missing_vars}")
        
        mlflow.set_tracking_uri("databricks")

        model_name = os.environ.get("MODEL_NAME")
        alias = os.environ.get("ALIAS", "Production")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = mlflow.pytorch.load_model(f"models:/{model_name}@{alias}", map_location="cpu").to(self.device)
        self.model.eval()

    
    @bentoml.api(route="/model_predict", batchable=True, max_batch_size = 200, max_latency_ms = 100)
    def predict(self, inputs):
        try:
            with torch.no_grad():
                inputs = torch.tensor(inputs, device=self.device, dtype=torch.float32)
                pred = self.model(inputs)
            return pred.cpu().tolist()
        except Exception as err:
            logging.error(f"Inference failed: {str(err)}", exc_info=True)
            raise InternalServerError("Internal Server Error")


@bentoml.service(name = "TaxiDurationService", image=my_image, resources={"cpu": "4"})
class TaxiDuration:
    model = bentoml.depends(DurationModel)

    @bentoml.api(route="/async_predict")
    async def async_predict(self, inputs: np.ndarray = Field(examples=[[0.1, 0.4, 0.2, 1.0]])) -> dict:
        try:
            pred = await self.model.to_async.predict(inputs)
            return {"Predction":pred}
        except Exception as err:
            logging.error(f"Prediction failed: {str(err)}", exc_info=True)
            raise InternalServerError("Internal Server Error")

    @bentoml.api(route="/predict")
    def predict(self, inputs: np.ndarray = Field(examples=[[0.1, 0.4, 0.2, 1.0]])) -> dict:
        try:
            pred = self.model.predict(inputs)
            return {"Predction":pred}
        except Exception as err:
            logging.error(f"Prediction failed: {str(err)}", exc_info=True)
            raise InternalServerError("Internal Server Error")
        
    @bentoml.api(route="/health")
    def health_check(self):
        try:
            if hasattr(self.model, 'model') and self.model.model is not None:
                return {"status": "healthy", "model_loaded": True}
            else:
                return {"status": "unhealthy", "model_loaded": False}
        except Exception:
            return {"status": "unhealthy", "error": "Health check failed"}

