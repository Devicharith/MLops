import os, asyncio
import json
import numpy as np
from pydantic import Field
import mlflow
import torch
import bentoml
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")


@bentoml.service(name = "TaxiDurationService", resources={"cpu": "4"})
class TaxiDuration:
    def __init__(self):
        model_name = os.environ.get("MODEL_NAME")
        alias = os.environ.get("ALIAS", "Production")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # mlflow.set_tracking_uri(os.environ.get("DATABRICKS_HOST"))
        # mlflow.set_registry_uri("databricks")

        mlflow.login()
        self.model = mlflow.pytorch.load_model(f"models:/{model_name}@{alias}").to(self.device)
        self.model.eval()

    @bentoml.api(route="/async_predict")
    async def async_predict(self, inputs: np.ndarray = Field(examples=[[0.1, 0.4, 0.2, 1.0]])) -> dict:
        with torch.no_grad():
            inputs = torch.tensor(inputs, device=self.device, dtype=torch.float32)
            pred = await asyncio.get_event_loop().run_in_executor(None, lambda: self.model(inputs))
        return {"Predction": pred.cpu().tolist()}
        
    @bentoml.api(route="/predict")
    def predict(self, inputs: np.ndarray = Field(examples=[[0.1, 0.4, 0.2, 1.0]])) -> dict:
        with torch.no_grad():
            inputs = torch.tensor(inputs, device=self.device, dtype=torch.float32)
            pred = self.model(inputs)
        return {"Predction": pred.cpu().tolist()}