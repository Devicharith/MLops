from mlflow.tracking import MlflowClient
import mlflow.pytorch
import torch
from torch import nn
import pandas as pd

test_data = pd.read_csv("data/test_data_V2.csv")

# mlflow.set_tracking_uri("http://localhost:8080")
mlflow.login()
client = MlflowClient()

# Replace with your model name and version
model_name = "dev.models.Taxi_Durations"
version = 2 # or v1, v2, etc.

# Fetch Params of the model run
model_version = client.get_model_version(name=model_name, version=str(version))
run_id = model_version.run_id

# Fetch run info and Access parameters
run = client.get_run(run_id)
params = run.data.params

# load model
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = mlflow.pytorch.load_model(f"models:/{model_name}/{version}").to(device)
loss_fn = nn.MSELoss()

test_X = torch.tensor(test_data.loc[:, test_data.columns!=params["target_col"]].values, device=device, dtype=torch.float32)             
test_y = torch.tensor(test_data[params["target_col"]], device = device, dtype = torch.float32)   

model.eval()
with torch.no_grad():
    predict = model(test_X)
    test_loss = loss_fn(predict, test_y.view(-1, 1))
                  
    # Now log test_loss to that run
    client.log_metric(run_id, key="test_loss", value=test_loss.item())
    print(f"Test loss: {test_loss.item()}")

