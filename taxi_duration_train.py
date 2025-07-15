import torch
import mlflow
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.init as init
from torchinfo import summary
import torch.optim as optim
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import sys

class TeeLogger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.error = sys.stderr
        self.log = log_file

    def write(self, message):
        self.terminal.write(message)
        self.error.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.error.flush()
        self.log.flush()


class Taxi_Dataset(Dataset):
    def __init__(self, data, target_col, device):
        super().__init__()
        self.dataset = data
        self.target_label = target_col
        self.device = device
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = torch.tensor(self.dataset.iloc[idx][self.dataset.columns != self.target_label], dtype=torch.float32, device = self.device)
        target = torch.tensor(self.dataset.iloc[idx][self.target_label], dtype=torch.float32, device=self.device)

        return sample, target

# Model
class Taxi_NeuralNet(nn.Module):
    def __init__(self, n_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_layer = nn.Linear(n_features, 4*n_features, dtype=torch.float32)
        self.layer_norm1 = nn.LayerNorm(4*n_features)
        self.activation1 = nn.LeakyReLU()
        self.hidden_layer = nn.Linear(4*n_features, 2*n_features, dtype=torch.float32)
        self.layer_norm2 = nn.LayerNorm(2*n_features)
        self.activation2 = nn.LeakyReLU()
        self.output_layer = nn.Linear(2*n_features, 1, dtype=torch.float32)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.layer_norm1(x)
        x = self.activation1(x)
        x = self.hidden_layer(x)
        x = self.layer_norm2(x)
        x = self.activation2(x)
        x = self.output_layer(x)
        return x

class EarlyStopping:
    def __init__(self, patience = 5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss == None:
            # save the best weights
            torch.save(model.state_dict(), "weights/best_model.pth") 
            self.best_loss = val_loss

        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # save the best weights
            torch.save(model.state_dict(), "weights/best_model.pth")
        else:
            self.counter += 1
            if self.counter >= self.patience:
              self.early_stop = True

def train_TaxiNN(model, loss_fn, optimizer, train_dataloader, epochs, Val_X, Val_y, patience, min_delta):

    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    model.train()
    for epoch in range(epochs):
        training_loss, n_batches = 0, 0
        for X, y in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):

            # Clear past grads
            optimizer.zero_grad()

            # Forward pass
            predict = model(X)

            loss = loss_fn(predict, y.view(-1, 1))
            training_loss+=loss.item()
            n_batches+=1

            # compute grads
            loss.backward()

            # Backward pass
            optimizer.step()
        
        # Validation loss
        val_predict = model(Val_X)
        val_loss = loss_fn(val_predict, Val_y.view(-1, 1))

        # Log epoch metrics
        mlflow.log_metrics(
            {
                "train_loss": training_loss/n_batches,
                "val_loss": val_loss,
            },
            step = epoch,
        )
        print(F"Training loss: {training_loss/n_batches} Validation loss: {val_loss}")

        # Early Stopping
        early_stopping(val_loss.item(), model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    

def evaluate_TaxiNN(model, loss_fn, test_X, test_y=None):
    model.eval()
    with torch.no_grad():
        predict = model(test_X)

        test_loss = None
        if test_y is not None and test_y.numel() > 0:
            test_loss = loss_fn(predict, test_y.view(-1, 1))
        
    return test_loss, predict 

def initialize_weights(module):
    if isinstance(module, nn.Linear):
        init.kaiming_uniform_(module.weight, nonlinearity='relu')  # or 'leaky_relu', etc.
        if module.bias is not None:
            init.zeros_(module.bias)

    elif isinstance(module, nn.Conv2d):
        init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            init.zeros_(module.bias)

    elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
        init.ones_(module.weight)
        init.zeros_(module.bias)

# Main

# Local MLFlow
# mlflow.set_tracking_uri("http://localhost:8080")

# Databricks MLFlow
mlflow.login()
mlflow.set_experiment("/Users/devicharith12@gmail.com/Taxi-Durations")

log_file_path = "training_output.log"
with open(log_file_path, "w") as log_file:
    sys.stdout = TeeLogger(log_file)
    sys.stderr = TeeLogger(log_file)

    with mlflow.start_run() as run:
        try:
            params = {
            "epochs": 50,
            "learning_rate": 0.01,
            "batch_size": 32,
            "n_features": 14,
            "optimizer": "Adam",
            "validaion_size": 0.1,
            "test_size": 0.2,
            "device": "mps",
            "target_col" : "durations",
            "patience": 6,
            "min_delta": 0.01
            }

            train_data = pd.read_csv("data/training_data_V2.csv")

            # Log Dataset
            dataset = mlflow.data.from_pandas(train_data, source="data/training_data_V2.csv", name="TrainData-V2", targets=params["target_col"])
            mlflow.log_input(dataset, context="training")

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            print(device)

            train_df, val_df = train_test_split(train_data, test_size=0.1, random_state=42)

            # train data
            data = Taxi_Dataset(train_df, params["target_col"], device)
            train_dataloader = DataLoader(data, batch_size = params["batch_size"], shuffle=True)

            # Validation data
            Val_X = torch.tensor(val_df.loc[:, val_df.columns!=params["target_col"]].values, device=device, dtype=torch.float32)
            Val_y = torch.tensor(val_df[params["target_col"]].values, device = device, dtype = torch.float32)

            # Model Initialization
            model = Taxi_NeuralNet(n_features = params["n_features"]).to(device)
            model.apply(initialize_weights)
            # model = torch.compile(model)

            # Log model summary.
            with open("model_summary.txt", "w") as f:
                f.write(str(summary(model)))
            mlflow.log_artifact("model_summary.txt")

            # loss function
            loss_fn = nn.MSELoss()

            # Backprop optimizer
            optimizer = optim.Adam(model.parameters(), lr = params["learning_rate"])
            params["loss_function"] = loss_fn.__class__.__name__

            # Log training parameters.
            mlflow.log_params(params)

            # training
            train_TaxiNN(model=model, 
                        loss_fn=loss_fn, 
                        optimizer=optimizer, 
                        train_dataloader=train_dataloader, 
                        epochs = params["epochs"],
                        Val_X=Val_X, 
                        Val_y=Val_y, 
                        patience = params["patience"], 
                        min_delta = params["min_delta"])
            
            model.load_state_dict(torch.load("weights/best_model.pth"))

            # 5. Log the trained model
            input_example = torch.randn(2, params["n_features"], dtype=torch.float32).to('mps')
            output_example = model(input_example)

            # Create a signature
            signature = infer_signature(input_example.cpu().numpy(), output_example.detach().cpu().numpy())
            model_info = mlflow.pytorch.log_model(model, 
                                                  name = "Model-V2", 
                                                  input_example=input_example.cpu().numpy(), 
                                                  signature=signature,
                                                  registered_model_name="dev.models.Taxi_Durations")
            print()
            print("Training Done!!!!")

        except Exception as err:
            print(f"Error: {err}")

        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            mlflow.log_artifact(log_file_path)
