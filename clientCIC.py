import warnings
from collections import OrderedDict
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np


warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(78, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x


def train(net, trainloader, epochs):
    
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for features, labels in tqdm(trainloader):
            optimizer.zero_grad()
            outputs = net(features.to(DEVICE))
            loss = criterion(outputs, labels.to(DEVICE))
            loss.backward()
            optimizer.step()


def test(net, testloader):
    
    criterion = torch.nn.BCELoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for features, labels in tqdm(testloader):
            outputs = net(features.to(DEVICE))
            loss += criterion(outputs, labels.to(DEVICE)).item()
            predicted = torch.round(outputs).to(torch.int)
            correct += (predicted == labels.to(DEVICE)).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def load_data():
    data_dir = "./DatasetCIC/"
    train_file = data_dir + "CICentrenamiento.csv"
    test_file = data_dir + "CICprueba.csv"

    # Load training data
    train_data = pd.read_csv(train_file, nrows=600000)
    train_data = train_data.dropna()
    train_features = train_data.iloc[:, :-1].values
    train_labels = train_data.iloc[:, -1].values.reshape(-1, 1)  # Reshape labels

    # Load test data
    test_data = pd.read_csv(test_file, nrows=30000)
    test_features = test_data.iloc[:, :-1].values
    test_labels = test_data.iloc[:, -1].values.reshape(-1, 1)  # Reshape labels

    # Clean data
    train_features = np.nan_to_num(train_features)
    test_features = np.nan_to_num(test_features)
    train_features = np.where(np.isfinite(train_features), train_features, 0)
    test_features = np.where(np.isfinite(test_features), test_features, 0)

    # Normalize features
    scaler = MinMaxScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    # Convert to PyTorch tensors
    trainset = torch.utils.data.TensorDataset(
        torch.tensor(train_features, dtype=torch.float),
        torch.tensor(train_labels, dtype=torch.float),
    )
    testset = torch.utils.data.TensorDataset(
        torch.tensor(test_features, dtype=torch.float),
        torch.tensor(test_labels, dtype=torch.float),
    )

    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)



net = Net().to(DEVICE)
trainloader, testloader = load_data()


# Client Flower
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        y_true, y_pred = [], []
        with torch.no_grad():
            for features, labels in tqdm(testloader):
                outputs = net(features.to(DEVICE))
                predicted = torch.round(outputs).to(torch.int)
                y_true.extend(labels.to(DEVICE).tolist())
                y_pred.extend(predicted.tolist())
        avg_recall = recall_score(y_true, y_pred)
        avg_f1_score = f1_score(y_true, y_pred)
        print("Average Recall:", avg_recall)
        print("Average F1-score:", avg_f1_score)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


# Inicialización Cliente Flower
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)
