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





warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(42, 32)
        #self.fc2 = nn.Linear(64, 32)
        #self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
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
    """Load dataset (training and test set) from CSV files."""
    data_dir = "./DatasetUNSW/"
    train_file = data_dir + "unswentrenamiento.csv"
    test_file = data_dir + "unswprueba.csv"

    # Leer el archivo de entrenamiento
    df_train = pd.read_csv(train_file)

    # Eliminar la columna "attack_cat" del dataset de entrenamiento
    df_train = df_train.drop("attack_cat", axis=1)

    # Obtener las características y etiquetas del dataset de entrenamiento
    train_features = df_train.iloc[:, :-1].values
    train_labels = df_train.iloc[:, -1].values.reshape(-1, 1)  

    # Crear el conjunto de datos de entrenamiento
    trainset = torch.utils.data.TensorDataset(
        torch.tensor(train_features, dtype=torch.float),
        torch.tensor(train_labels, dtype=torch.float),
    )

    # Leer el archivo de prueba
    df_test = pd.read_csv(test_file)

    # Eliminar la columna "attack_cat" del dataset de prueba
    df_test = df_test.drop("attack_cat", axis=1)

    # Obtener las características y etiquetas del dataset de prueba
    test_features = df_test.iloc[:, :-1].values
    test_labels = df_test.iloc[:, -1].values.reshape(-1, 1)  # Reshape labels

    # Crear el conjunto de datos de prueba
    testset = torch.utils.data.TensorDataset(
        torch.tensor(test_features, dtype=torch.float),
        torch.tensor(test_labels, dtype=torch.float),
    )

    # Crear los dataloaders para el entrenamiento y prueba
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset)

    return trainloader, testloader




net = Net().to(DEVICE)
trainloader, testloader = load_data()


# Cliente Flower
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
