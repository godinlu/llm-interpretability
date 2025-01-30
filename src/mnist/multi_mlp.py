"""
Ce script permet de créer un jeux de données petit à petit.
Le jeux de données contient des activations de neuronnes lors de prédictions de nombre par modèles.
Chaque modèle est entraîné sur 5 épochs puis évalué 1 fois sur chaque nombre et on sauvegarde
dans le fichier de sortie les 10 activations.

Ce script peut être lancé plusieurs fois pour remplir petit à petit le fichier de sortie.
Il s'execute indéfiniment et peut être arrêté sans problème avec Ctrl-C
"""
import torch
import pyrallis
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import csv
import os
import signal
import sys
from dataclasses import dataclass

@dataclass
class Config:
    # Nombre d'épochs de l'entraînement
    epochs: int = 5

    # Cette variable indique les nombres de neuronnes et le nombre de couche caché 
    # ici [128, 64] indique 2 couches caché avec une couche de 128 neuronnes et une autres de 64 
    hidden_sizes = [128, 64] 

    # learning_rate pour l'apprentissage
    learning_rate: float = 0.001

    # taille des batchs
    batch_size: int = 64

    # chemin du fichier de sortie
    output_file: str = "../../data/activations/multi_mlp.csv"


class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[128, 64], output_size=10):
        super(MLP, self).__init__()
        self.hidden_layers = nn.ModuleList()
        in_features = input_size
        for h in hidden_sizes:
            self.hidden_layers.append(nn.Linear(in_features, h))
            in_features = h
        self.output_layer = nn.Linear(in_features, output_size)
        self.activations = []

    def forward(self, x):
        self.activations = []  # Réinitialiser les activations à chaque passage
        x = x.view(x.size(0), -1)  # Aplatir l'image en vecteur
        for layer in self.hidden_layers:
            x = layer(x)
            x = torch.relu(x)
            self.activations.append(x.clone())  # Sauvegarder les activations après ReLU
        x = self.output_layer(x)
        return x
    

def handle_sigint(signal_received, frame):
    """
    Cette fonction est appelé pour capturer le Ctrl-C et ferme proprement
    """
    print("\nInterruption détectée ! Sauvegarde et sortie propre...")
    sys.exit(0)


def load_data(cfg:Config) -> tuple[DataLoader, DataLoader]:
    """
    importe les données de MNIST avec un train et un test.
    """
    # Préparer les données MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normaliser entre -1 et 1
    ])

    train_dataset = datasets.MNIST(root='../../data', train=True, transform=transform, download=False)
    test_dataset = datasets.MNIST(root='../../data', train=False, transform=transform, download=False)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    return (train_loader, test_loader)


def train_model(train_loader: DataLoader, cfg: Config) -> MLP:
    """
    Créer un MLP et l'entraîne sur les données d'entraînement.
    """
    model = MLP(hidden_sizes = cfg.hidden_sizes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Époch {epoch+1}/{cfg.epochs}, Perte: {running_loss/len(train_loader)}")
    
    return model

def evaluate_model(model: MLP, test_loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cette fonction utilise le modèle model en paramètre pour faire des prédiction sur le jeux de données d'entraînement
    test_loader. Cette fonction renvoie 10 activations de neuronnes, 1 pour chaque label.
    """
    model.eval()
    all_activations = []
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.numpy())
            all_labels.extend(labels.numpy())
            
            # Collecter les activations des couches cachées
            batch_activations = [activation.numpy() for activation in model.activations]
            # Transposer pour obtenir (échantillons, activations par couche)
            batch_activations = np.concatenate(batch_activations, axis=1)  # Concatène les activations des couches
            all_activations.extend(batch_activations)

    all_predictions = np.array(all_predictions)
    all_activations = np.vstack(all_activations)
    all_labels = np.array(all_labels)

    indices = [np.where(all_predictions == i)[0][np.random.randint(0, np.sum(all_predictions == i))] for i in range(10)]

    return (all_labels[indices], all_predictions[indices], all_activations[indices])



if __name__ == "__main__":
    cfg: Config = pyrallis.parse(config_class=Config)
    signal.signal(signal.SIGINT, handle_sigint)

    train_loader, test_loader = load_data(cfg)

    with open(cfg.output_file, "a", newline="") as file:
        writer = csv.writer(file)

        # si le fichier est vide alors on écrit l'entête.
        if file.tell() == 0:
            header = ["nb_true", "nb_pred"] + [f"activation_{i}" for i in range(sum(cfg.hidden_sizes))]
            writer.writerow(header)

        try:
            while True:
                model = train_model(train_loader, cfg)
                labels, predictions, activations = evaluate_model(model, test_loader)
                
                # écriture des résultats dans le fichier CSV
                for i in range(len(labels)):
                    row = np.concatenate(([labels[i], predictions[i]], activations[i] ))
                    writer.writerow(row)
                    file.flush() # on sauvegarde immédiatement
                
                print(f"Données sauvegardés dans '{cfg.output_file}' !")
        except KeyboardInterrupt:
            print("\nInterruption par l'utilisateur. Données sauvegardées !")
            

        

    
