import torch
import torch.nn as nn



# -------------------------------
# Step 1: Define Neural Network
# -------------------------------
    
def build_network(layer_sizes):
    layers = []
    
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))  # Fully connected layer
        if i < len(layer_sizes) - 2:  # No activation for last layer
            layers.append(nn.ReLU())  # Activation function (can be modified)
    
    return nn.Sequential(*layers)

class NeuralNet(nn.Module):
    def __init__(self, layer_sizes):
        super(NeuralNet, self).__init__()
        self.network = build_network(layer_sizes)  # Use the dynamic builder

    def forward(self, x):
        return self.network(x)