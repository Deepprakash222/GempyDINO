mport torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

dtype = torch.float32


    
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
    
def L2_accuaracy(true_Data, nueral_network_output):
    '''
        Find the expectation of relative error norm
    '''
    RL=0
    for i in range(true_Data.shape[0]):
        RL += (torch.norm(true_Data[i] - nueral_network_output[i])**2)/ (torch.norm(true_Data[i])**2)
    Expecation_RL = RL/true_Data.shape[0]
    L2_error = 1- torch.sqrt(Expecation_RL)
    return L2_error

def custom_loss_without_grad(output_PDE, outputs_NN):
    """
    
    Computes the loss: sum over N samples of
    ||output_PDE_i - output_NN_i||^2_2 
    
    """
    N, D = outputs_NN.shape
    l2_norm = torch.norm(output_PDE - outputs_NN, p=2, dim=1) ** 2  # Squared L2 norm for each sample
    
    total_loss = torch.sum(l2_norm) /N #(N*D)  # Sum over all N samples (N *D)
    return total_loss

def calulate_matrix_norm_square(output_grad_PDE, output_final):
    """
    Computes the loss: sum over N samples of
    ||output_grad_PDE - output_final||^2_2 
    
    """
    A = output_final - output_grad_PDE
    N, D, _ = A.shape
    frob_norm = (torch.norm(A, p='fro', dim=(1, 2)) ** 2) / N #(N * D * D)
    return frob_norm

def calculate_jacobian(U_k, inputs, outputs_NN):
    """
    Computes the Jacobian matrix of the neural network output with respect to the input.
    """
    
    # Ensure input requires gradients
    # outputs_NN = model(inputs)
    outputs = (U_k.T @ outputs_NN.T).T
    
    
    #print(outputs.shape)
    jacobian_list = []
    for i in range(outputs.shape[1]):  # Loop over each output dimension
        grad_outputs = torch.zeros_like(outputs)
        grad_outputs[:, i] = 1.0  # Compute gradient for one output at a time

        # Retain graph so it can be used for loss.backward()
        jacobian_row = torch.autograd.grad(outputs, inputs, grad_outputs=grad_outputs,
                                        retain_graph=True, create_graph=True)[0]

        jacobian_list.append(jacobian_row)
    
    jacobian_matrix = torch.stack(jacobian_list, dim=1)  # Shape: [batch_size, output_dim, input_dim]
    
    return jacobian_matrix
    
def train_network(Gempy_Inputs, PDE_outputs, Jacobian, layer_sizes, num_epochs):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ######################################################################################
    # Convert the dataset into pytorch tensor
    ######################################################################################
    Gempy_Inputs = torch.tensor(Gempy_Inputs, dtype=dtype, device=device)
    PDE_outputs = torch.tensor(PDE_outputs, dtype=dtype, device=device)
    Jacobian = torch.tensor(Jacobian, dtype=dtype, device=device)
    ######################################################################################
    # Create the Dataset for Neural network
    ######################################################################################
    dataset = TensorDataset(Gempy_Inputs, PDE_outputs, Jacobian)
    
    N = Gempy_Inputs.shape[0]
    train_size = int(0.6 * N)
    valid_size = int(0.2 * N)
    test_size = N - train_size - valid_size

    # Randomly split dataset
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)
    
    # Check sizes
    print(f"Train size: {len(train_dataset)},Valid size: {len(valid_dataset)},  Test size: {len(test_dataset)}")
    
    ######################################################################################
    # Instantiate model
    ######################################################################################
    model = NeuralNet(layer_sizes=layer_sizes)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    
    # -------------------------------
    # Training Loop with Loss Tracking
    # -------------------------------
    num_epochs = num_epochs
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        epoch_train_loss = 0

        for inputs, output_PDE , output_grad_PDE in train_loader:
            inputs, output_PDE , output_grad_PDE = inputs.float().to(device), output_PDE.float().to(device) , output_grad_PDE.float().to(device)  # Ensure float32
            optimizer.zero_grad()
            outputs_NN = model(inputs)  # Forward pass
            loss = custom_loss_without_grad(output_PDE, outputs_NN) 
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            epoch_train_loss += loss.item()
        
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # -------------------------------
        # Evaluate on Validation (Test) Set
        # -------------------------------
        model.eval()  # Set model to evaluation mode
        epoch_val_loss = 0

        with torch.no_grad():
            for inputs, output_PDE , output_grad_PDE in valid_loader:
                inputs, output_PDE , output_grad_PDE = inputs.float().to(device), output_PDE.float().to(device) , output_grad_PDE.float().to(device)  # Ensure float32
                outputs_NN = model(inputs)
                loss = custom_loss_without_grad(output_PDE, outputs_NN)
                epoch_val_loss += loss.item()
                
        # Compute average validation loss
        epoch_val_loss /= len(test_loader)
        val_losses.append(epoch_val_loss)

        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")