import torch
import torch.nn as nn
import numpy as np

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

def calculate_jacobian_full(inputs, outputs_NN):
    """
    Computes the Jacobian matrix of the neural network output with respect to the input.
    """
    
    # Ensure input requires gradients
    # outputs_NN = model(inputs)
    outputs =  outputs_NN
    
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
    
def train_network(Gempy_Inputs, PDE_outputs, Jacobian, layer_sizes, num_epochs,  network_type=None):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ###############################################################################
    # Seed the randomness 
    ###############################################################################
    seed = 42           
    torch.manual_seed(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    ######################################################################################
    # Convert the dataset into pytorch tensor
    ######################################################################################
    Gempy_Inputs = torch.tensor(Gempy_Inputs, dtype=dtype, device=device)
    PDE_outputs = torch.tensor(PDE_outputs, dtype=dtype, device=device)
    Jacobian = torch.tensor(Jacobian, dtype=dtype, device=device)
    
    ######################################################################################
    # Find U and V by taking expextation of Jacobian
    ######################################################################################
    N, r, m = Jacobian.shape
    result_grad_sum_left = torch.zeros(r,r, dtype=dtype, device=device)  # Initialize a tensor to accumulate the sum
    result_grad_sum_right = torch.zeros(m,m, dtype=dtype, device=device)
    
    for i in range(N):
        grad_u = Jacobian[i]
        result_grad_sum_left += grad_u @ grad_u.T
        result_grad_sum_right += grad_u.T @ grad_u
    # Compute the average
    # print(u_grad_list)
    cov_matrix_grad_m_u_left = result_grad_sum_left / N
    cov_matrix_grad_m_u_right = result_grad_sum_right / N
    
    print(cov_matrix_grad_m_u_left.shape, cov_matrix_grad_m_u_right.shape)
    # find the eigen_values of u 
    eigenvalues_grad_u_left, U = torch.linalg.eig(cov_matrix_grad_m_u_left.detach())
    eigenvalues_grad_u_right, V = torch.linalg.eig(cov_matrix_grad_m_u_right.detach())
    
    U = U.real
    V = V.real
    ######################################################################################
    # Create the Dataset for Neural network
    ######################################################################################
    dataset = TensorDataset(Gempy_Inputs, PDE_outputs, Jacobian)
    
   
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
    if network_type==None:
        
        model = NeuralNet(layer_sizes=layer_sizes)
        model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.002)
        
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
            if epoch % 10 == 0 :
                print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
            elif epoch ==(num_epochs-1):
                print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
            
            
        model.eval()  # Set model to evaluation mode
        
        for inputs, output_PDE , output_grad_PDE in test_loader:
            inputs, output_PDE , output_grad_PDE = inputs.float().to(device), output_PDE.float().to(device) , output_grad_PDE.float().to(device)  # Ensure float32
            inputs.requires_grad_(True)
            #jacobian_data, outputs_NN = jacobian_cal(inputs, model)
            outputs_NN = model(inputs)
            L_2_1 = L2_accuaracy(true_Data=output_PDE , nueral_network_output=outputs_NN)
            
            print("L2 accuaracy without Jacobian: ",L_2_1)
        
    ################################################################################
    # Find the result just based on L2 norm of output of NN and and reduced basis
    # and respespect to Derivative
    # || q(k) - f_\theta(k)||_2^2 + || D(q(k)) - D(f_\theta(k))||_2^2
    ################################################################################
    # Instantiate model
    elif network_type=="full":
        model = NeuralNet(layer_sizes=layer_sizes)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.002)
        train_losses = []
        val_losses = []
        L_2 = []
        F_2 = []
        a=1e0
        
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode
            epoch_train_loss = 0
            epoch_L2_loss = 0
            epoch_F2_loss = 0

            for inputs, output_PDE , output_grad_PDE in train_loader:
                inputs, output_PDE , output_grad_PDE = inputs.float().to(device), output_PDE.float().to(device) , output_grad_PDE.float().to(device)  # Ensure float32
                optimizer.zero_grad()
                # 🔥 Ensure inputs track gradients before computing Jacobian
                inputs.requires_grad_(True)
                outputs_NN = model(inputs)  # Forward pass
                L_2_loss = custom_loss_without_grad(output_PDE, outputs_NN)/2
                Jacobian_NN = calculate_jacobian_full(inputs, outputs_NN)
                Frobenius_norm = calulate_matrix_norm_square(output_grad_PDE, Jacobian_NN) 
                F_2_loss = a * torch.sum(Frobenius_norm)/2
                loss = (L_2_loss +   F_2_loss)
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights
                epoch_train_loss += loss.item()
                epoch_L2_loss    += L_2_loss.item()
                epoch_F2_loss    += F_2_loss.item()
                
            # Compute average training loss
            
            epoch_train_loss /= len(train_loader)
            epoch_L2_loss  /= len(train_loader)
            epoch_F2_loss  /= len(train_loader)
            train_losses.append(epoch_train_loss)
            L_2.append(epoch_L2_loss)
            F_2.append(epoch_F2_loss)
            # -------------------------------
            # Step 5: Evaluate on Validation (Test) Set
            # -------------------------------
            model.eval()  # Set model to evaluation mode
            epoch_val_loss = 0

            #with torch.no_grad():
            for inputs, output_PDE , output_grad_PDE in valid_loader:
                inputs, output_PDE , output_grad_PDE = inputs.float().to(device), output_PDE.float().to(device) , output_grad_PDE.float().to(device)  # Ensure float32
                inputs.requires_grad_(True)
                outputs_NN = model(inputs)
                L_2_loss = custom_loss_without_grad(output_PDE, outputs_NN)/2
                Jacobian_NN = calculate_jacobian_full(inputs, outputs_NN)
                Frobenius_norm = calulate_matrix_norm_square(output_grad_PDE, Jacobian_NN) 
                F_2_loss = a * torch.sum(Frobenius_norm)/2
                loss = (L_2_loss +   F_2_loss)
                epoch_val_loss += loss.item()

            # Compute average validation loss
            epoch_val_loss /= len(test_loader)
            val_losses.append(epoch_val_loss)

            # Print progress every 10 epochs
            if epoch % 10 == 0 :
                print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, L2 Loss: {epoch_L2_loss:.4f}, F2 Loss: {epoch_F2_loss:.4f}")
            elif epoch ==(num_epochs-1):
                print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, L2 Loss: {epoch_L2_loss:.4f}, F2 Loss: {epoch_F2_loss:.4f}")
        
        # -------------------------------
        # Test data: Find accuracies
        # -------------------------------
        model.eval()  # Set model to evaluation mode
    
        for inputs, output_PDE , output_grad_PDE in test_loader:
            inputs, output_PDE , output_grad_PDE = inputs.float().to(device), output_PDE.float().to(device) , output_grad_PDE.float().to(device)  # Ensure float32
            inputs.requires_grad_(True)
            #jacobian_data, outputs_NN = jacobian_cal(inputs, model)
            outputs_NN = model(inputs)
            L_2 = L2_accuaracy(true_Data=output_PDE , nueral_network_output=outputs_NN)
            print("L2 accuracy with Jacobian: ",L_2)
            Jacobian_NN = calculate_jacobian_full(inputs, outputs_NN)
            Frobenius_norm = calulate_matrix_norm_square(output_grad_PDE, Jacobian_NN) 
            true_matrix_norm = torch.norm(output_grad_PDE, p='fro', dim=(1, 2))**2
            H1 = 1 - torch.sqrt(torch.mean((Frobenius_norm)/true_matrix_norm))
            print("H1 accuracy : ", H1)
            
    elif isinstance(network_type, int): 
         
        model = NeuralNet(layer_sizes=layer_sizes)
        model.to(device)
        
    
        optimizer = optim.Adam(model.parameters(), lr=0.002)
        train_losses = []
        val_losses = []
        L_2 = []
        F_2 = []
        a=1e0
        ran_size = int(network_type)
        
        ran_iter = 5 * int(U.shape[0]/ran_size)
        
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode
            epoch_train_loss = 0
            epoch_L2_loss = 0
            epoch_F2_loss = 0

            for inputs, output_PDE , output_grad_PDE in train_loader:
                inputs, output_PDE , output_grad_PDE = inputs.float().to(device), output_PDE.float().to(device) , output_grad_PDE.float().to(device)  # Ensure float32
                optimizer.zero_grad()
                # 🔥 Ensure inputs track gradients before computing Jacobian
                inputs.requires_grad_(True)
                outputs_NN = model(inputs)  # Forward pass
                norm = 0
                V_k_tilde = V
                for i in range(ran_iter):
                    # Draw k numbers uniformly from {1, 2, ..., r}
                    k = torch.randperm(r)[:ran_size] #+ 1  # Add 1 to shift range from [0, r-1] to [1, r]
                    U_k = U[:,k]
                    U_k_T_nabla_q = torch.matmul(U_k.T, output_grad_PDE)
                    grad_U_k_Output_NN = calculate_jacobian(U_k , inputs, outputs_NN)
                    output_final = grad_U_k_Output_NN
                    norm += calulate_matrix_norm_square(U_k_T_nabla_q, output_final) 
                matrix_norm_expectation = (norm * r )/ (k.shape[0] * ran_iter)
                L_2_loss = custom_loss_without_grad(output_PDE, outputs_NN)/2
                F_2_loss = a * torch.sum(matrix_norm_expectation)/2
                loss = (L_2_loss +   F_2_loss)
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights
                epoch_train_loss += loss.item()
                epoch_L2_loss    += L_2_loss.item()
                epoch_F2_loss    += F_2_loss.item()
                
            # Compute average training loss
            
            epoch_train_loss /= len(train_loader)
            epoch_L2_loss  /= len(train_loader)
            epoch_F2_loss  /= len(train_loader)
            train_losses.append(epoch_train_loss)
            L_2.append(epoch_L2_loss)
            F_2.append(epoch_F2_loss)
            # -------------------------------
            # Step 5: Evaluate on Validation (Test) Set
            # -------------------------------
            model.eval()  # Set model to evaluation mode
            epoch_val_loss = 0

            #with torch.no_grad():
            for inputs, output_PDE , output_grad_PDE in valid_loader:
                inputs, output_PDE , output_grad_PDE = inputs.float().to(device), output_PDE.float().to(device) , output_grad_PDE.float().to(device)  # Ensure float32
                inputs.requires_grad_(True)
                outputs_NN = model(inputs)
                norm = 0
                V_k_tilde = V
                for i in range(ran_iter):
                    # Draw k numbers uniformly from {1, 2, ..., r}
                    k = torch.randperm(r)[:ran_size] #+ 1  # Add 1 to shift range from [0, r-1] to [1, r]
                    U_k = U[:,k]
                    U_k_T_nabla_q = torch.matmul(U_k.T, output_grad_PDE)
                    grad_U_k_Output_NN = calculate_jacobian(U_k , inputs, outputs_NN)
                    output_final = grad_U_k_Output_NN
                    norm += calulate_matrix_norm_square(U_k_T_nabla_q, output_final) 
                    
                matrix_norm_expectation = (norm * r )/ (k.shape[0] * ran_iter)
                L_2_loss = custom_loss_without_grad(output_PDE, outputs_NN)/2
                F_2_loss = a * torch.sum(matrix_norm_expectation)/2
                loss = (L_2_loss +  F_2_loss)
                epoch_val_loss += loss.item()
                

            # Compute average validation loss
            epoch_val_loss /= len(test_loader)
            val_losses.append(epoch_val_loss)

            # Print progress every 10 epochs
            if epoch % 10 == 0 :
                print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, L2 Loss: {epoch_L2_loss:.4f}, F2 Loss: {epoch_F2_loss:.4f}")
            elif epoch ==(num_epochs-1):
                print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, L2 Loss: {epoch_L2_loss:.4f}, F2 Loss: {epoch_F2_loss:.4f}")
                
            model.eval()  # Set model to evaluation mode
    
        for inputs, output_PDE , output_grad_PDE in test_loader:
            inputs, output_PDE , output_grad_PDE = inputs.float().to(device), output_PDE.float().to(device) , output_grad_PDE.float().to(device)  # Ensure float32
            inputs.requires_grad_(True)
            #jacobian_data, outputs_NN = jacobian_cal(inputs, model)
            outputs_NN = model(inputs)
            L_2 = L2_accuaracy(true_Data=output_PDE , nueral_network_output=outputs_NN)
            
            print("L2 accuaracy with Jacobian: ",L_2)
            
            norm = 0
            for i in range(ran_iter):
                    
                # Draw k numbers uniformly from {1, 2, ..., r}
                k = torch.randperm(r)[:ran_size] #+ 1  # Add 1 to shift range from [0, r-1] to [1, r]
                
                U_k = U[:,k]
                
                #V_k_tilde = V[:,k_tilde]
                #Sigma_k = U_k.T @ U @ sigma_r @ V_k_tilde.T
                U_k_T_nabla_q = torch.matmul(U_k.T, output_grad_PDE)
                
                grad_U_k_Output_NN = calculate_jacobian(U_k , inputs, outputs_NN)
                #print(Sigma_k.shape, grad_U_k_Output_NN.shape, V_k_tilde.shape)
                #output_final = final_output_batch( grad_U_k_Output_NN , V_k_tilde)
                output_final = grad_U_k_Output_NN
                #inputs = inputs.clone().detach().requires_grad_(True) 
                
                norm += calulate_matrix_norm_square(U_k_T_nabla_q, output_final) 
                    
            matrix_norm_expectation = (norm * r )/ (k.shape[0] * ran_iter)
            #print(matrix_norm_expectation)
            true_matrix_norm = torch.norm(output_grad_PDE, p='fro', dim=(1, 2))**2
            #print(true_matrix_norm)
            H1 = 1- torch.sqrt(torch.mean((matrix_norm_expectation)/true_matrix_norm))
            print("H1 accuracy : ", H1)
            