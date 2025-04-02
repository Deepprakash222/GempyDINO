import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import arviz as az
import pandas as pd
from datetime import datetime
from train_nn import *
# from accuraices import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

dtype = torch.float64

#define loss function

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


def final_output_batch( grad_U_k_Output_NN , V_k_tilde):
    
    Batch_size,_, _ = grad_U_k_Output_NN.shape
    output_list=[]
    for i in range(Batch_size):
        output_final = grad_U_k_Output_NN[i] @ V_k_tilde
        output_list.append(output_final)
    return torch.stack(output_list, dim=0)
    

def train_nn_with_different_nodes(nodes, cuttoff=None):
    
    seed = 42           
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
   
    directory_path = "./Results"
    Nodes = nodes
    directory_path = directory_path+"/Nodes_"+str(Nodes)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Load the state variable
    u = torch.tensor(np.load(directory_path+ "/u.npy"), device=device)
    
    # Find the average of state variable
    mean_u = torch.mean(u,dim=0)
    # create a zero mean state variable
    u = u - mean_u
    N, q = u.shape # N= Number of samples, q = length of u
    
    # Load the Jacobian data
    J = torch.tensor(np.load(directory_path+ "/Jacobian.npy"), device=device)
    print(u.shape, J.shape, N, q )
    
    ######################################################################################
    # Caluculate the eigen values and eigen vecotor of E[qq^T]
    ######################################################################################
    result_sum = torch.zeros(q,q, dtype=dtype, device=device)  # Initialize a tensor to accumulate the sum
    for i in range(N):
        u_data = u[i].reshape(-1,1)# Reshape u to [1024, 1]
        result_sum += u_data @ u_data.T  # Matrix multiplication to get [1024, 1024]
    cov_matrix_u = result_sum / N
    #print("I have reached step 1: ", cov_matrix_u.shape)
    
    eigenvalues_u, eigenvectors_u = torch.linalg.eig(cov_matrix_u)
    eigenvalues_u = eigenvalues_u.real
    #print("Eigen_Values computed successfully")
    ######################################################################################
    # Find the eigen vectors based on some cutoff for eigen-values
    ######################################################################################
    if cuttoff ==None:
        cuttoff = 1e-6
        mask_u = eigenvalues_u >= cuttoff
        eig_u = eigenvalues_u[mask_u]
        eig_vec_u = eigenvectors_u[:,mask_u].real
        np.save(directory_path+ "/Truncated_Eigen_Vector_Matrix.npy", eig_vec_u.cpu().detach().numpy())
    else:
        eig_u = eigenvalues_u[:cuttoff]
        eig_vec_u = eigenvectors_u[:,:cuttoff].real
        np.save(directory_path+ "/Truncated_Eigen_Vector_Matrix.npy", eig_vec_u.cpu().detach().numpy())
        
    r = eig_vec_u.shape[1] # dimension after reduction
    
    ######################################################################################
    # Caluculate the eigen values and eigen vecotor of E[\nabla(q) \nabla(q)^T] and E[\nabla(q)^T \nabla(q)]
    ######################################################################################
     #cov_matrix_grad_m_u = (grad_m_u.T @ grad_m_u) / num_samples  # Shape (N, N)
    N,q,m  = J.shape # size of samples X number of grid points X numper of input paramter N x dU x m
    result_grad_sum_left = torch.zeros(r,r, dtype=dtype, device=device)  # Initialize a tensor to accumulate the sum
    result_grad_sum_right = torch.zeros(m,m, dtype=dtype, device=device)
    
    u_grad_list =[]
    for i in range(N):
        #print(eig_vec_u.shape,grad_k_u[i].shape)
        grad_u = eig_vec_u.T @ J[i] # phi.T \nabla u
        #print(grad_u.shape)
        result_grad_sum_left += grad_u @ grad_u.T
        #print(result_grad_sum_left.shape)
        result_grad_sum_right += grad_u.T @ grad_u
        #print(result_grad_sum_right.shape)
        u_grad_list.append(grad_u)
    u_grad  = torch.stack(u_grad_list, dim=0)
    # Compute the average
    # print(u_grad_list)
    cov_matrix_grad_m_u_left = result_grad_sum_left / N
    cov_matrix_grad_m_u_right = result_grad_sum_right / N
    
    print(cov_matrix_u.shape, cov_matrix_grad_m_u_left.shape)
    # find the eigen_values of u 
    eigenvalues_grad_u_left, U = torch.linalg.eig(cov_matrix_grad_m_u_left.detach())
    eigenvalues_grad_u_right, V = torch.linalg.eig(cov_matrix_grad_m_u_right.detach())
    
    
    plt.figure(figsize=(15,5))
    plt.subplot(1, 3, 1)
    plt.plot(torch.arange(0, len(eigenvalues_u.cpu())), torch.log10(eigenvalues_u.cpu() + 1e-20), marker='o', linestyle='-', color='b')
    plt.xlabel("Index")
    plt.ylabel(r"$\log_{10}(\lambda)$")
    plt.title(r"Eigenvalues of $\mathbb{E}[uu^T]$")
    
    # Second subplot
    plt.subplot(1, 3, 2)
    plt.plot(torch.arange(0,len(eigenvalues_grad_u_left.real.cpu())), torch.log10(eigenvalues_grad_u_left.real.cpu()), marker='o', linestyle='-', color='r')
    plt.xlabel("Index")
    plt.ylabel(r"$\log_{10}(\lambda)$")
    plt.title(r"Eigenvalues of $\mathbb{E}[\nabla_k u (\nabla_k u)^T]$")
    
    # Second subplot
    plt.subplot(1, 3, 3)
    plt.plot(torch.arange(0, len(eigenvalues_grad_u_right.real.cpu())), torch.log10(eigenvalues_grad_u_right.real.cpu()), marker='o', linestyle='-', color='r')
    plt.xlabel("Index")
    plt.ylabel(r"$\log_{10}(\lambda)$")
    plt.title(r"Eigenvalues of $\mathbb{E}[ (\nabla_k u)^T \nabla_k u]$")
    plt.savefig(directory_path+ "/Eigenvalues.png")
    plt.close()
    
    U = U.real
    V = V.real
    cuttoff_sing = 1e-8
    
    if r > m:
        # reduced dimension is greater than input dimension
        signular_value = torch.sqrt(eigenvalues_grad_u_right.real)
        # Replace values smaller than cutoff with 0
        sigular = torch.where(signular_value > cuttoff_sing, signular_value, torch.tensor(0.0))
        sigma_r = torch.diag(sigular)
        dummy = torch.zeros((r-m,m),device=device)
        sigma_r = torch.cat((sigma_r, dummy), dim=0)
    elif r==m:
        # reduced dimension is greater than input dimension
        signular_value = torch.sqrt(eigenvalues_grad_u_right.real)
        # Replace values smaller than cutoff with 0
        sigular = torch.where(signular_value > cuttoff_sing, signular_value, torch.tensor(0.0))
        sigma_r = torch.diag(sigular)
        
    else:
        # reduced dimension is greater than input dimension
        signular_value = torch.sqrt(eigenvalues_grad_u_left.real)
        # Replace values smaller than cutoff with 0
        sigular = torch.where(signular_value > cuttoff_sing, signular_value, torch.tensor(0.0))
        sigma_r = torch.diag(sigular)
        dummy = torch.zeros((m-r,r),device=device)
        sigma_r = torch.cat((sigma_r, dummy), dim=1)
        
        
    
    
    ######################################################################################
    # Load the input paramters
    ######################################################################################
    inputs = torch.tensor(np.load(directory_path+ "/input.npy"), device=device)
    print(inputs.shape)
    
    ######################################################################################
    # Find the reduced basis state variable
    ######################################################################################
    u_output = (eig_vec_u.T @ u.T).T 
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    
    ######################################################################################
    # Create the Dataset for Neural network
    ######################################################################################
    dataset = TensorDataset(inputs.float().to(device), u_output.float().to(device), u_grad.float().to(device))
    
    U = U.float().to(device)
    V = V.float().to(device)
    sigma_r = sigma_r.float().to(device)
    
    # Split sizes (80% train, 20% test)
    N = inputs.shape[0]
    train_size = int(0.6 * N)
    valid_size = int(0.2 * N)
    test_size = int(0.2 * N)

    # Randomly split dataset
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

   
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size=int(0.2 * N), shuffle=False)
    
    # Check sizes
    print(f"Train size: {len(train_dataset)},Valid size: {len(valid_dataset)},  Test size: {len(test_dataset)}")
   
    # Instantiate model
    m = inputs.shape[1]
    q = u_output.shape[1]
    
    print("m: ", m, " q: ", q)
    
    model = NeuralNet(layer_sizes=[m,q,q,q,q])
    model.to(device)
    
    
    
    
    # -------------------------------
    # Step 3: Define Loss and Optimizer
    # -------------------------------
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    # -------------------------------
    # Step 4: Training Loop with Loss Tracking
    # -------------------------------
    num_epochs = 101
    train_losses = []
    val_losses = []

    ################################################################################
    # Find the result just based on L2 norm of output of NN and and reduced basis
    # || q(k) - f_\theta(k)||_2^2
    ################################################################################
    print(U.shape, sigma_r.shape, V.shape)
    
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
        # Step 5: Evaluate on Validation (Test) Set
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

    
    # -------------------------------
    # Step 6: Plot Training & Validation Loss
    # -------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(range(num_epochs), train_losses, label="Training Loss", color="blue")
    plt.plot(range(num_epochs), val_losses, label="Validation Loss", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(directory_path+ "/Loss_L2.png")
    plt.close()
        
    # -------------------------------
    # Step 7: Save the model
    # -------------------------------
    torch.save(model, directory_path+ "/model_weights.pth")
    
    # -------------------------------
    # Step 8: Find accuracies
    # -------------------------------
    
    model.eval()  # Set model to evaluation mode
    
    for inputs, output_PDE , output_grad_PDE in test_loader:
        inputs, output_PDE , output_grad_PDE = inputs.float().to(device), output_PDE.float().to(device) , output_grad_PDE.float().to(device)  # Ensure float32
        inputs.requires_grad_(True)
        #jacobian_data, outputs_NN = jacobian_cal(inputs, model)
        outputs_NN = model(inputs)
        L_2_1 = L2_accuaracy(true_Data=output_PDE , nueral_network_output=outputs_NN)
        
        print("L2 loss without Jacobian: ",L_2_1)
    ################################################################################
    # Find the result just based on L2 norm of output of NN and and reduced basis
    # and respespect to Derivative
    # || q(k) - f_\theta(k)||_2^2 + || D(q(k)) - D(f_\theta(k))||_2^2
    ################################################################################
    # Instantiate model
    model = NeuralNet(layer_sizes=[m,q,q,q,q])
    model.to(device)
    
    num_epochs = 101
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    train_losses = []
    val_losses = []
    L_2 = []
    F_2 = []
    a=1e0
    ran_iter = 10
    ran_size = int(r/2)
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        epoch_train_loss = 0
        epoch_L2_loss = 0
        epoch_F2_loss = 0

        for inputs, output_PDE , output_grad_PDE in train_loader:
            inputs, output_PDE , output_grad_PDE = inputs.float().to(device), output_PDE.float().to(device) , output_grad_PDE.float().to(device)  # Ensure float32
            optimizer.zero_grad()
            # 🔥 Ensure inputs track gradients before computing Jacobian
            #inputs.requires_grad_(True)
            inputs.requires_grad_(True)
            outputs_NN = model(inputs)  # Forward pass
            
            norm = 0
            V_k_tilde = V
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
            
            #jacobian_data = compute_jacobian_(inputs, model)
            #jacobian_data, outputs_NN = jacobian_cal(inputs, model)
            L_2_loss = custom_loss_without_grad(output_PDE, outputs_NN)/2
            
            F_2_loss = a * torch.sum(matrix_norm_expectation)/2
            
            loss = (L_2_loss +   F_2_loss)
            #loss = criterion(output_PDE, outputs_NN)
            #loss = custom_loss(output_PDE=output_PDE, outputs_NN= outputs_NN, grad_PDE=output_grad_PDE, grad_NN= jacobian_data)
            #loss.backward(retain_graph=True)  # Backpropagation
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            #scheduler.step() # Update learning rate
            epoch_train_loss += loss.item()
            epoch_L2_loss    += L_2_loss.item()
            epoch_F2_loss    += F_2_loss.item()
            # del loss, outputs_NN, jacobian_data
            # gc.collect()
            # torch.mps.empty_cache()
            # print(epoch_train_loss)
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
            #jacobian_data, outputs_NN = jacobian_cal(inputs, model)
            outputs_NN = model(inputs)
            
            norm = 0
            V_k_tilde = V
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
            
            # jacobian_data = compute_jacobian_(inputs, model)
            #loss = custom_loss_without_grad(output_PDE, outputs_NN)
            L_2_loss = custom_loss_without_grad(output_PDE, outputs_NN)/2
            F_2_loss = a * torch.sum(matrix_norm_expectation)/2
            loss = (L_2_loss +  F_2_loss)
            #loss = criterion(output_PDE, outputs_NN)
            #loss = custom_loss(output_PDE=output_PDE, outputs_NN= outputs_NN, grad_PDE=output_grad_PDE, grad_NN= jacobian_data)
            
            # inputs.requires_grad_(True)
            # outputs_NN = model(inputs)  # Forward pass
        
           
            
            # #jacobian_data = compute_jacobian_(inputs, model)
            # #jacobian_data, outputs_NN = jacobian_cal(inputs, model)
            # loss = ( custom_loss_without_grad(output_PDE, outputs_NN) + 1e-5 * torch.sum(matrix_norm_expectation))/2
            epoch_val_loss += loss.item()
            # del loss, outputs_NN, jacobian_data
            # gc.collect()
            # torch.mps.empty_cache()

        # Compute average validation loss
        epoch_val_loss /= len(test_loader)
        val_losses.append(epoch_val_loss)

        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, L2 Loss: {epoch_L2_loss:.4f}, F2 Loss: {epoch_F2_loss:.4f}")

    # -------------------------------
    # Step 6: Plot Training & Validation Loss
    # -------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(range(num_epochs), L_2, label="L2 Loss", color="green")
    plt.plot(range(num_epochs), F_2, label="F2 Loss", color="yellow")
    plt.plot(range(num_epochs), train_losses, label="Training Loss", color="blue")
    plt.plot(range(num_epochs), val_losses, label="Validation Loss", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(directory_path+ "/Loss.png")
    plt.close()

    # -------------------------------
    # Step 7: Save the model
    # -------------------------------
    torch.save(model, directory_path+ "/model_weights_with_Jacobian.pth")
    
    # -------------------------------
    # Step 8: Find accuracies
    # -------------------------------
    
    model.eval()  # Set model to evaluation mode
    
    for inputs, output_PDE , output_grad_PDE in test_loader:
        inputs, output_PDE , output_grad_PDE = inputs.float().to(device), output_PDE.float().to(device) , output_grad_PDE.float().to(device)  # Ensure float32
        inputs.requires_grad_(True)
        #jacobian_data, outputs_NN = jacobian_cal(inputs, model)
        outputs_NN = model(inputs)
        L_2 = L2_accuaracy(true_Data=output_PDE , nueral_network_output=outputs_NN)
        
        print("L2 loss with Jacobian: ",L_2)
        
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
        true_matrix_norm = torch.norm(output_grad_PDE, p='fro', dim=(1, 2))**2
        
        H1 = 1- torch.sqrt(torch.mean((matrix_norm_expectation)/true_matrix_norm))
        print("H1 Loss : ", H1)
        
    return L_2_1, L_2, H1
            
def main():
    
    Nodes = [256, 1024, 4096]
    L2_data =[]
    L2_Jabobian_data = []
    H1_data =[]
    for nodes in Nodes:
        L2_full, L2_Jacobain, H1 = train_nn_with_different_nodes(nodes,cuttoff=100)
        L2_data.append(L2_full.cpu().detach().numpy())
        L2_Jabobian_data.append(L2_Jacobain.cpu().detach().numpy())
        H1_data.append(H1.cpu().detach().numpy())
    
    plt.figure(figsize=(8, 5))
    plt.plot(torch.log2(torch.tensor(Nodes)), L2_data, label="L2 accuracies", color="green")
    plt.plot(torch.log2(torch.tensor(Nodes)), L2_Jabobian_data, label="L2 Jacobian accuracie", color="blue")
    plt.plot(torch.log2(torch.tensor(Nodes)), H1_data, label="F2 accuracie", color="red")
    
    plt.xlabel("log2(Nodes)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs nodes")
    plt.legend()
    plt.grid()
    plt.savefig( "./Accuracies_nodes.png")
    plt.close()
    
if __name__ == "__main__":
    
    # Your main script code starts here
    print("Script started...")
    
    # Record the start time
    start_time = datetime.now()

    main()
    # Record the end time
    end_time = datetime.now()

    # Your main script code ends here
    print("Script ended...")
    
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    
    print(f"Elapsed time: {elapsed_time}")