import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import arviz as az
import pandas as pd
from datetime import datetime
import json
import argparse

import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive, EmpiricalMarginal, RandomWalkKernel
from pyro.infer.autoguide import init_to_mean, init_to_median, init_to_value
from pyro.infer.inspect import get_dependencies
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete
from pyro.infer.mcmc.util import TraceEinsumEvaluator
from torch.autograd import grad

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

import gempy as gp
import gempy_engine
import gempy_viewer as gpv
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_probability.plot_posterior import default_red, default_blue, PlotPosterior

from Initial_model import *
from model import MyModel

dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


import dolfin as dl
import ufl

import sys
import os
sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../") )
from hippylib import *

import logging
import math
import numpy as np

import matplotlib.pyplot as plt

from train_nn import *
#%matplotlib inline

logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

def main():
    
    ###############################################################################
    # Seed the randomness 
    ###############################################################################
    seed = 42           
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Setting the seed for Pyro sampling
    pyro.set_rng_seed(42)
    
    ###############################################################################
    # Create a directory to store plots and results
    ###############################################################################
    directory_path = "./Results"
    if not os.path.exists(directory_path):
        # Create the directory if it does not exist
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' was created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
        
    ################################################################################
    # Create initial model with higher refinement for better resolution and save it
    ###############################################################################
    '''
        Gempy is a package which is based on universal cokriging approach. It is similar to Gaussian 
        process. Thus, it creates a covariance matrix based on input parameters and find the value of
        scalar field at new location. So value of scalar field at new location is dependent on the input
        parameters directly but not on whole grid. Therefore setting refinement higher is good for plotting 
        whereas as it has no effect on find value at an new location. 
    '''
    prior_filename=  directory_path + "/prior_model.png" 
    geo_model_test = create_initial_gempy_model_3_layer(refinement=7,filename=prior_filename, save=True)
    # We can initialize again but with lower refinement because gempy solution are inddependent
    geo_model_test = create_initial_gempy_model_3_layer(refinement=3,filename=prior_filename, save=False)
    
    
    #####################################################################################################
    # Custom grid or rodes of mesh
    #####################################################################################################
    '''
        To find the output at any specific location can be done by creating a custom grid at that location. 
    '''
    
    
    # ---------------- 1️⃣ Create the Mesh ----------------
    nx = 31
    ny = 31
    nz = 7
    degree = 1
    mesh = dl.UnitSquareMesh(nx, ny)
    #mesh = dl.UnitCubeMesh(nx,ny,nz)
    loaded_array = mesh.coordinates()
    print(loaded_array)
    if loaded_array.shape[1]==2:
        xyz_coord = np.insert(loaded_array, 1, 0, axis=1)
    elif loaded_array.shape[1]==3:
        xyz_coord = loaded_array
    print("nodes shape: ",xyz_coord.shape)
    
    
    # Set the custom grid
    gp.set_custom_grid(geo_model_test.grid, xyz_coord=xyz_coord)
    
    geo_model_test.interpolation_options.mesh_extraction = False
    
    ###############################################################################
    # Solve the gempy to compute the model
    ###############################################################################
    sol = gp.compute_model(geo_model_test)
    
    ###############################################################################
    # Check the coordinates of the input parameters
    ###############################################################################
    sp_coords_copy_test = geo_model_test.interpolation_input.surface_points.sp_coords.copy()
    geo_model_test.transform.apply_inverse(sp_coords_copy_test) 
    
    '''
        Scalar field obtained by gempy is transformed so that it lies between 1 to total_number of layer.
        Directly assigning the discrete values will make the output discontinous. Therefore at the bounday
        between two layers, we like to join these transformed values using sigmoid. Therefore it regquires a 
        sigmoid slope.
    '''
    geo_model_test.interpolation_options.sigmoid_slope = 200 
    gp.compute_model(geo_model_test)
    
    sp_coords_copy_test = geo_model_test.interpolation_input.surface_points.sp_coords.copy()
    #print(sp_coords_copy_test)
    
    ###############################################################################
    # Output at custom grid
    ###############################################################################
    '''
        The output at custom grid values is continous and if rounded will give us the layer index. 
        We can put a function on the top of this to provide some properties to each layer. 
        
    '''
    custom_grid_values_prior = torch.tensor(geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values)
    print(custom_grid_values_prior.shape)
    
    ################################################################################
    # Store the Initial Interface data and orientation data
    ################################################################################
    df_sp_init = geo_model_test.surface_points.df
    df_or_init = geo_model_test.orientations.df
    
    filename_initial_sp = directory_path + "/Initial_sp.csv"
    filename_initial_op = directory_path + "/Initial_op.csv"
    df_sp_init.to_csv(filename_initial_sp)
    df_or_init.to_csv(filename_initial_op)
    
    ###############################################################################
    # Change the backend to PyTorch for probabilistic modeling
    ###############################################################################
    BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH)
    
    
    geo_model_test.interpolation_options.sigmoid_slope = 200
    
    ###############################################################################
    # Make a list of gempy parameter which would be treated as a random variable
    ###############################################################################
    test_list=[]
    test_list.append({"update":"interface_data","id":torch.tensor([1]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[1,2],dtype=dtype), "std":torch.tensor(0.06,dtype=dtype)}})
    test_list.append({"update":"interface_data","id":torch.tensor([4]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[4,2],dtype=dtype), "std":torch.tensor(0.06,dtype=dtype)}})
    
    num_layers = len(test_list) # length of the list

    ###
    # Example to show how to update the gempy paramter and show it's gradient
    ###
    
    mu_1 = torch.tensor(sp_coords_copy_test[1, 2], dtype=dtype, requires_grad=True)
    mu_2 = torch.tensor(sp_coords_copy_test[4, 2], dtype=dtype, requires_grad=True)
    list_paramter = [mu_1, mu_2]
    
    interpolation_input = geo_model_test.interpolation_input
    print(type(interpolation_input))
    
    interpolation_input.surface_points.sp_coords = torch.index_put(
                            interpolation_input.surface_points.sp_coords,
                            (torch.tensor([1]), torch.tensor([2])),
                            mu_1)
    interpolation_input.surface_points.sp_coords = torch.index_put(
                            interpolation_input.surface_points.sp_coords,
                            (torch.tensor([4]), torch.tensor([2])),
                            mu_2)
    # # Compute the geological model
    geo_model_test.solutions = gempy_engine.compute_model(
                interpolation_input=interpolation_input,
                options=geo_model_test.interpolation_options,
                data_descriptor=geo_model_test.input_data_descriptor,
                geophysics_input=geo_model_test.geophysics_input,
            )
            
    # Compute and observe the thickness of the geological layer
    custom_grid_values = geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values
    #print(custom_grid_values)
    
    gradient_argument = torch.ones_like(custom_grid_values)
    # Compute gradient
    #custom_grid_values.backward(gradient_argument)
    grad_K_k =torch.zeros((2, custom_grid_values.shape[0]),dtype=dtype)
    for i in range(len(list_paramter)):
        for j in range(custom_grid_values.shape[0]):
            y_x = grad(custom_grid_values[j], list_paramter[i],  create_graph=True)
            grad_K_k[i,j] = y_x[0]
    # Get the gradient
    
    
    #print(custom_grid_values)

    
    
    model = MyModel()
    
    #torch.multiprocessing.set_start_method("spawn", force=True)
    #torch.multiprocessing.set_sharing_strategy("file_system")
    
    filename_Bayesian_graph =directory_path +"/Bayesian_graph.png"
    
    pyro.clear_param_store()
    
    
    # We can build a probabilistic model using pyro by calling it 
    #dot = pyro.render_model(model.model_test, model_args=(test_list,geo_model_test,num_layers,mesh,degree, dtype),render_distributions=True,filename=filename_Bayesian_graph)
    dot = pyro.render_model(model.model_test, model_args=(test_list,geo_model_test,num_layers, mesh, degree, dtype))
    # Generate 50 samples
    num_samples = 400 # N
    predictive = Predictive(model.model_test, num_samples=num_samples)
    samples = predictive(test_list,geo_model_test,num_layers, mesh, degree,  dtype)
    
    ######store the samples ######
    u = samples["u"] # size of samples X number of grid points , (N X M)
    mean_u = torch.mean(u,dim=0)
    u = u - mean_u
    N, q = u.shape # N= Number of samples, q = length of u 
    
    result_sum = torch.zeros(q,q, dtype=dtype)  # Initialize a tensor to accumulate the sum
    for i in range(N):
        u_data = u[i].reshape(-1,1)# Reshape u to [1024, 1]
        result_sum += u_data @ u_data.T  # Matrix multiplication to get [1024, 1024]
    cov_matrix_u = result_sum / N
    eigenvalues_u, eigenvectors_u = torch.linalg.eig(cov_matrix_u)
    
    cuttoff = 1e-6
    mask_u = eigenvalues_u.real > cuttoff
    eig_u = eigenvalues_u[mask_u].real
    eig_vec_u = eigenvectors_u[:,mask_u].real
    r = eig_vec_u.shape[1] # dimension after reduction
    
    # conductivity/ porosity at each node
    K = samples["K"] # size of samples X number of grid points , (N X dU) 
    # gradient of u with k , ∂u/∂K
    grad_K_u = samples["grad_K_u"] # (N X dU x dU)
    
    print(grad_K_u.shape, grad_K_k.shape)
    grad_k_u = grad_K_u @ grad_K_k.T  # ∂u/∂k = ∂u/∂K * ∂K/∂k,  size of samples X number of paramters X number of grid points(N, p, M) 
    print(grad_k_u.shape) # size of samples X number of grid points X numper of input paramter N x dU x m
    
    #-----------------------------------------------------------------------------------------#
    # calculate the Expectation for the value and grad
    # Expected value of uuᵀ:
    # E[uuᵀ] = (1/N) ∑_{i=1}^{N} uuᵀ
    # E[uuᵀ] = (1/N) ∑_{i=1}^{N} uuᵀ
    #cov_matrix_u = (u.T @ u) / num_samples  # 
    
    #cov_matrix_grad_m_u = (grad_m_u.T @ grad_m_u) / num_samples  # Shape (N, N)
    N,q,m  = grad_k_u.shape # size of samples X number of grid points X numper of input paramter N x dU x m
    result_grad_sum_left = torch.zeros(r,r, dtype=dtype)  # Initialize a tensor to accumulate the sum
    result_grad_sum_right = torch.zeros(m,m, dtype=dtype)
    
    u_grad_list =[]
    for i in range(N):
        print(eig_vec_u.shape,grad_k_u[i].shape)
        grad_u = eig_vec_u.T @ grad_k_u[i] # phi.T \nabla u
        print(grad_u.shape)
        result_grad_sum_left += grad_u @ grad_u.T
        print(result_grad_sum_left.shape)
        result_grad_sum_right += grad_u.T @ grad_u
        print(result_grad_sum_right.shape)
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
    
    U = U.real
    V = V.real
    cuttoff_sing = 1e-6
    
    if r > m:
        # reduced dimension is greater than input dimension
        signular_value = torch.sqrt(eigenvalues_grad_u_right.real)
        # Replace values smaller than cutoff with 0
        sigular = torch.where(signular_value > cuttoff_sing, signular_value, torch.tensor(0.0))
        sigma_r = torch.diag(sigular)
        dummy = torch.zeros((r-m,m))
        sigma_r = torch.cat((sigma_r, dummy), dim=0)
    else:
        # reduced dimension is greater than input dimension
        signular_value = torch.sqrt(eigenvalues_grad_u_left.real)
        # Replace values smaller than cutoff with 0
        sigular = torch.where(signular_value > cuttoff_sing, signular_value, torch.tensor(0.0))
        sigma_r = torch.diag(sigular)
        dummy = torch.zeros((m-r,r))
        sigma_r = torch.cat((sigma_r, dummy), dim=1)
        
    # First subplot
    
    
    plt.figure(figsize=(15,5))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(eigenvalues_u)), torch.log10(eigenvalues_u.real), marker='o', linestyle='-', color='b')
    plt.xlabel("Index")
    plt.ylabel(r"$\log_{10}(\lambda)$")
    plt.title(r"Eigenvalues of $\mathbb{E}[uu^T]$")
    
    # Second subplot
    plt.subplot(1, 2, 2)
    plt.plot(range(len(eigenvalues_grad_u_left)), torch.log10(eigenvalues_grad_u_left.real), marker='o', linestyle='-', color='r')
    plt.xlabel("Index")
    plt.ylabel(r"$\log_{10}(\lambda)$")
    plt.title(r"Eigenvalues of $\mathbb{E}[\nabla_k u (\nabla_k u)^T]$")
    plt.savefig("Eigenvalues.png")
    plt.close()
    
    
    
    
    paramter = torch.stack((samples["mu_1"], samples["mu_2"]), dim=1) # (N, p) = number of sample X number of paramter
    
    print(eig_vec_u.shape, u.shape)
    # Map the u and grad_u into lower dimension best on the cuttoff value generated eigen vectors
    u_output = (eig_vec_u.T @ u.T).T # (eig_vec_u.T @ u) , the last Transformation is because vector is arranged row wise in data
    
    
    # u_grad_list =[]
    # for i in range(N):
    #     reduced_dim = eig_vec_grad_u.T @ grad_k_u[i].T
    #     u_grad_list.append(reduced_dim)
    
    # u_grad  = torch.stack(u_grad_list, dim=0)
    
    ######################################
    # Data 
    ########################################
    # Create dataset
    # Check for MPS availability and set the device
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    import gc # garbage collection
    torch.mps.empty_cache()
    
    dataset = TensorDataset(paramter.float().to(device), u_output.float().to(device), u_grad.float().to(device))
    U = U.float().to(device)
    V = V.float().to(device)
    sigma_r = sigma_r.float().to(device)
    
    # Split sizes (80% train, 20% test)
    N = paramter.shape[0]
    train_size = int(0.8 * N)
    test_size = N - train_size

    # Randomly split dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Check sizes
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    
    # Instantiate model
    model = NeuralNet(input_dim=paramter.shape[1], hidden_dim=32, output_dim=u_output.shape[1])
    model.to(device)
    
    #define loss function
    def custom_loss_without_grad(output_PDE, outputs_NN):
        """
        
        Computes the loss: sum over N samples of
        ||output_PDE_i - output_NN_i||^2_2 
        
        """
        N, D = outputs_NN.shape
        l2_norm = torch.norm(output_PDE - outputs_NN, p=2, dim=1) ** 2  # Squared L2 norm for each sample
        
        total_loss = torch.sum(l2_norm) /(N*D)  # Sum over all N samples
        return total_loss
    
    def calulate_matrix_norm_square(Sigma_k, output_final):
        """
        Computes the loss: sum over N samples of
        ||Sigma_k - output_final||^2_2 
        
        """
        A = output_final - Sigma_k.unsqueeze(0)
        frob_norm = (torch.norm(A, p='fro', dim=(1, 2)) ** 2)/A.shape[0]
        return frob_norm
    
    def calculate_jacobian3(U_k, inputs, outputs_NN):
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
    
    
    # -------------------------------
    # Step 3: Define Loss and Optimizer
    # -------------------------------
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=0.01)

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
            for inputs, output_PDE , output_grad_PDE in test_loader:
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
    plt.savefig("Los_L2.png")
    plt.close()
        
    # -------------------------------
    # Step 7: Save the model
    # -------------------------------
    torch.save(model.state_dict(), "model_weights.pth")
    ################################################################################
    # Find the result just based on L2 norm of output of NN and and reduced basis
    # and respespect to Derivative
    # || q(k) - f_\theta(k)||_2^2 + || D(q(k)) - D(f_\theta(k))||_2^2
    ################################################################################
    # # Instantiate model
    # model = NeuralNet(input_dim=paramter.shape[1], hidden_dim=32, output_dim=u_output.shape[1])
    # model.to(device)
    
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        epoch_train_loss = 0

        for inputs, output_PDE , output_grad_PDE in train_loader:
            inputs, output_PDE , output_grad_PDE = inputs.float().to(device), output_PDE.float().to(device) , output_grad_PDE.float().to(device)  # Ensure float32
            optimizer.zero_grad()
            # 🔥 Ensure inputs track gradients before computing Jacobian
            #inputs.requires_grad_(True)
            inputs.requires_grad_(True)
            outputs_NN = model(inputs)  # Forward pass
            
            norm = 0
            V_k_tilde = V
            for i in range(100):
                
                # Draw k numbers uniformly from {1, 2, ..., r}
                k = torch.randperm(r)[:10] + 1  # Add 1 to shift range from [0, r-1] to [1, r]
                U_k = U[:,k]
                #V_k_tilde = V[:,k_tilde]
                Sigma_k = U_k.T @ sigma_r @ V_k_tilde
                grad_U_k_Output_NN = calculate_jacobian3(U_k , inputs, outputs_NN)
                #print(Sigma_k.shape, grad_U_k_Output_NN.shape, V_k_tilde.shape)
                output_final = final_output_batch( grad_U_k_Output_NN , V_k_tilde)
                #inputs = inputs.clone().detach().requires_grad_(True) 
                #print(output_final.shape)
                
                norm += calulate_matrix_norm_square(Sigma_k, output_final) 
                
            matrix_norm_expectation = (norm/ 100)* (r/ k.shape[0])
            
            #jacobian_data = compute_jacobian_(inputs, model)
            #jacobian_data, outputs_NN = jacobian_cal(inputs, model)
            loss = ( custom_loss_without_grad(output_PDE, outputs_NN)  +  torch.sum(matrix_norm_expectation))/2
            #loss = criterion(output_PDE, outputs_NN)
            #loss = custom_loss(output_PDE=output_PDE, outputs_NN= outputs_NN, grad_PDE=output_grad_PDE, grad_NN= jacobian_data)
            #loss.backward(retain_graph=True)  # Backpropagation
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            epoch_train_loss += loss.item()
            # del loss, outputs_NN, jacobian_data
            # gc.collect()
            # torch.mps.empty_cache()
            # print(epoch_train_loss)
        # Compute average training loss
        
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # -------------------------------
        # Step 5: Evaluate on Validation (Test) Set
        # -------------------------------
        model.eval()  # Set model to evaluation mode
        epoch_val_loss = 0

        #with torch.no_grad():
        for inputs, output_PDE , output_grad_PDE in test_loader:
            inputs, output_PDE , output_grad_PDE = inputs.float().to(device), output_PDE.float().to(device) , output_grad_PDE.float().to(device)  # Ensure float32
            inputs.requires_grad_(True)
            #jacobian_data, outputs_NN = jacobian_cal(inputs, model)
            outputs_NN = model(inputs)
            
            norm = 0
            V_k_tilde = V
            for i in range(100):
            
                # Draw k numbers uniformly from {1, 2, ..., r}
                k = torch.randperm(r)[:10] + 1  # Add 1 to shift range from [0, r-1] to [1, r]
                U_k = U[:,k]
                #V_k_tilde = V[:,k_tilde]
                Sigma_k = U_k.T @ sigma_r @ V_k_tilde
                grad_U_k_Output_NN = calculate_jacobian3(U_k , inputs, outputs_NN)
                #print(Sigma_k.shape, grad_U_k_Output_NN.shape, V_k_tilde.shape)
                output_final = final_output_batch( grad_U_k_Output_NN , V_k_tilde)
                #inputs = inputs.clone().detach().requires_grad_(True) 
                #print(output_final.shape)
                
                norm += calulate_matrix_norm_square(Sigma_k, output_final) 
            
            matrix_norm_expectation = (norm/ 100)* (r/ k.shape[0])
            
            # jacobian_data = compute_jacobian_(inputs, model)
            #loss = custom_loss_without_grad(output_PDE, outputs_NN)
            loss = ( custom_loss_without_grad(output_PDE, outputs_NN)  +  torch.sum(matrix_norm_expectation))/2
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
    plt.savefig("Loss.png")
    plt.close()

   # -------------------------------
    # Step 7: Save the model
    # -------------------------------
    torch.save(model.state_dict(), "model_weights_with_Jacobian.pth")
        
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