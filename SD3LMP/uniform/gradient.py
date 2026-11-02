from train_nn import calculate_jacobian_full_fwd
import torch
import torch.nn as nn
import numpy as np
import random
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

import h5py
import os

import pdb

def compute_gradient(neural_network, Decoder_matrix, Encoder_matrix, bias, d, interpolation_matrix, Noise_cov, mean_p, Sigma_p):
    # Compute the gradient of the neural network output with respect to the input data

    seed = 2025   
    random.seed(seed)
    np.random.seed(seed)        
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create a generator for deterministic behavior
    g = torch.Generator()
    g.manual_seed(seed)

    
    current_folder = os.getcwd()
    Sampling_type = current_folder.split("/")[-1]
    print("Sampling_type:", Sampling_type)
    parent_folder = os.path.dirname(current_folder)
    model_name = parent_folder.split("/")[-1]
    print("Model name:", model_name)
    nx=31
    ny=31
    nodes = (nx+1)*(ny+1)
    directory_path = "../../Results/"+str(model_name)+"/"+str(Sampling_type)+"/Nodes_"+str(nodes)

    # Path to your JSON file
    filename = directory_path+ "/POD_data.h5"
    with h5py.File(filename, "r") as final_data:
        k = final_data["Gempy_Inputs"][:]
        u_data = final_data["PDE_outputs"][:]
        J_data = final_data["Jacobian"][:]
        
        
    dtype = torch.float32
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
        
    dtype  = torch.float32
    
    Gempy_Inputs = torch.tensor(k, dtype=dtype, device=device)
    PDE_outputs = torch.tensor(u_data, dtype=dtype, device=device)
    Jacobian = torch.tensor(J_data, dtype=dtype, device=device)
    interpolation_matrix = torch.tensor(interpolation_matrix, dtype=dtype, device=device)
    # Encoder_matrix = torch.tensor(Encoder_matrix, dtype=dtype, device=device)
    Decoder_matrix = torch.tensor(Decoder_matrix, dtype=dtype, device=device)
    bias = torch.tensor(bias, dtype=dtype, device=device)
    data = torch.tensor(d, dtype=dtype, device=device)
    Noise_cov = torch.tensor(Noise_cov, dtype=dtype, device=device)
    Sigma_p = torch.tensor(Sigma_p, dtype=dtype, device=device)
    mean_p = torch.tensor(mean_p, dtype=dtype, device=device)
    
    dataset = TensorDataset(Gempy_Inputs, PDE_outputs, Jacobian)
    N, _, _ = Jacobian.shape

    train_size = int(0.6 * N)
    valid_size = int(0.2 * N)
    test_size = N - train_size - valid_size

    # Randomly split dataset
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size], generator=g)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=g, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle = False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False,num_workers=0)
    
    # Check sizes
    print(f"Train size: {len(train_dataset)},Valid size: {len(valid_dataset)},  Test size: {len(test_dataset)}")
    
    neural_network = neural_network.to(device)
    neural_network.eval()  # Set model to evaluation mode

    for inputs, output_PDE , output_grad_PDE in test_loader:
        inputs.requires_grad_(True)
        outputs_NN = neural_network(inputs)
        Jacobian_NN = calculate_jacobian_full_fwd(inputs, model=neural_network)
        del_K = inputs - mean_p
        
    
    outputs = torch.matmul(outputs_NN.detach(), Decoder_matrix.T) + bias
    Jacobian_output = torch.einsum("ca,nab->ncb",Decoder_matrix, Jacobian_NN.detach())
    
    q_true = torch.matmul(output_PDE, interpolation_matrix.T)
    q_NN = torch.matmul(outputs, interpolation_matrix.T)
    
    r_NN = q_NN - data
    r_true = q_true - data
    
    Jacobian_output_obs_NN = torch.einsum("ca, nab->ncb", interpolation_matrix, Jacobian_output)
    Jacobian_output_obs_true = torch.einsum("ca, nab->ncb", interpolation_matrix, output_grad_PDE)
    
    
    Noise_cov_inv = 1/ (Noise_cov)**2
    
    prior_term = torch.einsum("ab, cb->ca", torch.linalg.inv(Sigma_p), del_K)
    
    
    # grad = J^T Sigma_Noise^-1 r
    #grad_true = torch.einsum("ncb,cd,nd->nb", Jacobian_output_obs_true, Noise_cov_inv, r_true) + prior_term
    #grad_NN = torch.einsum("ncb,cd,nd->nb", Jacobian_output_obs_NN, Noise_cov_inv, r_NN) + prior_term
    
    grad_true = torch.einsum("ncb,nc->nb", Jacobian_output_obs_true, r_true)* Noise_cov_inv + prior_term
    grad_NN = torch.einsum("ncb,nc->nb", Jacobian_output_obs_NN, r_NN) * Noise_cov_inv+ prior_term
    
    # grad_true = torch.einsum("nab,na->nb", Jacobian_output_obs_true, r_true)
    # grad_NN = torch.einsum("nab,na->nb", Jacobian_output_obs_NN, r_NN)
    
    grad_diff = grad_NN - grad_true
    
    grad_diff_norm_squared = torch.sum(grad_diff**2, axis=1)
    grad_true_norm_squared = torch.sum(grad_true**2, axis=1) #+ 1e-8
    
    #obs_data_term = torch.einsum("ncb,cd, d->nb", Jacobian_output_obs_true -Jacobian_output_obs_NN, Noise_cov_inv, data) 
    obs_data_term = torch.einsum("ncb, c->nb", Jacobian_output_obs_true -Jacobian_output_obs_NN, data) * Noise_cov_inv
    #print("error\n", torch.sum(obs_data_term**2, dim=1))
    #print("grad_diff_norm_squared\n",grad_diff_norm_squared)
    #print("grad_true_norm_squared \n",grad_true_norm_squared )
    
    ratio_grad_diff_true = grad_diff_norm_squared / grad_true_norm_squared
    # Flag = ratio_grad_diff_true > 1.0
    # print("ratio_grad_diff_true\n", Flag)
    
    # print("grad_diff_norm_squared\n", grad_diff_norm_squared[1])
    # print("grad_true_norm_squared\n", grad_true_norm_squared[1])
    
    # print("grad_diff[1]:", grad_diff[1])
    # print("grad_true[1]:", grad_true[1])
    # print("grad_NN[1]:", grad_NN[1])
    del_J = Jacobian_output_obs_true[1] - Jacobian_output_obs_NN[1]
    #print("obs_data_term[1]:", obs_data_term[1])
    #exit()
    expectation_ratio = torch.mean(ratio_grad_diff_true)
    
    Gradient_accuracy = 1 - torch.sqrt(expectation_ratio)
    print("Gradient accuracy:", Gradient_accuracy)
    

    
            
            
    

