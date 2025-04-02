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
from torch.func import jacrev

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
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

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

from mpi4py import MPI


def check_gempy_jacobian(epsilon, geo_model_test,mu_1, mu_2, custom_grid_values, J):
    sum=0
    #print("mu_1 :", mu_1 , "mu_2 :", mu_2)
    for i in range(2):
        
        if i==0:
            mu_1_ =mu_1 + epsilon
            mu_2_ = mu_2
        elif i==1:
            mu_1_=mu_1
            mu_2_ =mu_2 + epsilon
        #print("mu_1_ :", mu_1_ , "mu_2_ :", mu_2_)
        
        interpolation_input = geo_model_test.interpolation_input
        
        
        interpolation_input.surface_points.sp_coords = torch.index_put(
                                interpolation_input.surface_points.sp_coords,
                                (torch.tensor([1]), torch.tensor([2])),
                                mu_1_)
        interpolation_input.surface_points.sp_coords = torch.index_put(
                                interpolation_input.surface_points.sp_coords,
                                (torch.tensor([4]), torch.tensor([2])),
                                mu_2_)
        # # Compute the geological model
        geo_model_test.solutions = gempy_engine.compute_model(
                    interpolation_input=interpolation_input,
                    options=geo_model_test.interpolation_options,
                    data_descriptor=geo_model_test.input_data_descriptor,
                    geophysics_input=geo_model_test.geophysics_input,
                )
                
        # Compute and observe the thickness of the geological layer
        custom_grid_values_current = geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values
        #####
        # Check the gradient
        #####
        error_i = - (custom_grid_values - custom_grid_values_current)/epsilon
        #print(J[i,:], error_i)
        sum = sum + torch.linalg.norm(J[i,:] - error_i)
        
    return sum
        
def main():
    
    # Get the MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
        
    comm.Barrier()
    print(f"Process {rank} of {size}")
    
    
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
    nx = 127
    ny = 127
    nz = 7
    degree = 1
    mesh = dl.UnitSquareMesh(comm, nx, ny)
    #mesh = dl.UnitCubeMesh(nx,ny,nz)
    loaded_array = mesh.coordinates()
    #print(loaded_array)
    if loaded_array.shape[1]==2:
        xyz_coord = np.insert(loaded_array, 1, 0, axis=1)
    elif loaded_array.shape[1]==3:
        xyz_coord = loaded_array
    
    comm.Barrier()
    print("nodes shape: ",xyz_coord.shape)
    # Gather mesh coordinates on rank 0
    all_coords = comm.gather(loaded_array, root=0)
    # Ensure all ranks have sent their mesh before proceeding
    comm.Barrier()
    
    
    
    if rank == 0:
        nodes_coord_combined = np.vstack(all_coords)
        unique_nodes = np.unique(nodes_coord_combined, axis=0)  # Remove ghost nodes
        directory_path = directory_path +"/Nodes_"+ str(unique_nodes.shape[0])
        if not os.path.exists(directory_path):
            # Create the directory if it does not exist
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' was created.")
        else:
            print(f"Directory '{directory_path}' already exists.")
        # Set the custom grid
        
    comm.Barrier()
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
    
    # mu_1 = torch.tensor(sp_coords_copy_test[1, 2], dtype=dtype, requires_grad=True)
    # mu_2 = torch.tensor(sp_coords_copy_test[4, 2], dtype=dtype, requires_grad=True)
    # list_paramter = [mu_1, mu_2]
    
    
    # interpolation_input = geo_model_test.interpolation_input
    # print(type(interpolation_input))
    # # If 'sp_coords' is a tensor and you want to convert it to float32
    

    # interpolation_input.surface_points.sp_coords = torch.index_put(
    #                         interpolation_input.surface_points.sp_coords,
    #                         (torch.tensor([1]), torch.tensor([2])),
    #                         mu_1)
    # interpolation_input.surface_points.sp_coords = torch.index_put(
    #                         interpolation_input.surface_points.sp_coords,
    #                         (torch.tensor([4]), torch.tensor([2])),
    #                         mu_2)
    # # # Compute the geological model
    # geo_model_test.solutions = gempy_engine.compute_model(
    #             interpolation_input=interpolation_input,
    #             options=geo_model_test.interpolation_options,
    #             data_descriptor=geo_model_test.input_data_descriptor,
    #             geophysics_input=geo_model_test.geophysics_input,
    #         )
            
    # # Compute and observe the thickness of the geological layer
   
    # custom_grid_values = geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values
    
    
    # grad_K_k =torch.zeros((2, custom_grid_values.shape[0]),dtype=dtype)
    
    # for i in range(len(list_paramter)):
    #     for j in range(custom_grid_values.shape[0]):
    #         y_x = grad(custom_grid_values[j], list_paramter[i],  retain_graph=True)
    #         grad_K_k[i,j] = y_x[0]
    
    ############check if the jacobian calcualted for Gempy is corect ##########    
    # epsilon_data = [1e-13, 1e-12,1e-11, 1e-10, 1e-9,1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    # norm_data = []
    # for epsilon in epsilon_data:
    #     norm_sum = check_gempy_jacobian(epsilon, geo_model_test,mu_1, mu_2, custom_grid_values, grad_K_k.detach())
    #     norm_data.append(norm_sum.detach().numpy())
    # plt.figure(figsize=(10,5))
    # plt.plot(np.log10(np.array(epsilon_data)), np.log10(np.array(norm_data)))
    # plt.xlabel("epsilon")
    # plt.ylabel("error norm sum")
    # plt.savefig("Gempy_Jacobian_test.png")
    # plt.close()
    
    model = MyModel()
    if rank==0:
    
        #torch.multiprocessing.set_start_method("spawn", force=True)
        #torch.multiprocessing.set_sharing_strategy("file_system")
        
        filename_Bayesian_graph =directory_path +"/Bayesian_graph.png"
        
        pyro.clear_param_store()
        
        
        # We can build a probabilistic model using pyro by calling it 
        #dot = pyro.render_model(model.model_test, model_args=(test_list,geo_model_test,num_layers,mesh,degree, dtype),render_distributions=True,filename=filename_Bayesian_graph)
        dot = pyro.render_model(model.create_sample, model_args=(test_list,geo_model_test,num_layers,dtype))
        # Generate 50 samples
        num_samples = 1000 # N
        predictive = Predictive(model.create_sample, num_samples=num_samples)
        samples = predictive(test_list,geo_model_test,num_layers,dtype)
        
        ######store the samples ######
        parameters = torch.stack((samples["mu_1"], samples["mu_2"]), dim=1) # (N, p) = number of sample X number of paramter
    
        np.save(directory_path +"/input.npy",parameters.detach().numpy())
    else:
        parameters = None
    
    comm.Barrier()
    # Broadcast GemPy output to all ranks
    parameters = comm.bcast(parameters, root=0)

    
    # # conductivity/ porosity at each node
    # K = samples["K"] # size of samples X number of grid points , (N X dU) 
    # print(K.shape)
    
    grad_data , J, U =[], [], []
    for i in range(parameters.shape[0]):
        
        mu_1 = parameters[i,0].clone().requires_grad_(True)
        mu_2 = parameters[i,1].clone().requires_grad_(True)
        list_paramter = [mu_1, mu_2]
        
        
        interpolation_input = geo_model_test.interpolation_input
        
        # If 'sp_coords' is a tensor and you want to convert it to float32
        

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
        
        # Identity = torch.eye(custom_grid_values.shape[0], dtype=dtype)
        
        grad_K_k =torch.zeros(( parameters.shape[1],custom_grid_values.shape[0]),dtype=dtype)
        
        for k in range(len(list_paramter)):
            for j in range(custom_grid_values.shape[0]):
                y_x = grad(custom_grid_values[j], list_paramter[k],  retain_graph=True)
                grad_K_k[k,j] = y_x[0]
        
        grad_data.append(grad_K_k)
        
        #print("Custom_grid_values: ", custom_grid_values.shape)
        
        U_data, J_data, =  model.solve_pde(custom_grid_values, mesh, grad_K_k,  degree, comm, rank, dtype)
        if rank ==0:
            J.append(J_data)
            U.append(U_data)
        
    
    #u = samples["u"] # size of samples X number of grid points , (N X M)
    # grad_K_k = torch.stack(grad_data)
    # print(grad_K_k.shape)
    
    
    comm.Barrier()
    
    if rank==0:
        u = np.vstack(U)
        grad_k_u = np.stack(J)
        print(u.shape)
        print(grad_k_u.shape)
        #u = torch.stack(U)
        np.save(directory_path +"/u.npy", u)
        np.save(directory_path +"/Jacobian.npy", grad_k_u)
    
    # mean_u = torch.mean(u,dim=0)
    # np.save(directory_path+"/Mean_u.npy", mean_u)
    # u = u - mean_u
    # N, q = u.shape # N= Number of samples, q = length of u 
    
    # result_sum = torch.zeros(q,q, dtype=dtype)  # Initialize a tensor to accumulate the sum
    # for i in range(N):
    #     u_data = u[i].reshape(-1,1)# Reshape u to [1024, 1]
    #     result_sum += u_data @ u_data.T  # Matrix multiplication to get [1024, 1024]
    # cov_matrix_u = result_sum / N
    # eigenvalues_u, eigenvectors_u = torch.linalg.eig(cov_matrix_u)
    
    # cuttoff = 1e-6
    # mask_u = eigenvalues_u.real > cuttoff
    # eig_u = eigenvalues_u[mask_u].real
    # eig_vec_u = eigenvectors_u[:,mask_u].real
    # np.save(directory_path+"/Truncated_Eigen_Vector_Matrix.npy", eig_vec_u)
    
    # r = eig_vec_u.shape[1] # dimension after reduction
    
    # # conductivity/ porosity at each node
    # K = samples["K"] # size of samples X number of grid points , (N X dU) 
    # gradient of u with k , ∂u/∂K
    #grad_K_u = samples["grad_K_u"] # (N X dU x dU)
    # grad_K_u = torch.stack(J)
    # print(grad_K_u.shape, grad_K_k.shape)
    # # Perform batch matrix multiplication
    # grad_k_u = torch.matmul(grad_K_u, grad_K_k)  # ∂u/∂k = ∂u/∂K * ∂K/∂k,  size of samples X number of paramters X number of grid points(N, p, M) 
    # print(grad_k_u.shape) # size of samples X number of grid points X numper of input paramter N x dU x m
    # grad_k_u = grad_K_u
    # np.save(directory_path +"/Jacobian.npy", grad_k_u.detach().numpy())
    
    
        
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