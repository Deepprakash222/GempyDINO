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


import gempy as gp
import gempy_engine
import gempy_viewer as gpv
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_probability.plot_posterior import default_red, default_blue, PlotPosterior

from Initial_model import *
from model import MyModel
dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    # Parameters for the grid
    grid_size = 5  # 100 x 100 grid
    x_values = np.linspace(0, 1000, grid_size)  # Create 100 points between -50 and 50
    z_values = np.linspace(-1000, 0, grid_size)  # Create 100 points between -50 and 50
    y_value = 0  # Fixed y coordinate

    # Create the grid
    x, z = np.meshgrid(x_values, z_values)  # Generate x and z grid points
    y = np.full_like(x, y_value)  # Create y grid with all values as 0

    # Combine into a 3D array of points (10000, 3)
    xyz_coord = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
    
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
    test_list.append({"update":"interface_data","id":torch.tensor([1]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[1,2],dtype=dtype, device=device), "std":torch.tensor(0.06,dtype=dtype, device=device)}})
    test_list.append({"update":"interface_data","id":torch.tensor([4]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[4,2],dtype=dtype, device=device), "std":torch.tensor(0.06,dtype=dtype, device=device)}})
    
    num_layers = len(test_list) # length of the list


    model = MyModel()
    
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")
    
    filename_Bayesian_graph =directory_path +"/Bayesian_graph.png"
    pyro.clear_param_store()
    #pyro.set_rng_seed(46)
    
    # We can build a probabilistic model using pyro by calling it 
    #dot = pyro.render_model(model.model_test, model_args=(test_list,geo_model_test,num_layers, dtype, device),render_distributions=True,filename=filename_Bayesian_graph)
    #dot = pyro.render_model(model.model_test, model_args=(test_list,geo_model_test,num_layers, dtype,device))
    
    ####
    #  Example to show how to update the gempy paramter and show it's gradient
    ####
    
    mu_1 = torch.tensor(sp_coords_copy_test[1, 2], dtype=dtype, device=device, requires_grad=True)

    interpolation_input = geo_model_test.interpolation_input
    
    interpolation_input.surface_points.sp_coords = torch.index_put(
                            interpolation_input.surface_points.sp_coords,
                            (torch.tensor([1]), torch.tensor([2])),
                            mu_1)
    # # Compute the geological model
    geo_model_test.solutions = gempy_engine.compute_model(
                interpolation_input=interpolation_input,
                options=geo_model_test.interpolation_options,
                data_descriptor=geo_model_test.input_data_descriptor,
                geophysics_input=geo_model_test.geophysics_input,
            )
            
    # Compute and observe the thickness of the geological layer
    custom_grid_values = geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values
    gradient_argument = torch.ones_like(custom_grid_values)
    # Compute gradient
    #custom_grid_values.backward(gradient_argument)

    y_x = grad (custom_grid_values[0], mu_1,  create_graph=True)
    # Get the gradient
    print(y_x)
    #print(custom_grid_values)
    
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