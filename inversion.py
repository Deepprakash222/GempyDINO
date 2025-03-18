import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import arviz as az
import pandas as pd
from datetime import datetime
import json
import argparse
import io
import contextlib

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
from model_inversion import MyModel

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

# Load the fixed neural network model
class SurrogateModel(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.net = torch.load(model_path)
        self.net.eval()  # Ensure it remains fixed

    def forward(self, theta):
        return self.net(theta)  # Neural network output


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
    directory_path = "./Results_inversion"
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
    
    
    interpolation_input = geo_model_test.interpolation_input
    
    
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
    
    ################################################################################################################
    # Build Likelihood function here
    ################################################################################################################
    
            
    Vk = dl.FunctionSpace(mesh, 'Lagrange', 1)
    Vu  = dl.FunctionSpace(mesh, 'Lagrange', degree) # degree=1 is piecewise linear, degree=2 is piecewise quadratic
    # print( "dim(Vh) = ", Vu.dim() )
    
    # define function for state and adjoint
    u = dl.Function(Vu)
    p = dl.Function(Vu)

    # define Trial and Test Functions
    u_trial, p_trial, k_trial = dl.TrialFunction(Vu), dl.TrialFunction(Vu), dl.TrialFunction(Vk)
    u_test, p_test, k_test = dl.TestFunction(Vu), dl.TestFunction(Vu), dl.TestFunction(Vk)
    
    # ---------------- 2️⃣ Define Boundaries ----------------
    
    class TopBoundary(dl.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[1] - 1) < dl.DOLFIN_EPS

    class BottomBoundary(dl.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[1]) < dl.DOLFIN_EPS

    class LeftBoundary(dl.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[0]) < dl.DOLFIN_EPS

    class RightBoundary(dl.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[0] - 1) < dl.DOLFIN_EPS
    
    class FrontBoundary(dl.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[0]) < dl.DOLFIN_EPS

    class BackBoundary(dl.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[0] - 1) < dl.DOLFIN_EPS


    boundary_parts = dl.MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    #boundary_parts = FacetFunction("size_t", mesh)
    boundary_parts.set_all(0)

    Gamma_top = TopBoundary()
    Gamma_top.mark(boundary_parts, 1)
    Gamma_bottom = BottomBoundary()
    Gamma_bottom.mark(boundary_parts, 2)
    Gamma_left = LeftBoundary()
    Gamma_left.mark(boundary_parts, 3)
    Gamma_right = RightBoundary()
    Gamma_right.mark(boundary_parts, 4)
    Gamma_front = TopBoundary()
    Gamma_front.mark(boundary_parts, 5)
    Gamma_back = BottomBoundary()
    Gamma_back.mark(boundary_parts, 6)
    
    u_L = dl.Constant(100.)
    u_R = dl.Constant(0.)
    
    

    #sigma_bottom = Expression('-(pi/2.0)*sin(2*pi*x[0])', degree=5)
    sigma_right = dl.Constant(10.)
    sigma_left    = dl.Constant(10.)

    # f = Expression('(4.0*pi*pi+pi*pi/4.0)*(sin(2*pi*x[0])*sin((pi/2.0)*x[1]))', degree=5)
    #f = - 10

    
    bc_state = [dl.DirichletBC(Vu, u_L, boundary_parts, 3),
        dl.DirichletBC(Vu, u_R, boundary_parts, 4)]

    bc_adj = [dl.DirichletBC(Vu, dl.Constant(0.), boundary_parts, 3),
        dl.DirichletBC(Vu, dl.Constant(0.), boundary_parts, 4)]
    
    ds = dl.Measure("ds", subdomain_data=boundary_parts)
    
    # ---------------- 3️⃣ Define Discrete k(x,y) Values ----------------
    
    # Define a custom thermal conductivity function
    class ThermalConductivity(dl.UserExpression):
        def __init__(self, custom_grid_values, mesh, **kwargs):
            super().__init__(**kwargs)  # Initialize UserExpression properly
            self.custom_grid_values = custom_grid_values  # Custom grid values passed in
            self.mesh = mesh  # Mesh to check coordinates

            # Create a dictionary of coordinates -> custom values
            self.coord_map = {tuple(self.mesh.coordinates()[i]): self.custom_grid_values[i] 
                            for i in range(self.mesh.num_vertices())}
        def eval(self, value, x):
            # Direct lookup from the coordinate dictionary
            coord_tuple = tuple(x)  # Convert coordinate to tuple for hashing
            if coord_tuple in self.coord_map:
                value[0] = self.coord_map[coord_tuple]
            else:
                value[0] = 1.0  # Default value if no match (shouldn't happen)

        def value_shape(self):
                        return ()  # Scalar function, so empty shape

    # # Create an instance of the thermal conductivity function with custom grid values
    k = ThermalConductivity(custom_grid_values=custom_grid_values, mesh=mesh, degree=0)
    
    
    # Interpolate onto the function space
    k_func = dl.interpolate(k, Vk)
    
    plt.figure(figsize=(15,5))
    nb.plot(mesh,subplot_loc=131, mytitle="Mesh", show_axis='on')
    nb.plot(k_func ,subplot_loc=132, mytitle="k", show_axis='on')
    #nb.plot(m_func,subplot_loc=133, mytitle="m")
    plt.savefig(directory_path+"/Mesh_k_m.png")
    plt.close()
    
    # ---------------- 3️⃣ Define the PDE Weak Formulation  ----------------
    # weak form for setting up the state equation
    a_state = dl.inner( k_func * dl.grad(u_trial), dl.grad(u_test)) * dl.dx
    #L_state = m_func * u_test * dl.dx
    L_state =  dl.Constant(0.) * u_test * dl.dx #+ sigma_left * u_test * ds(3) - sigma_right * u_test * ds(4)
    # solve state equation
    state_A, state_b = dl.assemble_system (a_state, L_state, bc_state)
    utrue = dl.Function(Vu)
    dl.solve (state_A, utrue.vector(), state_b)
    
    ########## Add noise #################
    # noise level
    noise_level = 0.05
    ud = dl.Function(Vu)
    ud.assign(utrue)
    
    # perturb state solution and create synthetic measurements ud
    # ud = u + ||u||/SNR * random.normal
    MAX = ud.vector().norm("linf")
    noise = dl.Vector()
    state_A.init_vector(noise,1)
    parRandom.normal(noise_level * MAX, noise)
    bc_adj[0].apply(noise)
    bc_adj[1].apply(noise)

    ud.vector().axpy(1., noise)

    # plot
    nb.multi1_plot([utrue, ud], ["State solution with mtrue", "Synthetic observations"])
    plt.savefig(directory_path+"/output_noise.png")
    plt.close()
    
    mean_u = np.load("Mean_u.npy")
    Eigen_vector = np.load("Truncated_Eigen_Vector_Matrix.npy")
    y_obs = Eigen_vector.T @ (ud.vector().get_local() - mean_u)
    #y_obs = Eigen_vector.T @ (utrue.vector().get_local() - mean_u)
    # Load the saved model
    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")
    
    
    dtype_ = torch.float32
    Pyro_model = MyModel()
    y_obs = torch.tensor(y_obs, device=device, dtype = dtype_)
    model_path = "model_weights_with_jacobian.pth"
    #model_path = "model_weights.pth"
    # Load the trained model
    model = SurrogateModel(model_path)
    #model = torch.load(model_path)
    model.to(device)
    #model.eval()
    

    test_list_=[]
    test_list_.append({"update":"interface_data","id":torch.tensor([1]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[1,2],dtype=dtype_), "std":torch.tensor(0.06,dtype=dtype_)}})
    test_list_.append({"update":"interface_data","id":torch.tensor([4]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[4,2],dtype=dtype_), "std":torch.tensor(0.06,dtype=dtype_)}})
    #print(output, output.shape)
    
    filename_Bayesian_graph =directory_path +"/Bayesian_graph.png"
    #dot = pyro.render_model(model.model_test, model_args=(normalised_hsi,test_list,geo_model_test,mean_init,cov_init,factor,num_layers,posterior_condition, scale, cluster, alpha, beta),render_distributions=True,filename=filename_Bayesian_graph)
    dot = pyro.render_model(Pyro_model.model_test, model_args=(model, test_list_, y_obs, device, dtype_),render_distributions=True)
    ################################################################################
    # Prior
    ################################################################################
    pyro.set_rng_seed(42)
    prior = Predictive(Pyro_model.model_test, num_samples=1000)(model, test_list_, y_obs, device, dtype_)
    # Key to avoid
    avoid_key =[]
    for i in range(len(test_list)+1):
                if i==0:
                    avoid_key.append(f'mu_{i+1} < mu_{i+1} + 2 * std')
                elif i==len(test_list):
                    avoid_key.append(f'mu_{i} > mu_{i} - 2 * std')
                else:
                    avoid_key.append(f'mu_{i} > mu_{i+1} ')
                    
    avoid_key.append('log_likelihood')
    #avoid_key = ['mu_1 < 0','mu_1 > mu_2','mu_2 > mu_3', 'mu_3 > mu_4' , 'mu_4 > -83']
    # Create sub-dictionary without the avoid_key
    prior = dict((key, value) for key, value in prior.items() if key not in avoid_key)
    plt.figure(figsize=(8,10))
    data = az.from_pyro(prior=prior)
    az.plot_trace(data.prior)
    filename_prior_plot = directory_path + "/prior.png"
    plt.savefig(filename_prior_plot)
    plt.close()
    # Pyro_model.model_test(model=model,input_parameter= test_list_, y_obs=y_obs, device=device, dtype=torch.float32)
    pyro.primitives.enable_validation(is_validate=True)
    nuts_kernel = NUTS(Pyro_model.model_test, step_size=0.01, adapt_step_size=True, target_accept_prob=0.75, max_tree_depth=10, init_strategy=init_to_mean)
    mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=1000, warmup_steps=1000,num_chains=5, disable_validation=False)
    mcmc.run(model, test_list_, y_obs, device, dtype_)
   
    posterior_samples = mcmc.get_samples()
    
    print("MCMC summary results")
    
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        mcmc.summary()  
        summary_output = buf.getvalue()

    
    print(summary_output)
    
    with open(f'{directory_path}/mcmc_summary_p.txt', 'w') as f:
        f.write(summary_output)
    
    
    posterior_predictive = Predictive(Pyro_model.model_test, posterior_samples)(model, test_list_, y_obs, device, dtype_)
    plt.figure(figsize=(8,10))
    data = az.from_pyro(posterior=mcmc, prior=prior, posterior_predictive=posterior_predictive)
    az.plot_trace(data)
    filename_posteriro_plot = directory_path + "/posterior.png"
    plt.savefig(filename_posteriro_plot)
    plt.close()
    
    ###############################################TODO################################
    # Plot and save the file for each parameter
    ###################################################################################
    for i in range(len(test_list)):
        plt.figure(figsize=(8,10))
        az.plot_density(
        data=[data.posterior, data.prior],
        shade=.9,
        var_names=['mu_' +str(i+1)],
        data_labels=["Posterior Predictive", "Prior Predictive"],
        colors=[default_red, default_blue],
        )
        filename_mu = directory_path + "/mu_"+str(i+1)+".png"
        plt.savefig(filename_mu)
        plt.close()
    #print("Posterior samples of k:", posterior_samples)
    post_mu_1_mean, post_mu_1_std= posterior_samples["mu_1"].mean(), posterior_samples["mu_1"].std()
    post_mu_2_mean, post_mu_2_std= posterior_samples["mu_2"].mean(), posterior_samples["mu_2"].std()
    print("True mu_1: ",torch.tensor(sp_coords_copy_test[1,2], dtype=dtype_))
    print("Initial std_1: ",torch.tensor(0.06,dtype=dtype_))
    print("post_mu_1_mean: ",post_mu_1_mean)
    print("post_mu_1_std: ",post_mu_1_std)
    print("True mu_2: ",torch.tensor(sp_coords_copy_test[4,2],dtype=dtype_))
    print("post_mu_2_mean: ",post_mu_2_mean)
    print("post_mu_2_std: ",post_mu_2_std)
    
    #Set to evaluation mode
    ######## Load Neural network#########
    # Instantiate model
    # model = NeuralNet(input_dim=2, hidden_dim=32, output_dim=u_output.shape[1])
    # model.to(device)
    
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