import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import arviz as az
import pandas as pd
import os
import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive, EmpiricalMarginal
from pyro.infer.autoguide import init_to_mean, init_to_median, init_to_value
from pyro.infer.inspect import get_dependencies
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete
from pyro.contrib.gp.kernels import Matern32

import gempy as gp
import gempy_engine
import gempy_viewer as gpv
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_probability.plot_posterior import default_red, default_blue, PlotPosterior

import scipy.io
from scipy.stats import zscore
from sklearn.manifold import TSNE

from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import KMeans

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
#%matplotlib inline

logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)


from pyro.nn import PyroModule, PyroSample
# Change the backend to PyTorch for probabilistic modeling
BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH)

def FD_u(u_init, u_current, epsilon):
    delta_u = (u_current - u_init)/epsilon
    return delta_u

def perturbation(mesh, custom_grid_values, epsilon, iteration, u_init, J_init):
    error_sum = 0
    
    for i in range(iteration):
        # create mesh and define function spaces
        # Define mesh
        oder =1
        Vk = dl.FunctionSpace(mesh, 'Lagrange', 1)
        Vu = dl.FunctionSpace(mesh, 'Lagrange', oder)
        
        
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
        n = k_func.vector().get_local().shape[0]
        random_index= np.random.randint(0, n)
        k_func.vector()[random_index] = k_func.vector()[random_index] + epsilon
        # ktrue = dl.Constant(2.0)
        # define function for state and adjoint
        u_current = dl.Function(Vu)
        p = dl.Function(Vu)

        # define Trial and Test Functions
        u_trial, p_trial, k_trial = dl.TrialFunction(Vu), dl.TrialFunction(Vu), dl.TrialFunction(Vk)
        u_test, p_test, k_test = dl.TestFunction(Vu), dl.TestFunction(Vu), dl.TestFunction(Vk)

        

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
        
        u_L = dl.Constant(1.)
        u_R = dl.Constant(0.)
        
        

        #sigma_bottom = Expression('-(pi/2.0)*sin(2*pi*x[0])', degree=5)
        sigma_right = dl.Constant(10.)
        sigma_left    = dl.Constant(10.)


        
        bc_state = [dl.DirichletBC(Vu, u_L, boundary_parts, 3),
            dl.DirichletBC(Vu, u_R, boundary_parts, 4)]


        
        ds = dl.Measure("ds", subdomain_data=boundary_parts)
        
        # We added our weak formulation to the cost functional to form the Langerangian 
        ##########################################################################################
        # weak form for setting up the state equation
        ##########################################################################################

        # ---------------- 3️⃣ Define the PDE Weak Formulation  ----------------
        # weak form for setting up the state equation
        a_state = dl.inner( k_func * dl.grad(u_trial), dl.grad(u_test)) * dl.dx
        #L_state = m_func * u_test * dl.dx
        L_state =  dl.Constant(0.) * u_test * dl.dx #+ sigma_left * u_test * ds(3) - sigma_right * u_test * ds(4)
        # solve state equation
        state_A, state_b = dl.assemble_system (a_state, L_state, bc_state)
        dl.solve (state_A, u_current.vector(), state_b)
        
        du_dk_tilde_k = FD_u(u_init=u_init.vector().get_local(), u_current=u_current.vector().get_local(), epsilon=epsilon)
        J_true = J_init[:,random_index]
        error = np.linalg.norm(J_true - du_dk_tilde_k )
        error_sum = error_sum + error
        
        
    return error_sum

class MyModel(PyroModule):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def create_sample(self, interpolation_input_,geo_model_test,num_layers,dtype):
        
            """
            This Pyro model represents the probabilistic aspects of the geological model.
            It defines a prior distribution for the top layer's location and
            computes the thickness of the geological layer as an observed variable.

            
            interpolation_input_: represents the dictionary of random variables for surface parameters
            geo_model_test : gempy model
            
            num_layers: represents the number of layers we want to include in the model
            
            """

            Random_variable ={}
            
            interpolation_input = geo_model_test.interpolation_input
            
            # Create a random variable based on the provided dictionary used to modify input data of gempy
            counter=1
            for interpolation_input_data in interpolation_input_[:num_layers]:
                
                # Check if user wants to create random variable based on modifying the surface points of gempy
                if interpolation_input_data["update"]=="interface_data":
                    # Check what kind of distribution is needed
                    if interpolation_input_data["prior_distribution"]=="normal":
                        mean = interpolation_input_data["normal"]["mean"]
                        std  = interpolation_input_data["normal"]["std"]
                        Random_variable["mu_"+ str(counter)] = pyro.sample("mu_"+ str(counter), dist.Normal(mean, std))
                        print(Random_variable["mu_"+ str(counter)])
                        
                    elif interpolation_input_data["prior_distribution"]=="uniform":
                        min = interpolation_input_data["uniform"]["min"]
                        max = interpolation_input_data["uniform"]["min"]
                        Random_variable["mu_"+ str(interpolation_input_data['id'])] = pyro.sample("mu_"+ str(interpolation_input_data['id']), dist.Uniform(min, max))
                        #print(counter)
                        #counter=counter+1
                        
                    else:
                        print("We have to include the distribution")
                
                
                    # # Check which co-ordinates direction we wants to allow and modify the surface point data
                    # if interpolation_input_data["direction"]=="X":
                    #     interpolation_input.surface_points.sp_coords = torch.index_put(
                    #         interpolation_input.surface_points.sp_coords,
                    #         (torch.tensor([interpolation_input_data["id"]]), torch.tensor([0])),
                    #         Random_variable["mu_"+ str(counter)])
                    # elif interpolation_input_data["direction"]=="Y":
                    #     interpolation_input.surface_points.sp_coords = torch.index_put(
                    #         interpolation_input.surface_points.sp_coords,
                    #         (torch.tensor([interpolation_input_data["id"]]), torch.tensor([1])),
                    #         Random_variable["mu_"+ str(counter)])
                    # elif interpolation_input_data["direction"]=="Z":
                    #     interpolation_input.surface_points.sp_coords = torch.index_put(
                    #         interpolation_input.surface_points.sp_coords,
                    #         (interpolation_input_data["id"], torch.tensor([2])),
                    #         Random_variable["mu_"+ str(counter)])
                        
                    # else:
                    #     print("Wrong direction")
                
                counter=counter+1
            
          
            
            for i in range(len(interpolation_input_)+1):
                if i==0:
                    pyro.sample(f'mu_{i+1} < mu_{i+1} + 2 * std', dist.Delta(torch.tensor(1.0, dtype=dtype)), obs=(Random_variable[f'mu_{i+1}'] < interpolation_input_[0]["normal"]["mean"] + 2 * interpolation_input_[0]["normal"]["std"]))
                elif i==len(interpolation_input_):
                    pyro.sample(f'mu_{i} > mu_{i} - 2 * std', dist.Delta(torch.tensor(1.0, dtype=dtype)), obs=(Random_variable[f"mu_{i}"] > interpolation_input_[-1]["normal"]["mean"] - 2 * interpolation_input_[-1]["normal"]["std"]))
                else:
                    pyro.sample(f'mu_{i} > mu_{i+1} ', dist.Delta(torch.tensor(1.0, dtype=dtype)), obs=(Random_variable[f"mu_{i}"] > Random_variable[f"mu_{i+1}"]))
                
            
            # Update the model with the new top layer's location
            
            #print(interpolation_input.surface_points.sp_coords)
            
            # # # Compute the geological model
            # geo_model_test.solutions = gempy_engine.compute_model(
            #     interpolation_input=interpolation_input,
            #     options=geo_model_test.interpolation_options,
            #     data_descriptor=geo_model_test.input_data_descriptor,
            #     geophysics_input=geo_model_test.geophysics_input,
            # )
            
            # # Compute and observe the thickness of the geological layer
            # custom_grid_values = geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values
            
            # pyro.deterministic("K", custom_grid_values)  # Register y explicitly
    #@config_enumerate
    
    def solve_pde(self, custom_grid_values, mesh, gradient_K_k, degree, comm, rank,  dtype):

            ################################################################################################################
            # Build Likelihood function here
            ################################################################################################################
            
            # comm = dl.MPI.comm_world
            # rank = dl.MPI.rank(comm)
            # size = dl.MPI.size(comm)
            # print(f"Process {rank} out of {size}")
            
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
            
            u_L = dl.Constant(1.)
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
            
            bc_tan = [dl.DirichletBC(Vu, dl.Constant(0.), boundary_parts, 3),
                dl.DirichletBC(Vu, dl.Constant(0.), boundary_parts, 4)]
            
            ds = dl.Measure("ds", subdomain_data=boundary_parts)
            
            # ---------------- 3️⃣ Define Discrete k(x,y) Values ----------------
            
            # # Define a custom thermal conductivity function
            # class ThermalConductivity(dl.UserExpression):
            #     def __init__(self, custom_grid_values, mesh, **kwargs):
            #         super().__init__(**kwargs)  # Initialize UserExpression properly
            #         self.custom_grid_values = custom_grid_values  # Custom grid values passed in
            #         self.mesh = mesh  # Mesh to check coordinates

            #         # Create a dictionary of coordinates -> custom values
            #         self.coord_map = {tuple(self.mesh.coordinates()[i]): self.custom_grid_values[i] 
            #                         for i in range(self.mesh.num_vertices())}
            #     def eval(self, value, x):
            #         # Direct lookup from the coordinate dictionary
            #         coord_tuple = tuple(x)  # Convert coordinate to tuple for hashing
            #         if coord_tuple in self.coord_map:
            #             value[0] = self.coord_map[coord_tuple]
            #         else:
            #             value[0] = 1.0  # Default value if no match (shouldn't happen)

            #     def value_shape(self):
            #                     return ()  # Scalar function, so empty shape

            # # # Create an instance of the thermal conductivity function with custom grid values
            # k = ThermalConductivity(custom_grid_values=custom_grid_values, mesh=mesh, degree=0)
            
            
            # # Interpolate onto the function space
            # k_func = dl.interpolate(k, Vk)
            
            # Assuming Vk is the function space
            k_func = dl.Function(Vk)  

            # Get mesh node coordinates and DoF coordinates
            mesh_coords = mesh.coordinates()
            dof_coords = Vk.tabulate_dof_coordinates().reshape((-1, mesh.geometry().dim()))

            # Round to avoid floating-point mismatches and create a dictionary
            mesh_coord_map = {tuple(np.round(mesh_coords[i], decimals=10)): i for i in range(len(mesh_coords))}

            # Find correct indices for DoF coordinates
            correct_order = [mesh_coord_map[tuple(np.round(coord, decimals=10))] for coord in dof_coords]

            # Reorder custom grid values
            ordered_values = custom_grid_values[correct_order]
            
            # Assign reordered values
            k_func.vector().set_local(ordered_values.detach().numpy())
            
            # plt.figure(figsize=(15,5))
            # nb.plot(mesh,subplot_loc=131, mytitle="Mesh", show_axis='on')
            # nb.plot(k_func ,subplot_loc=132, mytitle="k", show_axis='on')
            # #nb.plot(m_func,subplot_loc=133, mytitle="m")
            # plt.savefig("Mesh_k_m.png")
            # plt.close()
            
            # ---------------- 3️⃣ Define the PDE Weak Formulation  ----------------
            # weak form for setting up the state equation
            a_state = dl.inner( k_func * dl.grad(u_trial), dl.grad(u_test)) * dl.dx
            #L_state = m_func * u_test * dl.dx
            L_state =  dl.Constant(0.) * u_test * dl.dx #+ sigma_left * u_test * ds(3) - sigma_right * u_test * ds(4)
            # solve state equation
            state_A, state_b = dl.assemble_system (a_state, L_state, bc_state)
            dl.solve (state_A, u.vector(), state_b)
            
            dof_coords = Vu.tabulate_dof_coordinates()
            #dof_coords
            
            # Get Dirichlet boundary condition indices
            dirichlet_dof_indices = set()  # Store unique indices

            for bc in bc_adj:  # Iterate over all Dirichlet BCs
                dirichlet_dof_indices.update(bc.get_boundary_values().keys())
            
            

            ############################################################################
            # Solve tangent equation to get the Sensitivity. 
            ############################################################################
            S_solutions = []
            #print(gradient_K_k.shape)
            # #Get mesh node coordinates and DoF coordinates
            
            mesh_coords = mesh.coordinates()
            #print("mesh_coords" , mesh_coords.shape)
            #print(mesh_coords)
            dof_coords = Vu.tabulate_dof_coordinates().reshape((-1, mesh.geometry().dim()))
            #print("dof_coords",dof_coords.shape)
            #print(dof_coords)
            
            # Round to avoid floating-point mismatches and create a dictionary
            mesh_coord_map = {tuple(np.round(mesh_coords[i], decimals=10)): i for i in range(len(mesh_coords))}
            correct_order = [mesh_coord_map[tuple(np.round(coord, decimals=10))] for coord in dof_coords]
            
            #print(correct_order)
            # print(gradient_K_k)
            if rank ==0:
                Sensitivty =[]
            else:
                Sensitivty = None
                
            comm.Barrier()
            
            for k in range(gradient_K_k.shape[0]):
                # Convert column j of dK/dk into a FEniCS function
                #k_grad_func = dl.Function(Vu)
                # Sensitivity equation
                dK_dk_fenics = dl.Function(Vu)
                S = dl.TrialFunction(Vu)
                w = dl.TestFunction(Vu)
                # Reorder custom grid values

                #Find correct indices for DoF coordinates
                
                ordered_values = gradient_K_k[k,:][correct_order]
                #print(ordered_values)
                #comm.Barrier()  
                
                dK_dk_fenics.vector().set_local(ordered_values.detach().numpy())
                # print(ordered_values)
                # comm.Barrier()
                
                # Interpolate onto the function space
                # # Create an instance of the thermal conductivity function with custom grid values
                #k_grad = ThermalConductivity(custom_grid_values=gradient_K_k[k,:], mesh=mesh, degree=0)
                
                
                # Interpolate onto the function space
                #k_grad_func = dl.interpolate(k_grad, Vu)
        
                a_S = dl.inner(k_func * dl.grad(S), dl.grad(w)) * dl.dx
                #L_S = - dl.inner(k_grad_func * dl.grad(u), dl.grad(w)) * dl.dx
                L_S = - dl.inner(dK_dk_fenics * dl.grad(u), dl.grad(w)) * dl.dx

                S_sol = dl.Function(Vu)
                dl.solve(a_S == L_S, S_sol, bc_tan)
                
                comm.Barrier()
                # print("Process :", rank)
                # Convert local solution to numpy array
                local_values = S_sol.vector().get_local()
    
                # Gather sizes of all local solutions
                sizes = comm.gather(len(local_values), root=0)
                
                if rank == 0:
                    # Allocate space for full solution
                    total_size = sum(sizes)
                    full_sensitivty = np.zeros(total_size, dtype=np.float64)
                   
                else:
                    full_sensitivty = None
                
                # Gather all local solutions
                comm.Gatherv(sendbuf=local_values, recvbuf=(full_sensitivty, sizes), root=0)
                
                if rank == 0:
                    #print("Full solution gathered:", full_sensitivty)
                    Sensitivty.append(full_sensitivty)
                    
                
                # Store solution for this parameter
                #S_solutions.append(S_sol.vector().get_local())
            #S_solutions = np.stack(S_solutions)
            
            
            
            # ############################################################################
            # # Solve adjoint equation to get the adjoint matrix.
            # ############################################################################
            # a_adj = ufl.inner(k_func * ufl.grad(p_trial), ufl.grad(p_test)) * ufl.dx
            # L_adj = dl.Constant(0) * p_test * ufl.dx
            
            
            
            # adj_A, adj_b = dl.assemble_system(a_adj, L_adj, bc_adj)
            
            # adjoint_matrix_1 = np.zeros_like(adj_A.array())
            
            # # for i, x_0 in enumerate(Vu.tabulate_dof_coordinates()):
            # #     if i not in sorted(dirichlet_dof_indices): # check for non Boundary points
            # #         adj_b[i] = -1
            # #         p_sol = dl.Function(Vu)
            # #         dl.solve (adj_A, p_sol.vector(), adj_b)
            # #         #print(adj_b.get_local())
            # #         adj_b[i] = 0
            # #         adjoint_matrix_1[:,i] = p_sol.vector().get_local() 
            
            # Mass_matrix = dl.inner(u_trial, u_test) * dl.dx
            # M_u = dl.assemble(Mass_matrix)
            # # print(M_u.array())
            
            # # mesh_coords = mesh.coordinates()
            # # #print(mesh_coords)
            # # dof_coords = Vu.tabulate_dof_coordinates().reshape((-1, mesh.geometry().dim()))
            # # #print(dof_coords)
            # # # Round to avoid floating-point mismatches and create a dictionary
            # # mesh_coord_map = {tuple(np.round(mesh_coords[i], decimals=10)): i for i in range(len(mesh_coords))}
            # # correct_order = [mesh_coord_map[tuple(np.round(coord, decimals=10))] for coord in dof_coords]
            # adjoint_matrix = np.zeros_like(adj_A.array())
            # comm.Barrier()
            # #print(adjoint_matrix.shape)
            
            # #column_identity = np.zeros_like(np.array(correct_order))
            # # A = Identity[correct_order,:]
            # # print(A)
            
            # # print(Vu.tabulate_dof_coordinates().shape)
            # # # Find rows and columns that are NOT all zero
            # # nonzero_rows = (A != 0).any(dim=1)  # Keep rows with at least one nonzero value
            # # nonzero_cols = (A != 0).any(dim=0)  # Keep columns with at least one nonzero value

            # # # Remove zero rows and columns
            # # A_reduced = A[nonzero_rows][:, nonzero_cols]

            # #print(A_reduced)
            # #exit()
            # for k, x_0 in enumerate(Vu.tabulate_dof_coordinates()):
            # #for k, x_0 in enumerate(mesh.coordinates()):
            #     #if i not in sorted(dirichlet_dof_indices): # check for non Boundary points
                    
            #         # b = ThermalConductivity(custom_grid_values=column_identity, mesh=mesh, degree=0)
            #         # # Interpolate onto the function space
            #         # b_func = dl.interpolate(b, Vu)
            #         f = dl.Function(Vu)
                    
            #         #column_identity = - A[:,k]
                    
            #         p_sol = dl.Function(Vu)
            #         #print(p_sol.vector().get_local())
                    
            #         f.vector()[k]=-1
            #         #comm.Barrier()
            #         #f.vector().set_local(column_identity.detach().numpy())
                    
            #         #print(correct_order)
            #         #f.vector()[:]= column_identity[correct_order]
                
            #         #f.vector().apply("insert")
            #         L_adj  = f * p_test  * ufl.dx
            #         #L_adj  = b_func * p_test  * ufl.dx
                    
            #         adj_A, adj_b = dl.assemble_system (a_adj, L_adj, bc_adj)
                    
            #         dl.solve (adj_A, p_sol.vector(), adj_b)
            #         #print(adj_b.get_local())
            #         #adj_b[i] = 0
            #         adjoint_matrix[:,k] = p_sol.vector().get_local() 
            #         #column_identity[i] = 0
            # # # Compute Darcy velocity v = -k ∇u
            # # W = dl.VectorFunctionSpace(mesh, "CG", 1)  # Velocity space (vector field)
            # # velocity = dl.project(- k_func * dl.grad(u), W)
            # # # Extract components of velocity
            # # v_x, v_y = velocity.split()  
            
        
            # # plt.figure(figsize=(15,5))
            # # #nb.plot(u,subplot_loc=121, mytitle="u(k_ini, m_ini)")
            # # nb.plot(u,subplot_loc=131, mytitle="P(k_ini)")
            # # #nb.plot(p,subplot_loc=142, mytitle="Adjoint")
            # # nb.plot(v_x,subplot_loc=132, mytitle="v_x")
            # # nb.plot(v_y,subplot_loc=133, mytitle="v_y")
            # # # Adjust layout to prevent overlap
            # # plt.savefig("plot_u_p_.png")
            # # plt.close()
            
            # # Sensitivity 
            # C_equ   = ufl.inner( k_trial * ufl.grad(u), ufl.grad(p_test)) * ufl.dx
            # # assemble matrix C
            # C =  dl.assemble(C_equ)

            # M_equ = ufl.inner(k_trial, k_test) * ufl.dx
            # M_ = ufl.inner(u_trial, u_test) * ufl.dx
            # # assemble matrix M
            # M_k = dl.assemble(M_equ)
            # M_u = dl.assemble(M_)


            # C_ = C.array()[1:-1,1:-1]
            # J =   (C.array().T @ adjoint_matrix_1).T
            # #C_tilde = M_k.array() @ C.array() @ np.linalg.inv(M_u.array())
            # ########check the Jacobian if it is correct or not #####
            # # epsilon_data = np.array([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
            # # error_data = []
            # # for i in range(len(epsilon_data)):
            # #     error_data.append(perturbation(mesh=mesh, custom_grid_values=custom_grid_values, epsilon=epsilon_data[i], iteration=500, u_init=u, J_init=J))
            # # plt.figure(figsize=(15,5))
            # # plt.plot(np.log10(epsilon_data), np.log10(np.array(error_data)))
            # # plt.xlabel("log10(epsilon)")
            # # plt.ylabel("log10(error_norm_sum)")
            # # plt.savefig("Test_Jacobain_matrix.png")
            # # plt.close()
            
            # # Test_adjoint = adjoint_matrix_1 @ M_u.array() - adjoint_matrix
            # # Test_adjoint[abs(Test_adjoint)< 1e-10] = 0
            
            
            
            # J = np.linalg.solve(M_u.array(),  adjoint_matrix.T @ C.array())
            # #J_tilde = np.linalg.solve(M_u.array().T,  J)
            
            # # pyro.deterministic("grad_K_u", torch.tensor(J))  # Register y explicitly
            # # pyro.deterministic("u", torch.tensor(u.vector().get_local()))  # Register y explicitly
            # # Convert local solution to numpy array
            
            comm.Barrier()
            #print("Process :", rank)
            # Convert local solution to numpy array
            local_values = u.vector().get_local()
            
            # Gather sizes of all local solutions
            sizes = comm.gather(len(local_values), root=0)
            
            if rank == 0:
                # Allocate space for full solution
                total_size = sum(sizes)
                full_solution = np.zeros(total_size, dtype=np.float64)
                
            else:
                full_solution = None
                
            
            # Gather all local solutions
            comm.Gatherv(sendbuf=local_values, recvbuf=(full_solution, sizes), root=0)
            

            if rank == 0:
                # print("Full solution gathered:", full_solution)
                Sensitivty = np.vstack(Sensitivty).T
                #print(Sensitivty)
                
            #print(J)
            #print(Sensitivty)
            # print(correct_order)
            # print(gradient_K_k.shape)
            
            # A = Sensitivty.T - J @ (gradient_K_k[:,correct_order]).detach().numpy().T
            # A[abs(A)<1e-10] = 0.0
            
            # print(A)
            # print(np.sum(A))
            # comm.Barrier()
            # exit()
            
            return full_solution, Sensitivty
            #return torch.tensor(J) , torch.tensor(u.vector().get_local())
            
            #####################################TODO##########################################
            
            
            
            
                
