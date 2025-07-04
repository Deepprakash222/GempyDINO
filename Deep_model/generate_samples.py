
import torch
import numpy as np
from torch.autograd import grad
import pyro
import pyro.distributions as dist
from pyro.infer import Predictive

from pyro.nn import PyroModule, PyroSample

import gempy as gp
import gempy_engine
from gempy_engine.core.backend_tensor import BackendTensor


import dolfin as dl

from helpers import *
import json

class GempyModel(PyroModule):
    def __init__(self, interpolation_input_, geo_model_test, num_layers, slope, dtype, device):
        super(GempyModel, self).__init__()
        BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH)
        self.gempy_engine = gempy_engine
        self.interpolation_input_ = interpolation_input_
        self.geo_model_test = geo_model_test
        self.num_layers = num_layers
        self.dtype = dtype
        self.geo_model_test.interpolation_options.sigmoid_slope = slope
        self.device = device
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
        
    
    def create_sample(self):
        
            """
            This Pyro model represents the probabilistic aspects of the geological model.
            It defines a prior distribution for the top layer's location and
            computes the thickness of the geological layer as an observed variable.

            
            interpolation_input_: represents the dictionary of random variables for surface parameters
            geo_model_test : gempy model
            
            num_layers: represents the number of layers we want to include in the model
            
            """

            Random_variable ={}
            
            # interpolation_input = self.geo_model_test.interpolation_input
            
            # Create a random variable based on the provided dictionary used to modify input data of gempy
            counter=1
            for interpolation_input_data in self.interpolation_input_[:self.num_layers]:
                
                # Check if user wants to create random variable based on modifying the surface points of gempy
                if interpolation_input_data["update"]=="interface_data":
                    # Check what kind of distribution is needed
                    if interpolation_input_data["prior_distribution"]=="normal":
                        mean = interpolation_input_data["normal"]["mean"]
                        std  = interpolation_input_data["normal"]["std"]
                        Random_variable["mu_"+ str(counter)] = pyro.sample("mu_"+ str(counter), dist.Normal(mean, std))
                        #print(Random_variable["mu_"+ str(counter)])
                        
                    elif interpolation_input_data["prior_distribution"]=="uniform":
                        min = interpolation_input_data["uniform"]["min"]
                        max = interpolation_input_data["uniform"]["min"]
                        Random_variable["mu_"+ str(interpolation_input_data['id'])] = pyro.sample("mu_"+ str(interpolation_input_data['id']), dist.Uniform(min, max))

                        
                    else:
                        print("We have to include the distribution")
                
                    # # Check which co-ordinates direction we wants to allow and modify the surface point data
                counter=counter+1
                
          
            
            # for i in range(len(self.interpolation_input_)+1):
            #     if i==0:
            #         pyro.sample(f'mu_{i+1} < mu_{i+1} + 2 * std', dist.Delta(torch.tensor(1.0, dtype=self.dtype)), obs=(Random_variable[f'mu_{i+1}'] < self.interpolation_input_[0]["normal"]["mean"] + 2 * self.interpolation_input_[0]["normal"]["std"]))
            #     elif i==len(self.interpolation_input_):
            #         pyro.sample(f'mu_{i} > mu_{i} - 2 * std', dist.Delta(torch.tensor(1.0, dtype=self.dtype)), obs=(Random_variable[f"mu_{i}"] > self.interpolation_input_[-1]["normal"]["mean"] - 2 * self.interpolation_input_[-1]["normal"]["std"]))
            #     else:
            #         pyro.sample(f'mu_{i} > mu_{i+1} ', dist.Delta(torch.tensor(1.0, dtype=self.dtype)), obs=(Random_variable[f"mu_{i}"] > Random_variable[f"mu_{i+1}"]))
                    
    def GenerateInputSamples(self, number_samples):
        
        pyro.clear_param_store()
        # We can build a probabilistic model using pyro by calling it 
        
        dot = pyro.render_model(self.create_sample, model_args=())
        # Generate 50 samples
        num_samples = number_samples # N
        predictive = Predictive(self.create_sample, num_samples=num_samples)
        samples = predictive()
        
        samples_list=[]
        for i in range(len(self.interpolation_input_)):
            samples_list.append(samples["mu_"+str(i+1)].reshape(-1,1))
        ######store the samples ######
        parameters=torch.hstack(samples_list) # (N, p) = number of sample X number of paramter

        return parameters.cpu().detach().numpy()
    
    def GempyForward(self, *params):
        index=0
        interpolation_input = self.geo_model_test.interpolation_input
        
        
        for interpolation_input_data in self.interpolation_input_[:self.num_layers]:
            # Check which co-ordinates direction we wants to allow and modify the surface point data
            if interpolation_input_data["direction"]=="X":
                interpolation_input.surface_points.sp_coords = torch.index_put(
                    interpolation_input.surface_points.sp_coords,
                    (torch.tensor([interpolation_input_data["id"]]), torch.tensor([0])),
                    params[index])
            elif interpolation_input_data["direction"]=="Y":
                interpolation_input.surface_points.sp_coords = torch.index_put(
                    interpolation_input.surface_points.sp_coords,
                    (torch.tensor([interpolation_input_data["id"]]), torch.tensor([1])),
                    params[index])
            elif interpolation_input_data["direction"]=="Z":
                interpolation_input.surface_points.sp_coords = torch.index_put(
                    interpolation_input.surface_points.sp_coords,
                    (interpolation_input_data["id"], torch.tensor([2])),
                    params[index])
                
            else:
                print("Wrong direction")
            
            index=index+1
        
        self.geo_model_test.solutions = self.gempy_engine.compute_model(
                    interpolation_input=interpolation_input,
                    options=self.geo_model_test.interpolation_options,
                    data_descriptor=self.geo_model_test.input_data_descriptor,
                    geophysics_input=self.geo_model_test.geophysics_input,
                )
        
        # Compute and observe the thickness of the geological layer
    
        m_samples = self.geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values
        return m_samples
    
    def GenerateOutputSamples_(self, Inputs_samples):
        from torch.autograd.functional import jacobian
        
        Inputs_samples = torch.tensor(Inputs_samples, dtype=self.dtype, device=self.device)
        m_data =[]
        dmdc_data =[]
        for i in range(Inputs_samples.shape[0]):
            params_tuple = tuple([Inputs_samples[i,j].clone().requires_grad_(True) for j in range(Inputs_samples.shape[1])])
            m_samples = self.GempyForward(*params_tuple)
            m_data.append(m_samples)
            J = jacobian(self.GempyForward, params_tuple)
            J_matrix = torch.tensor([[J[j][i] for j in range(len(J))] for i in  range(J[0].shape[0])])
            dmdc_data.append(J_matrix)
        
        return torch.stack(m_data).detach().numpy() , torch.stack(dmdc_data).detach().numpy()


def generate_input_output_gempy_data(mesh, nodes, number_samples, comm, device, slope=200, filename=None):
    
    mesh_coordinates = mesh.coordinates()
    
    global_indices = mesh.topology().global_indices(0)  # vertex global IDs
    data ={}
    geo_model_test = create_initial_gempy_model(refinement=3, save=True)
    if mesh_coordinates.shape[1]==2:
        xyz_coord = np.insert(mesh_coordinates, 1, 0, axis=1)
    elif mesh_coordinates.shape[1]==3:
        xyz_coord = mesh_coordinates
    gp.set_custom_grid(geo_model_test.grid, xyz_coord=xyz_coord)
    geo_model_test.interpolation_options.mesh_extraction = False
    
    sp_coords_copy_test = geo_model_test.interpolation_input.surface_points.sp_coords.copy()
    
    ###############################################################################
    # Make a list of gempy parameter which would be treated as a random variable
    ###############################################################################
    dtype =torch.float64
    test_list=[]
    std = 0.03  # 0.125 , 4*std = gap between two layers
    test_list.append({"update":"interface_data","id":torch.tensor([1]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[1,2],dtype=dtype, device=device), "std":torch.tensor(std,dtype=dtype, device=device)}})
    test_list.append({"update":"interface_data","id":torch.tensor([2]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[2,2],dtype=dtype, device=device), "std":torch.tensor(std,dtype=dtype, device=device)}})
    test_list.append({"update":"interface_data","id":torch.tensor([3]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[3,2],dtype=dtype, device=device), "std":torch.tensor(std,dtype=dtype, device=device)}})
    test_list.append({"update":"interface_data","id":torch.tensor([6]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[6,2],dtype=dtype, device=device), "std":torch.tensor(std,dtype=dtype, device=device)}})
    test_list.append({"update":"interface_data","id":torch.tensor([7]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[7,2],dtype=dtype, device=device), "std":torch.tensor(std,dtype=dtype, device=device)}})
    test_list.append({"update":"interface_data","id":torch.tensor([8]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[8,2],dtype=dtype, device=device), "std":torch.tensor(std,dtype=dtype, device=device)}})
    test_list.append({"update":"interface_data","id":torch.tensor([11]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[11,2],dtype=dtype, device=device), "std":torch.tensor(std,dtype=dtype, device=device)}})
    test_list.append({"update":"interface_data","id":torch.tensor([12]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[12,2],dtype=dtype, device=device), "std":torch.tensor(std,dtype=dtype, device=device)}})
    test_list.append({"update":"interface_data","id":torch.tensor([13]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[13,2],dtype=dtype, device=device), "std":torch.tensor(std,dtype=dtype, device=device)}})
    num_layers = len(test_list) # length of the list
    
    
    Gempy = GempyModel(test_list, geo_model_test, num_layers, slope=slope,  dtype=dtype, device=device)
    
    comm.Barrier()
    if comm.Get_rank()==0:
        c = Gempy.GenerateInputSamples(number_samples=number_samples)
        # print(c)
        data["input"] = c.tolist()
        
    else:
        c = None
    
    # Broadcast GemPy output to all ranks
    
    c = comm.bcast(c, root=0)
    
    #m_data, dmdc_data = Gempy.GenerateOutputSamples(Inputs_samples=c)
    m_data, dmdc_data = Gempy.GenerateOutputSamples_(Inputs_samples=c)
   
    local_results = [(int(global_indices[idx]), m_data[:,idx], dmdc_data[:,idx]) for idx in range(global_indices.shape[0])]
    comm.Barrier()
    
    all_results = comm.gather(local_results, root=0)
    if comm.Get_rank() == 0:
    # Initialize global output and gradient tensors
        global_output = np.zeros((c.shape[0],nodes))
        global_gradient = np.zeros((c.shape[0],nodes,num_layers))
        for result_list in all_results:
            for idx_, output_, grad_ in result_list:
                global_output[:,idx_] = output_
                global_gradient[:,idx_] = grad_
        data["Gempy_output"] = global_output.tolist()
        data["Jacobian_Gempy"] = global_gradient.tolist()
    
    return data

def create_true_data(mesh, nodes, slope=200, filename=None):
    
    mesh_coordinates = mesh.coordinates()
    data ={}
    geo_model_test = create_initial_gempy_model(refinement=3, save=True)
    if mesh_coordinates.shape[1]==2:
        xyz_coord = np.insert(mesh_coordinates, 1, 0, axis=1)
    elif mesh_coordinates.shape[1]==3:
        xyz_coord = mesh_coordinates
    gp.set_custom_grid(geo_model_test.grid, xyz_coord=xyz_coord)
    geo_model_test.interpolation_options.mesh_extraction = False
    sol = gp.compute_model(geo_model_test)
    
    geo_model_test.interpolation_options.sigmoid_slope = slope
    gp.compute_model(geo_model_test)
    sp_coords_copy_test = geo_model_test.interpolation_input.surface_points.sp_coords.copy()
    m_initial = geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values
    
    return m_initial, sp_coords_copy_test , geo_model_test

def generate_final_model(geo_model,interpolation_input_,num_layers,  posterior_data, slope=200):
    
    BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH)
    
    interpolation_input = geo_model.interpolation_input
    geo_model.interpolation_options.sigmoid_slope = slope
    index=0
    for interpolation_input_data in interpolation_input_[:num_layers]:
        # Check which co-ordinates direction we wants to allow and modify the surface point data
        if interpolation_input_data["direction"]=="X":
            interpolation_input.surface_points.sp_coords = torch.index_put(
                interpolation_input.surface_points.sp_coords,
                (torch.tensor([interpolation_input_data["id"]]), torch.tensor([0])),
                posterior_data[index])
        elif interpolation_input_data["direction"]=="Y":
            interpolation_input.surface_points.sp_coords = torch.index_put(
                interpolation_input.surface_points.sp_coords,
                (torch.tensor([interpolation_input_data["id"]]), torch.tensor([1])),
                posterior_data[index])
        elif interpolation_input_data["direction"]=="Z":
            interpolation_input.surface_points.sp_coords = torch.index_put(
                interpolation_input.surface_points.sp_coords,
                (interpolation_input_data["id"], torch.tensor([2])),
                posterior_data[index])
            # print("posterior_data : ",posterior_data[index])
            # print("interpolation_input : ",interpolation_input.surface_points.sp_coords)
            
        else:
            print("Wrong direction")
        
        index=index+1
        

    Phyiscal_Data = geo_model.transform.apply_inverse(interpolation_input.surface_points.sp_coords.detach().numpy())
    posterior_data_ =[Phyiscal_Data[1,2],Phyiscal_Data[2,2],Phyiscal_Data[3,2],Phyiscal_Data[6,2],Phyiscal_Data[7,2],Phyiscal_Data[8,2],Phyiscal_Data[11,2],Phyiscal_Data[12,2],Phyiscal_Data[13,2]]
    
    geo_model_test = create_final_gempy_model(posterior_data_, refinement=7, save=True)
    geo_model_test.interpolation_options.mesh_extraction = False
    sol = gp.compute_model(geo_model_test)
    geo_model_test.interpolation_options.sigmoid_slope = slope
    gp.compute_model(geo_model_test)
        
        
def generate_input_output_gempy_data_(mesh, nodes, number_samples, slope=200, filename=None):
    
    mesh_coordinates = mesh.coordinates()
    
    data ={}
    geo_model_test = create_initial_gempy_model(refinement=3, save=True)
    if mesh_coordinates.shape[1]==2:
        xyz_coord = np.insert(mesh_coordinates, 1, 0, axis=1)
    elif mesh_coordinates.shape[1]==3:
        xyz_coord = mesh_coordinates
    gp.set_custom_grid(geo_model_test.grid, xyz_coord=xyz_coord)
    geo_model_test.interpolation_options.mesh_extraction = False
    
    sp_coords_copy_test = geo_model_test.interpolation_input.surface_points.sp_coords.copy()
    # m_initial = geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values
    # print(m_initial)
    ################################################################################
    # Store the Initial Interface data and orientation data
    ################################################################################
    df_sp_init = geo_model_test.surface_points.df
    df_or_init = geo_model_test.orientations.df
    
    filename_initial_sp = "./Initial_sp.csv"
    filename_initial_op = "./Initial_op.csv"
    df_sp_init.to_csv(filename_initial_sp)
    df_or_init.to_csv(filename_initial_op)
    ###############################################################################
    # Make a list of gempy parameter which would be treated as a random variable
    ###############################################################################
    dtype =torch.float64
    test_list=[]
    std = 0.03  # 0.125 , 4*std = gap between two layers
    test_list.append({"update":"interface_data","id":torch.tensor([1]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[1,2],dtype=dtype), "std":torch.tensor(std,dtype=dtype)}})
    test_list.append({"update":"interface_data","id":torch.tensor([2]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[2,2],dtype=dtype), "std":torch.tensor(std,dtype=dtype)}})
    test_list.append({"update":"interface_data","id":torch.tensor([3]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[3,2],dtype=dtype), "std":torch.tensor(std,dtype=dtype)}})
    test_list.append({"update":"interface_data","id":torch.tensor([6]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[6,2],dtype=dtype), "std":torch.tensor(std,dtype=dtype)}})
    test_list.append({"update":"interface_data","id":torch.tensor([7]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[7,2],dtype=dtype), "std":torch.tensor(std,dtype=dtype)}})
    test_list.append({"update":"interface_data","id":torch.tensor([8]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[8,2],dtype=dtype), "std":torch.tensor(std,dtype=dtype)}})
    test_list.append({"update":"interface_data","id":torch.tensor([11]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[11,2],dtype=dtype), "std":torch.tensor(std,dtype=dtype)}})
    test_list.append({"update":"interface_data","id":torch.tensor([12]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[12,2],dtype=dtype), "std":torch.tensor(std,dtype=dtype)}})
    test_list.append({"update":"interface_data","id":torch.tensor([13]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[13,2],dtype=dtype), "std":torch.tensor(std,dtype=dtype)}})
    
    num_layers = len(test_list) # length of the list

    Gempy = GempyModel(test_list, geo_model_test, num_layers, slope, dtype=torch.float64)
    
    c = Gempy.GenerateInputSamples(number_samples=number_samples)
    exit()
    m_data, dmdc_data = Gempy.GenerateOutputSamples_(Inputs_samples=c)
    
    data["input"] = c
    data["Gempy_output"] =m_data
    data["Jacobian_Gempy"] = dmdc_data
    return data
    
        