
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
    def __init__(self, interpolation_input_, geo_model_test, num_layers, dtype):
        super(GempyModel, self).__init__()
        
        BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH)
        self.interpolation_input_ = interpolation_input_
        self.geo_model_test = geo_model_test
        self.num_layers = num_layers
        self.dtype = dtype
        self.geo_model_test.interpolation_options.sigmoid_slope = 200
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
          
            
            for i in range(len(self.interpolation_input_)+1):
                if i==0:
                    pyro.sample(f'mu_{i+1} < mu_{i+1} + 2 * std', dist.Delta(torch.tensor(1.0, dtype=self.dtype)), obs=(Random_variable[f'mu_{i+1}'] < self.interpolation_input_[0]["normal"]["mean"] + 2 * self.interpolation_input_[0]["normal"]["std"]))
                elif i==len(self.interpolation_input_):
                    pyro.sample(f'mu_{i} > mu_{i} - 2 * std', dist.Delta(torch.tensor(1.0, dtype=self.dtype)), obs=(Random_variable[f"mu_{i}"] > self.interpolation_input_[-1]["normal"]["mean"] - 2 * self.interpolation_input_[-1]["normal"]["std"]))
                else:
                    pyro.sample(f'mu_{i} > mu_{i+1} ', dist.Delta(torch.tensor(1.0, dtype=self.dtype)), obs=(Random_variable[f"mu_{i}"] > Random_variable[f"mu_{i+1}"]))
                    
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

        return parameters.detach().numpy()
    
    def GenerateOutputSamples(self, Inputs_samples):
        Inputs_samples = torch.tensor(Inputs_samples, dtype=self.dtype)
        m_true = []
        dmdc = []
        for i in range(Inputs_samples.shape[0]):
            mu_1 = Inputs_samples[i,0].clone().requires_grad_(True)
            mu_2 = Inputs_samples[i,1].clone().requires_grad_(True)
            list_paramter = [mu_1, mu_2]
            
            
            interpolation_input = self.geo_model_test.interpolation_input
            
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
            self.geo_model_test.solutions = gempy_engine.compute_model(
                        interpolation_input=interpolation_input,
                        options=self.geo_model_test.interpolation_options,
                        data_descriptor=self.geo_model_test.input_data_descriptor,
                        geophysics_input=self.geo_model_test.geophysics_input,
                    )
                    
            # Compute and observe the thickness of the geological layer
        
            m_samples = self.geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values
            grad_K_k =torch.zeros(( Inputs_samples.shape[1], m_samples.shape[0]),dtype=self.dtype)
            
            for k in range(len(list_paramter)):
                for j in range(m_samples.shape[0]):
                    y_x = grad(m_samples[j], list_paramter[k],  retain_graph=True)
                    grad_K_k[k,j] = y_x[0]
            
            dmdc.append(grad_K_k.T)
            m_true.append(m_samples)
        
        
        return torch.stack(m_true).detach().numpy() , torch.stack(dmdc).detach().numpy()


def generate_input_output_gempy_data(mesh, nodes, number_samples, comm, filename=None):
    
    mesh_coordinates = mesh.coordinates()
    global_indices = mesh.topology().global_indices(0)  # vertex global IDs
    data ={}
    geo_model_test = create_initial_gempy_model_3_layer(refinement=7, save=True)
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
    test_list.append({"update":"interface_data","id":torch.tensor([1]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[1,2],dtype=dtype), "std":torch.tensor(0.06,dtype=dtype)}})
    test_list.append({"update":"interface_data","id":torch.tensor([4]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[4,2],dtype=dtype), "std":torch.tensor(0.06,dtype=dtype)}})

    num_layers = len(test_list) # length of the list

    Gempy = GempyModel(test_list, geo_model_test, num_layers, dtype=torch.float64)
    comm.Barrier()
    if comm.Get_rank()==0:
        c = Gempy.GenerateInputSamples(number_samples=number_samples)
        # print(c)
        data["input"] = c.tolist()
        
    else:
        c = None
    
    # Broadcast GemPy output to all ranks
    c = comm.bcast(c, root=0)
    m_data, dmdc_data = Gempy.GenerateOutputSamples(Inputs_samples=c)
    
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
    
        
def generate_input_output_gempy_data_(mesh, nodes, number_samples, filename=None):
    
    mesh_coordinates = mesh.coordinates()
    
    data ={}
    geo_model_test = create_initial_gempy_model_3_layer(refinement=7, save=True)
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
    test_list.append({"update":"interface_data","id":torch.tensor([1]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[1,2],dtype=dtype), "std":torch.tensor(0.06,dtype=dtype)}})
    test_list.append({"update":"interface_data","id":torch.tensor([4]), "direction":"Z", "prior_distribution":"normal","normal":{"mean":torch.tensor(sp_coords_copy_test[4,2],dtype=dtype), "std":torch.tensor(0.06,dtype=dtype)}})

    num_layers = len(test_list) # length of the list

    Gempy = GempyModel(test_list, geo_model_test, num_layers, dtype=torch.float64)
    
    c = Gempy.GenerateInputSamples(number_samples=number_samples)
        
    m_data, dmdc_data = Gempy.GenerateOutputSamples(Inputs_samples=c)
    
    data["input"] = c
    data["Gempy_output"] =m_data
    data["Jacobian_Gempy"] = dmdc_data
    return data
        
        