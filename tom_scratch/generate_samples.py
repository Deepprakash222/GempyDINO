
import torch
from torch.autograd import grad
import pyro
import pyro.distributions as dist
from pyro.infer import Predictive

from pyro.nn import PyroModule, PyroSample
import gempy_engine
from gempy_engine.core.backend_tensor import BackendTensor
import gempy as gp

class GempyModel(PyroModule):
    def __init__(self, interpolation_input_, geo_model_test, num_layers, dtype):
        super(GempyModel, self).__init__()
        
        BackendTensor.change_backend_gempy(engine_backend=gp.data.AvailableBackends.PYTORCH)
        self.interpolation_input_ = interpolation_input_
        self.geo_model_test = geo_model_test
        self.num_layers = num_layers
        self.dtype = dtype
        self.geo_model_test.interpolation_options.sigmoid_slope = 200
        
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

        return parameters
    
    def GenerateOutputSamples(self, Inputs_samples):
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
        
        
        return torch.stack(m_true) , torch.stack(dmdc)


    