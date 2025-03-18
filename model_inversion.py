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
import sys
import os
sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../") )
from hippylib import *

import logging
import math
import numpy as np

import matplotlib.pyplot as plt
#%matplotlib inline

from pyro.nn import PyroModule, PyroSample

class MyModel(PyroModule):
    def __init__(self):
        super(MyModel, self).__init__()
    
    #@config_enumerate
    def model_test(self, model, input_parameter, y_obs, device, dtype):
            num_layers = len(input_parameter)
            Random_variable={}
            input_data=[]
            counter=1
            for parameter in input_parameter:
                # Check if user wants to create random variable based on modifying the surface points of gempy
                if parameter["update"]=="interface_data":
                    # Check what kind of distribution is needed
                    if parameter["prior_distribution"]=="normal":
                        mean = parameter["normal"]["mean"]
                        std  = parameter["normal"]["std"]
                        Random_variable["mu_"+ str(counter)] = pyro.sample("mu_"+ str(counter), dist.Normal(mean, std)).to(device)
                        input_data.append(Random_variable["mu_"+ str(counter)])
                        # print(Random_variable["mu_"+ str(counter)])
                        
                    elif parameter["prior_distribution"]=="uniform":
                        min = parameter["uniform"]["min"]
                        max = parameter["uniform"]["min"]
                        Random_variable["mu_"+ str(parameter['id'])] = pyro.sample("mu_"+ str(parameter['id']), dist.Uniform(min, max)).to(device)
                        #print(counter)
                        #counter=counter+1
                        
                    else:
                        print("We have to include the distribution")
                counter=counter+1
                
            
            for i in range(len(input_parameter)+1):
                if i==0:
                    pyro.sample(f'mu_{i+1} < mu_{i+1} + 2 * std', dist.Delta(torch.tensor(1.0, dtype=dtype)), obs=(Random_variable[f'mu_{i+1}'] < input_parameter[0]["normal"]["mean"] + 2 * input_parameter[0]["normal"]["std"]))
                elif i==len(input_parameter):
                    pyro.sample(f'mu_{i} > mu_{i} - 2 * std', dist.Delta(torch.tensor(1.0, dtype=dtype)), obs=(Random_variable[f"mu_{i}"] > input_parameter[-1]["normal"]["mean"] - 2 * input_parameter[-1]["normal"]["std"]))
                else:
                    pyro.sample(f'mu_{i} > mu_{i+1} ', dist.Delta(torch.tensor(1.0, dtype=dtype)), obs=(Random_variable[f"mu_{i}"] > Random_variable[f"mu_{i+1}"]))
            
            input_k = torch.tensor(input_data, device=device)
            model.eval()
            y_pred = model(input_k)
            
            # Likelihood of observed y given y_pred
            likelihood= pyro.sample("obs", dist.Normal(y_pred, 0.01), obs=y_obs)
            
            ####
            