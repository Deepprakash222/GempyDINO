import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import arviz as az
import pandas as pd
from datetime import datetime

import torch

def L2_accuaracy(true_Data, nueral_network_output):
    '''
        Find the expectation of relative error norm
    '''
    RL=0
    for i in range(true_Data.shape[0]):
        RL += torch.norm(true_Data[i] - nueral_network_output[i])/ torch.norm(true_Data[i])
    Expecation_RL = RL/true_Data.shape[0]
    L2_error = 1- torch.sqrt(Expecation_RL)
    return L2_error
def H1_accuracy(Jacobain_true, Jacobian_neural_network):
    '''
        Find the expectation of relative error norm of Jacobain
    '''
    J_RL = 0
    c = torch.norm(Jacobain_true - Jacobian_neural_network, p='fro', dim=(1, 2))/ torch.norm(Jacobain_true,p='fro', dim=(1, 2))
    print(torch.mean(c))
    for i in range(Jacobain_true.shape[0]):
       
        J_RL += torch.norm(Jacobain_true[i] - Jacobian_neural_network[i], p='fro', dim=(0, 1))/ torch.norm(Jacobain_true[i],p='fro', dim=(0, 1))
    Expecation_J_RL = J_RL/Jacobain_true.shape[0]
    print(Expecation_J_RL)
    H1_error = 1- torch.sqrt(Expecation_J_RL)
    return H1_error
    
