import os,sys
from datetime import datetime
import numpy as np
import random
import torch
import pyro
import dolfin as dl
import h5py


#from helpers import *
from generate_samples2 import *
from mpi4py import MPI

import warnings
warnings.filterwarnings("ignore")


def main():
    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        #device = torch.device("mps")
        device = torch.device("cpu")
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    # Get the MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    
    print(f"Process {rank} of {size}")
    
    # ---------------- 1️⃣ Create the Mesh ----------------
    nx = 31
    ny = 31
    mesh = dl.UnitSquareMesh(comm, nx, ny)
    current_folder = os.getcwd()
    Sampling_type = current_folder.split("/")[-1]
    print("Sampling_type:", Sampling_type)
    parent_folder = os.path.dirname(current_folder)
    model_name = parent_folder.split("/")[-1]
    print("Model name:", model_name)
    
    
    nodes = (nx+1)*(ny+1)
    if rank==0:
        directory_path = "../../Results/"+str(model_name)+"/"+str(Sampling_type)+"/Nodes_"+str(nodes)
        if not os.path.exists(directory_path):
            # Create the directory if it does not exist
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' was created.")
        else:
            print(f"Directory '{directory_path}' already exists.")
    else:
        directory_path = None
    
    comm.Barrier()
    
    data = generate_input_output_gempy_data(mesh=mesh,nodes=nodes, number_samples=14000, comm=comm, device=device)
    
    comm.Barrier()
    
    if rank==0:
        filename = directory_path + "/gempy_data.h5"
        c ,m_data, dmdc_data = np.array(data["input"]),np.array(data["Gempy_output"]), np.array(data["Jacobian_Gempy"])
        print("Shapes-" , "Gempy Input: ", c.shape, "Gempy Output:", m_data.shape, "Jacobian shape:", dmdc_data.shape)
        
        with h5py.File(filename, "w") as f:
            f["input"] = c
            f["Gempy_output"] = m_data
            f["Jacobian_Gempy"] = dmdc_data
            
        
    

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
