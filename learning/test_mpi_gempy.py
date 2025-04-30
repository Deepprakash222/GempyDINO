import dolfin as dl
import ufl
from datetime import datetime
import sys
import os
sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../") )
from hippylib import *
import torch
dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.autograd import grad
from mpi4py import MPI

def f(x, a, b):
    return torch.stack([a * x[0]**2, b**2 * x[1], a+b] )

def main():
    # Get the MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
        
    comm.Barrier()
    print(f"Process {rank} of {size}")
    nx = 3
    ny = 3
    mesh = dl.UnitSquareMesh(comm, nx, ny)
    coords = torch.tensor(mesh.coordinates(), dtype=dtype)
    print(coords)
    global_indices = mesh.topology().global_indices(0)  # vertex global IDs
    print(global_indices)
    a = torch.tensor(2, dtype=dtype, requires_grad=True)
    b = torch.tensor(2, dtype=dtype, requires_grad=True)
    list_paramter=[a,b]
    ouptut_data = []
    grad_data = []
    for i in range(coords.shape[0]):
        y = f(coords[i], a,b )
        ouptut_data.append(y)
        grad_K_k =torch.zeros((len(list_paramter),y.shape[0]),dtype=dtype)
        for k in range(len(list_paramter)):
            for j in range(y.shape[0]):
                y_x = grad(y[j], list_paramter[k],  retain_graph=True)
                grad_K_k[k,j] = y_x[0]
       
        grad_data.append(grad_K_k.T)
    Jacobian = torch.stack(grad_data)
   
    local_results = [(int(global_indices[idx]), ouptut_data[idx], Jacobian[idx]) for idx in range(global_indices.shape[0])]
    comm.Barrier()
    
    all_results = comm.gather(local_results, root=0)
    if rank == 0:
    # Initialize global output and gradient tensors
        global_output = torch.zeros((16,3), dtype=dtype)
        global_gradient = torch.zeros((16,3,2), dtype=dtype)
        for result_list in all_results:
            for idx_, output_, grad_ in result_list:
                global_output[idx_] = output_
                global_gradient[idx_] = grad_

        # Now you have global_output and global_gradient fully populated
        #print("Global Output Tensor: ", global_output.shape)
        #print("Global Gradient Tensor: ", global_gradient.shape)

    comm.Barrier()
    exit()
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