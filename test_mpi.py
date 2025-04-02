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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.func import jacrev

import gempy as gp
import gempy_engine
import gempy_viewer as gpv
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_probability.plot_posterior import default_red, default_blue, PlotPosterior

from Initial_model import *
from model import MyModel

dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

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

from mpi4py import MPI

def main():
    # Get the MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    comm.Barrier()
    print(f"Process {rank} of {size}")

    # Create a 2D mesh
    n = 4095
    degree = 1
    mesh = dl.UnitSquareMesh(n, n)
    # nb.plot(mesh)

    Vh  = dl.FunctionSpace(mesh, 'Lagrange', degree)

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

    boundary_parts = dl.MeshFunction("size_t", mesh, 1)
    boundary_parts.set_all(0)

    Gamma_top = TopBoundary()
    Gamma_top.mark(boundary_parts, 1)
    Gamma_bottom = BottomBoundary()
    Gamma_bottom.mark(boundary_parts, 2)
    Gamma_left = LeftBoundary()
    Gamma_left.mark(boundary_parts, 3)
    Gamma_right = RightBoundary()
    Gamma_right.mark(boundary_parts, 4)
    
    u_L = dl.Constant(0.)
    u_R = dl.Constant(0.)

    sigma_bottom = dl.Expression('-(pi/2.0)*sin(2*pi*x[0])', degree=5)
    sigma_top    = dl.Constant(0.)

    f = dl.Expression('(4.0*pi*pi+pi*pi/4.0)*(sin(2*pi*x[0])*sin((pi/2.0)*x[1]))', degree=5)

    bcs = [dl.DirichletBC(Vh, u_L, boundary_parts, 3),
        dl.DirichletBC(Vh, u_R, boundary_parts, 4)]

    ds = dl.Measure("ds", subdomain_data=boundary_parts)
    
    u = dl.TrialFunction(Vh)
    v = dl.TestFunction(Vh)
    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
    L = f*v*ufl.dx + sigma_top*v*ds(1) + sigma_bottom*v*ds(2)

    u = dl.Function(Vh)

    #dl.solve(a == L, uh, bcs=bcs)
    A, b = dl.assemble_system(a,L, bcs=bcs)
    dl.solve(A, u.vector(), b)
    
    u_e = dl.Expression('sin(2*pi*x[0])*sin((pi/2.0)*x[1])', degree=5)

    # Get local mesh coordinates and solution
    local_coords = Vh.tabulate_dof_coordinates().flatten()  # Flatten for gathering
    local_values = u.vector().get_local()
    u_true = dl.interpolate(u_e,Vh)
    local_true_values = u_true.vector().get_local()
    
    # Gather sizes of all local solutions
    sizes = comm.gather(len(local_values), root=0)
    coord_sizes = comm.gather(len(local_coords), root=0)  # Coordinates are dim*N, not just N

    if rank == 0:
        # Allocate space for full solution and coordinates
        total_size = sum(sizes)
        full_solution = np.zeros(total_size, dtype=np.float64)
        full_solution_true = np.zeros(total_size, dtype=np.float64)

        total_coord_size = sum(coord_sizes)
        full_coords = np.zeros(total_coord_size, dtype=np.float64)

    else:
        full_solution = None
        full_coords = None
        full_solution_true = None

    # Gather all local solutions and coordinates
    comm.Gatherv(sendbuf=local_values, recvbuf=(full_solution, sizes), root=0)
    comm.Gatherv(sendbuf=local_true_values, recvbuf=(full_solution_true, sizes), root=0)
    comm.Gatherv(sendbuf=local_coords, recvbuf=(full_coords, coord_sizes), root=0)

    if rank == 0:
        # Reshape coordinates back to (N, dim)
        full_coords = full_coords.reshape(-1, 2)
        # print(full_coords)
        # Compute analytical solution for 2D
        x_coord = full_coords[:, 0]
        y_coord = full_coords[:, 1]
        
        u_true = np.sin(2 * np.pi * x_coord) * np.sin((np.pi/2) * y_coord)
        
        print("Numerical solution shape:", full_solution.shape)
        print("Mesh coordinates shape:", full_coords.shape)
        print("Analytical solution shape:",full_solution_true.shape )

        # Error analysis
        error = full_solution - u_true
        #error = full_solution - full_solution_true
        #error = u_true - full_solution_true
        error[np.abs(error) < 1e-10] = 0
        print(error)
        print(np.sum(np.abs(error)))
    


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