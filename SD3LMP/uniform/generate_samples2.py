from zmq import device

import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer import Predictive
from pyro.nn import PyroModule

import random

import torch

import gempy as gp
import gempy_engine
from gempy_engine.core.backend_tensor import BackendTensor

import dolfin as dl

from helpers import *


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

        seed = 2025
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        pyro.set_rng_seed(seed)

    def create_sample(self):
        Random_variable = {}
        counter = 1
        for interpolation_input_data in self.interpolation_input_[:self.num_layers]:
            if interpolation_input_data["update"] == "interface_data":
                if interpolation_input_data["prior_distribution"] == "normal":
                    mean = interpolation_input_data["normal"]["mean"]
                    std = interpolation_input_data["normal"]["std"]
                    Random_variable["mu_" + str(counter)] = pyro.sample(
                        "mu_" + str(counter), dist.Normal(mean, std))
                elif interpolation_input_data["prior_distribution"] == "uniform":
                    min_ = interpolation_input_data["uniform"]["min"]
                    max_ = interpolation_input_data["uniform"]["max"]
                    Random_variable["mu_" + str(counter)] = pyro.sample(
                        "mu_" + str(counter), dist.Uniform(min_, max_))
                else:
                    print("Distribution not implemented.")
            counter += 1

    def GenerateInputSamples(self, number_samples):
        
        pyro.clear_param_store()
        predictive = Predictive(self.create_sample, num_samples=number_samples)
        samples = predictive()
        samples_list = []
        for i in range(len(self.interpolation_input_)):
            samples_list.append(samples["mu_" + str(i + 1)].reshape(-1, 1))
        parameters = torch.hstack(samples_list)  # (N, p)
        return parameters.cpu().detach().numpy()

    def _apply_params(self, params):
        """Apply parameter values to the interpolation input in-place."""
        interpolation_input = self.geo_model_test.interpolation_input
        
        
        for index, interpolation_input_data in enumerate(self.interpolation_input_[:self.num_layers]):
            direction = interpolation_input_data["direction"]
            id_ = interpolation_input_data["id"]
            coord_idx = {"X": 0, "Y": 1, "Z": 2}.get(direction)
            if coord_idx is None:
                print(f"Wrong direction: {direction}")
                continue
            interpolation_input.surface_points.sp_coords = torch.index_put(
                interpolation_input.surface_points.sp_coords,
                (id_, torch.tensor([coord_idx])),
                params[index],
            )
        
        return interpolation_input

    def GempyForward(self, *params):
        """Run GemPy forward model; returns output values at custom grid points."""
        interpolation_input = self._apply_params(params)
        self.geo_model_test.solutions = self.gempy_engine.compute_model(
            interpolation_input=interpolation_input,
            options=self.geo_model_test.interpolation_options,
            data_descriptor=self.geo_model_test.input_data_descriptor,
            geophysics_input=self.geo_model_test.geophysics_input,
        )
        return self.geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values

    def GempyForwardWithGrad(self, param_vec):
        """
        One forward pass + O(num_params) JVP passes for the full Jacobian.
        Correct for ANY output structure (independent or not).
        """
        params_tuple = tuple(
            param_vec[j].clone().requires_grad_(True)
            for j in range(param_vec.shape[0])
        )
        
        J_cols = []
        m_samples = None

        for j in range(len(params_tuple)):
            # Unit tangent vector: 1 in direction j, 0 elsewhere
            tangents = tuple(
                torch.ones_like(p) if i == j else torch.zeros_like(p)
                for i, p in enumerate(params_tuple)
            )
            
            # jvp gives: primals = m_samples, tangents = J @ e_j = j-th column of J
            primals, jvp_col = torch.autograd.functional.jvp(
                self.GempyForward,
                params_tuple,
                tangents,
                create_graph=False,
            )

            if m_samples is None:
                m_samples = primals  # free — we get the forward pass output too
            
            J_cols.append(jvp_col)  # (num_nodes,) — exact dm_i/dp_j for all i

        J_matrix = torch.stack(J_cols, dim=1)  # (num_nodes, num_params)
        
        return m_samples.detach(), J_matrix.detach()

    # ------------------------------------------------------------------
    # MPI-aware output generation
    # ------------------------------------------------------------------

    def GenerateOutputSamples_local(self, Inputs_samples, local_node_indices):
        """
        Compute forward model + Jacobian for all samples, but only STORE
        results for the node indices owned by this MPI rank.

        Parameters
        ----------
        Inputs_samples : np.ndarray, shape (N, num_params)
        local_node_indices : 1-D array of int
            Global node IDs assigned to this rank.

        Returns
        -------
        m_local    : np.ndarray (N, len(local_node_indices))
        grad_local : np.ndarray (N, len(local_node_indices), num_params)
        """
        Inputs_samples = torch.tensor(Inputs_samples, dtype=self.dtype, device=self.device)
        N = Inputs_samples.shape[0]
        num_local = len(local_node_indices)
        #print(local_node_indices)
        m_local = np.zeros((N, num_local))
        grad_local = np.zeros((N, num_local, self.num_layers))

        for i in range(N):
            param_vec = Inputs_samples[i]          # (num_params,)
            m_full, J_full = self.GempyForwardWithGrad(param_vec)
            # m_full  : (total_nodes,)
            # J_full  : (total_nodes, num_params)
            m_local[i] = m_full[local_node_indices].numpy()
            grad_local[i] = J_full[local_node_indices].numpy()
        
        return m_local, grad_local


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def generate_input_output_gempy_data(mesh, nodes, number_samples, comm, device, slope=200, filename=None):
    """
    MPI-parallel sample generation.

    Strategy
    --------
    * Rank 0 draws input samples and broadcasts them.
    * Every rank owns a disjoint subset of mesh nodes (DolFIN partitioning).
    * Each rank runs the full GemPy forward model for every sample but only
      retains the rows that correspond to its local nodes — this avoids any
      redundant storage while keeping the computation embarrassingly parallel
      across samples × ranks.
    * Results are gathered on rank 0.
    """
    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    torch.use_deterministic_algorithms(True)

    rank = comm.Get_rank()
    mesh_coordinates = mesh.coordinates()
    global_indices = mesh.topology().global_indices(0)   # local→global vertex map

    data = {}

    # ---- build GemPy model ----
    geo_model_test = create_initial_gempy_model(refinement=3, save=False)

    if mesh_coordinates.shape[1] == 2:
        xyz_coord = np.insert(mesh_coordinates, 1, 0, axis=1)
    elif mesh_coordinates.shape[1]==3:
        xyz_coord = mesh_coordinates

    gp.set_custom_grid(geo_model_test.grid, xyz_coord=xyz_coord)
    geo_model_test.interpolation_options.mesh_extraction = False

    sp_coords_copy_test = geo_model_test.interpolation_input.surface_points.sp_coords.copy()
    
    # Save initial interface / orientation data (rank 0 only to avoid races)
    if rank == 0:
        geo_model_test.surface_points.df.to_csv("./Initial_sp.csv")
        geo_model_test.orientations.df.to_csv("./Initial_op.csv")

    ###############################################################################
    # Make a list of gempy parameter which would be treated as a random variable
    ###############################################################################
    dtype = torch.float64
    std = 0.02  # 4*std = gap between two layers

    # Interface point 0 — std interval
    min_0 = torch.tensor(sp_coords_copy_test[0, 2], dtype=dtype, device=device) - torch.tensor(std, dtype=dtype, device=device)
    max_0 = torch.tensor(sp_coords_copy_test[0, 2], dtype=dtype, device=device) + torch.tensor(std, dtype=dtype, device=device)

    # Interface point 1 — std interval
    min_1 = torch.tensor(sp_coords_copy_test[1, 2], dtype=dtype, device=device) - torch.tensor(std, dtype=dtype, device=device)
    max_1 = torch.tensor(sp_coords_copy_test[1, 2], dtype=dtype, device=device) + torch.tensor(std, dtype=dtype, device=device)

    # Interface point 2 — std interval
    min_2 = torch.tensor(sp_coords_copy_test[2, 2], dtype=dtype, device=device) - torch.tensor(std, dtype=dtype, device=device)
    max_2 = torch.tensor(sp_coords_copy_test[2, 2], dtype=dtype, device=device) + torch.tensor(std, dtype=dtype, device=device)

    # Interface point 4 — 3*std interval (wider prior)
    min_4 = torch.tensor(sp_coords_copy_test[4, 2], dtype=dtype, device=device) - torch.tensor(3.0 * std, dtype=dtype, device=device)
    max_4 = torch.tensor(sp_coords_copy_test[4, 2], dtype=dtype, device=device) + torch.tensor(3.0 * std, dtype=dtype, device=device)

    test_list = [
        {
            "update": "interface_data",
            "id": torch.tensor([0]),
            "direction": "Z",
            "prior_distribution": "uniform",
            "uniform": {"min": min_0, "max": max_0}
        },
        {
            "update": "interface_data",
            "id": torch.tensor([1]),
            "direction": "Z",
            "prior_distribution": "uniform",
            "uniform": {"min": min_1, "max": max_1}
        },
        {
            "update": "interface_data",
            "id": torch.tensor([2]),
            "direction": "Z",
            "prior_distribution": "uniform",
            "uniform": {"min": min_2, "max": max_2}
        },
        {
            "update": "interface_data",
            "id": torch.tensor([4]),
            "direction": "Z",
            "prior_distribution": "uniform",
            "uniform": {"min": min_4, "max": max_4}
        },
    ]
    num_layers = len(test_list)

    Gempy = GempyModel(test_list, geo_model_test, num_layers, slope=slope, dtype=dtype, device=device)

    # ---- rank 0 draws samples, broadcast to all ----
    comm.Barrier()
    if rank == 0:
        c = Gempy.GenerateInputSamples(number_samples=number_samples)
        data["input"] = c.tolist()
    else:
        c = None
    c = comm.bcast(c, root=0)   # (N, num_params) on every rank

    # ---- each rank processes its own nodes ----
    # local_node_indices are positions within the full custom grid that belong
    # to this rank's partition of the mesh.
    
    local_node_indices = np.arange(len(global_indices))   # positions in local mesh order
    
    m_local, grad_local = Gempy.GenerateOutputSamples_local(Inputs_samples=c, local_node_indices=local_node_indices)
    # m_local    : (N, num_local_nodes)
    # grad_local : (N, num_local_nodes, num_params)

    # ---- gather on rank 0 ----
    local_results = [
        (int(global_indices[idx]), m_local[:, idx], grad_local[:, idx])
        for idx in range(len(global_indices))
    ]
    comm.Barrier()
    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        global_output   = np.zeros((c.shape[0], nodes))
        global_gradient = np.zeros((c.shape[0], nodes, num_layers))

        for result_list in all_results:
            for global_idx, output_, grad_ in result_list:
                global_output[:, global_idx]    = output_
                global_gradient[:, global_idx]  = grad_

        data["Gempy_output"]    = global_output.tolist()
        data["Jacobian_Gempy"]  = global_gradient.tolist()

    return data


def create_true_data(mesh, nodes, shift,  slope=200, filename=None):
    mesh_coordinates = mesh.coordinates()
    data = {}
    geo_model_test = create_initial_gempy_model(refinement=7, save=True)

    if mesh_coordinates.shape[1] == 2:
        xyz_coord = np.insert(mesh_coordinates, 1, 0, axis=1)
    else:
        xyz_coord = mesh_coordinates

    gp.set_custom_grid(geo_model_test.grid, xyz_coord=xyz_coord)
    geo_model_test.interpolation_options.mesh_extraction = False
    #gp.compute_model(geo_model_test)
    
    geo_model_test.interpolation_options.sigmoid_slope = slope
    gp.compute_model(geo_model_test)

    sp_coords_copy_test = geo_model_test.interpolation_input.surface_points.sp_coords.copy()
    # m_initial = geo_model_test.solutions.octrees_output[0].last_output_center.custom_grid_values
    
    dtype = torch.float64
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
        
    test_list = [
        {
            "update": "interface_data",
            "id": torch.tensor([0]),
            "direction": "Z",
        },
        {
            "update": "interface_data",
            "id": torch.tensor([1]),
            "direction": "Z",
        },
        {
            "update": "interface_data",
            "id": torch.tensor([2]),
            "direction": "Z",
        },
        {
            "update": "interface_data",
            "id": torch.tensor([4]),
            "direction": "Z",
        },
    ]
    num_layers = len(test_list)
    
    c = torch.tensor([sp_coords_copy_test[[test_list[0]["id"][0], test_list[1]["id"][0], test_list[2]["id"][0], test_list[3]["id"][0]], 2]], dtype=dtype, device=device)+\
        torch.tensor(shift, dtype=dtype, device=device)
    
    
    gempy_model =GempyModel(test_list, geo_model_test, num_layers, slope=slope, dtype=dtype, device=device)
    local_node_indices = np.arange(len(mesh_coordinates))
    
    m, grad = gempy_model.GenerateOutputSamples_local(Inputs_samples=c, local_node_indices=local_node_indices)
    
    
    return m[0], grad[0], sp_coords_copy_test, geo_model_test


def generate_final_model(geo_model, interpolation_input_, num_layers, posterior_data,
                         slope=200, filename='posterior_model.png', save=True):
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
    
    #posterior_data_ =[Phyiscal_Data[0,2], Phyiscal_Data[1,2], Phyiscal_Data[2,2], Phyiscal_Data[4,2]]
    print(Phyiscal_Data)
    
    geo_model_test = create_final_gempy_model(Phyiscal_Data, refinement=7, save=save, filename=filename)
    geo_model_test.interpolation_options.mesh_extraction = False
    sol = gp.compute_model(geo_model_test)
    geo_model_test.interpolation_options.sigmoid_slope = slope
    gp.compute_model(geo_model_test)
        


def generate_input_output_gempy_data_(mesh, nodes, number_samples, device, slope=200, filename=None):
    """Single-process (no MPI) version — unchanged logic, uses optimized grad."""
    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    torch.use_deterministic_algorithms(True)

    mesh_coordinates = mesh.coordinates()
    data = {}
    geo_model_test = create_initial_gempy_model(refinement=3, save=True)

    if mesh_coordinates.shape[1] == 2:
        xyz_coord = np.insert(mesh_coordinates, 1, 0, axis=1)
    else:
        xyz_coord = mesh_coordinates

    gp.set_custom_grid(geo_model_test.grid, xyz_coord=xyz_coord)
    geo_model_test.interpolation_options.mesh_extraction = False
    sp_coords_copy_test = geo_model_test.interpolation_input.surface_points.sp_coords.copy()

    geo_model_test.surface_points.df.to_csv("./Initial_sp.csv")
    geo_model_test.orientations.df.to_csv("./Initial_op.csv")

    ###############################################################################
    # Make a list of gempy parameter which would be treated as a random variable
    ###############################################################################
    dtype = torch.float64
    std = 0.02  # 4*std = gap between two layers

    # Interface point 0 — std interval
    min_0 = torch.tensor(sp_coords_copy_test[0, 2], dtype=dtype, device=device) - torch.tensor(std, dtype=dtype, device=device)
    max_0 = torch.tensor(sp_coords_copy_test[0, 2], dtype=dtype, device=device) + torch.tensor(std, dtype=dtype, device=device)

    # Interface point 1 — std interval
    min_1 = torch.tensor(sp_coords_copy_test[1, 2], dtype=dtype, device=device) - torch.tensor(std, dtype=dtype, device=device)
    max_1 = torch.tensor(sp_coords_copy_test[1, 2], dtype=dtype, device=device) + torch.tensor(std, dtype=dtype, device=device)

    # Interface point 2 — std interval
    min_2 = torch.tensor(sp_coords_copy_test[2, 2], dtype=dtype, device=device) - torch.tensor(std, dtype=dtype, device=device)
    max_2 = torch.tensor(sp_coords_copy_test[2, 2], dtype=dtype, device=device) + torch.tensor(std, dtype=dtype, device=device)

    # Interface point 4 — 3*std interval (wider prior)
    min_4 = torch.tensor(sp_coords_copy_test[4, 2], dtype=dtype, device=device) - torch.tensor(3.0 * std, dtype=dtype, device=device)
    max_4 = torch.tensor(sp_coords_copy_test[4, 2], dtype=dtype, device=device) + torch.tensor(3.0 * std, dtype=dtype, device=device)

    test_list = [
        {
            "update": "interface_data",
            "id": torch.tensor([0]),
            "direction": "Z",
            "prior_distribution": "uniform",
            "uniform": {"min": min_0, "max": max_0}
        },
        {
            "update": "interface_data",
            "id": torch.tensor([1]),
            "direction": "Z",
            "prior_distribution": "uniform",
            "uniform": {"min": min_1, "max": max_1}
        },
        {
            "update": "interface_data",
            "id": torch.tensor([2]),
            "direction": "Z",
            "prior_distribution": "uniform",
            "uniform": {"min": min_2, "max": max_2}
        },
        {
            "update": "interface_data",
            "id": torch.tensor([4]),
            "direction": "Z",
            "prior_distribution": "uniform",
            "uniform": {"min": min_4, "max": max_4}
        },
    ]
    num_layers = len(test_list)

    Gempy = GempyModel(test_list, geo_model_test, num_layers, slope=slope, dtype=dtype, device=device)
    c = Gempy.GenerateInputSamples(number_samples=number_samples)

    all_indices = np.arange(nodes)
    m_data, dmdc_data = Gempy.GenerateOutputSamples_local(c, all_indices)

    data["input"] = c
    data["Gempy_output"] = m_data
    data["Jacobian_Gempy"] = dmdc_data
    return data