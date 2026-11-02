import contextlib
import io
import json
import ufl
import dolfin as dl
from datetime import datetime
#sys.path.append(os.environ.get('HIPPYLIB_PATH', "../../"))
import hippylib as hp
#sys.path.append(os.environ.get('HIPPYFLOW_PATH',"../../"))
import hippyflow as hf

from helpers import *
from generate_samples2 import *
#from train_nn import *
import arviz as az
from pyro.infer import MCMC, NUTS, Predictive, EmpiricalMarginal
from pyro.infer.autoguide import init_to_mean, init_to_median, init_to_value
from pyro.infer.mcmc.util import summary

from gradient import compute_gradient
from train_nn import calculate_jacobian_full_fwd

from mpi4py import MPI

import pandas as pd
def pyro_model(interpolation_input_ ,num_layers, NN_model,Interpolation_matrix, u_shift, phi, obs_data, Sigma_l, beta, device ):
        
            """
            This Pyro model represents the probabilistic aspects of the geological model.
            It defines a prior distribution for the top layer's location and
            computes the thickness of the geological layer as an observed variable.

            
            interpolation_input_: represents the dictionary of random variables for surface parameters
            
            num_layers: represents the number of layers we want to include in the model
            
            """

            
            parameter = []
            
            # Create a random variable based on the provided dictionary used to modify input data of gempy
            counter=1
            for interpolation_input_data in interpolation_input_[:num_layers]:
                
                # Check if user wants to create random variable based on modifying the surface points of gempy
                if interpolation_input_data["update"]=="interface_data":
                    # Check what kind of distribution is needed
                    if interpolation_input_data["prior_distribution"]=="normal":
                        mean = interpolation_input_data["normal"]["mean"]
                        std  = interpolation_input_data["normal"]["std"]
                        parameter.append(pyro.sample("mu_"+ str(counter), dist.Normal(mean, std)).to(device))
                        
                    elif interpolation_input_data["prior_distribution"]=="uniform":
                        min = interpolation_input_data["uniform"]["min"]
                        max = interpolation_input_data["uniform"]["max"]
                        parameter.append(pyro.sample("mu_"+ str(interpolation_input_data['id']), dist.Uniform(min, max)).to(device))

                    elif interpolation_input_data["prior_distribution"]=="fixed":
                        parameter.append(pyro.sample("mu_"+ str(interpolation_input_data['id']), interpolation_input_data["fixed"]["value"]).to(device))
                    else:
                        print("We have to include the distribution")
                
                counter=counter+1
             
            input_data = torch.tensor(parameter, device=device)
            NN_model.eval()
            NN_output = NN_model(input_data)
            output =torch.matmul(Interpolation_matrix, torch.matmul(NN_output, phi.T) + u_shift)
            
            # mean = output
            # cov_matrix = Sigma_l
            # r = obs_data - mean
            
            # Sigma_r = torch.linalg.pinv(cov_matrix) @ r
            
            # r_T_Sigma_inv_r = torch.dot(r, Sigma_r)
            # log_likelihood = - 0.5 * r_T_Sigma_inv_r  
            
            # pyro.factor("log_likelihood",  log_likelihood)
            with pyro.plate("likelihood", obs_data.shape[0]):
                pyro.sample("obs", dist.Normal(output, beta), obs=obs_data).to(device)
            
            

def main():
    
    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    torch.use_deterministic_algorithms(True)
    
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
        
    
    nx =31; ny = 31
    nodes = (nx+1)*(ny+1)
    mesh = dl.RectangleMesh(dl.Point(0.0, 0.0), dl.Point(1.0, 1.0), nx, ny)
    
    
    Vh_STATE = dl.FunctionSpace(mesh, "CG", 2)
    Vh_PARAMETER = dl.FunctionSpace(mesh, "CG", 1)
    Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE]

    d2v = dl.dof_to_vertex_map(Vh[hp.PARAMETER])
    v2d = dl.vertex_to_dof_map(Vh[hp.PARAMETER])


    def u_boundary(x, on_boundary):
        return on_boundary and ( x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

    u_bdr = dl.Expression("x[1]", degree=1)
    u_bdr0 = dl.Constant(0.0)
    bc = dl.DirichletBC(Vh[hp.STATE], u_bdr, u_boundary)
    bc0 = dl.DirichletBC(Vh[hp.STATE], u_bdr0, u_boundary)

    f = dl.Constant(0.0)
    #f = dl.Expression("sin(2*pi*x[0]) * exp(x[1]+ x[0])", degree=5)

    def pde_varf(u,m,p):
        return m*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx - f*p*ufl.dx
        

    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)

    
    Borehole_location = list(np.linspace(0.05,0.95, 20))
    Borehole_extent = len(Borehole_location)*[0.05]
    Borehole_points = len(Borehole_location)*[20]
    
    target_list=[]
    for i, x in enumerate(Borehole_location):
        z_data = np.linspace(Borehole_extent[i], 0.95, Borehole_points[i])
        first_column = np.full((z_data.shape[0],), x)
        two_d_array = np.column_stack((first_column, z_data))
        target_list.append(two_d_array)
        
    targets = np.vstack(target_list)
    
    # targets = mesh.coordinates()
    
    B = hp.assemblePointwiseObservation(Vh[hp.STATE], targets)
    B_pesudo_inverse = np.linalg.pinv(B.array())
    
    m_trial = dl.TrialFunction(Vh[hp.PARAMETER])
    m_test = dl.TestFunction(Vh[hp.PARAMETER])
    u_trial = dl.TrialFunction(Vh[hp.STATE])
    u_test = dl.TestFunction(Vh[hp.STATE])

    M_U = dl.assemble(dl.inner(u_trial,u_test)*dl.dx)

    I_U = hf.StateSpaceIdentityOperator(M_U)
    #observable = hf.LinearStateObservable(pde,M_U)
    observable_1 = hf.LinearStateObservable(pde,I_U)
    observable_2 = hf.LinearStateObservable(pde,B)

    Jm_1 = hf.ObservableJacobian(observable_1)
    Jm_2 = hf.ObservableJacobian(observable_2)


    M_M = dl.assemble(dl.inner(m_trial,m_test)*dl.dx)
    # Noise level
    noise_level = 0.01
    
    #shift = np.array([0,0,0,-0.06]) #0.15
    #shift = np.array([0,0,0,-0.08333333 * 0.5]) #0.2
    #shift = np.array([0,0,0,0]) #0.3
    #shift = np.array([0,0,0,0.08333333 * 0.5]) #0.4
    shift = np.array([0,0,0,0.06]) # 0.444
    
    m_initial, dmdk, sp_coords_copy_test, gempy_model= create_true_data(mesh=mesh, shift=shift, nodes=nodes, filename=None)
    
    m_initial = 2 * m_initial
    dmdk = 2 * dmdk
    
    m = dl.Function(Vh[hp.PARAMETER])
    m.vector().set_local(m_initial[d2v])
    
    fig, ax = plt.subplots(figsize=(6,6))  # Create new figure explicitly
    im =plt.imshow(m_initial.reshape((nx+1,nx+1)),
                   cmap='viridis',
                   origin='lower',
                   interpolation='bilinear',
                   extent=[0, 1, 0, 1])
    # Add colorbar to this figure
    fig.colorbar(im, ax=ax)
    # Add axis labels and title
    ax.set_xlabel("x-coordinate")
    ax.set_ylabel("z-coordinate")
    ax.set_title("Parameter Field $m(x,z)$")
    plt.tight_layout()
    plt.show()
    
    '''
    for i in range(dmdk.shape[1]):
        # dm = dl.Function(Vh[hp.PARAMETER])
        # dm.vector().set_local(dmdk[:, i][d2v])
        
        fig, ax = plt.subplots(figsize=(6,6))  # Create new figure explicitly
        im =plt.imshow(dmdk[:,i].reshape((nx+1,nx+1)),
                    cmap='viridis',
                    origin='lower',
                    interpolation='bilinear',
                    extent=[0, 1, 0, 1])
        # Add colorbar to this figure
        fig.colorbar(im, ax=ax)
        # Add axis labels and title
        ax.set_xlabel("x-coordinate")
        ax.set_ylabel("z-coordinate")
        ax.set_title(r"Jacobian field $ \mu_{i} :D_k m(x,z)$")
        plt.tight_layout()
        plt.show()
    '''
        
        
    u = dl.Function(Vh[hp.STATE])
    uadj = dl.Function(Vh[hp.ADJOINT])
    uhat = dl.Function(Vh[hp.STATE])
    m_nodes_hat = dl.Function(Vh[hp.PARAMETER])
    dU = u.vector().get_local().shape[0]
    dO = targets.shape[0]
    u_data = np.zeros(dU)
    u_obs  = np.zeros(dO)
    Jacobian_true_1 = np.zeros((dU, dmdk.shape[1]))
    Jacobian_true_2 = np.zeros((dO, dmdk.shape[1]))
    
    m.vector().set_local(m_initial[d2v]) #uncomment this
    x = [u.vector(),m.vector(),uadj.vector()]
    pde.solveFwd(x[hp.STATE], x)
    
    # Get the data
    Jm_1.observable.setLinearizationPoint(x)
    
    u_true = x[hp.STATE].get_local()
    
    u_obs_1 = observable_1.evalu(x[hp.STATE]).get_local()
    
    
    for component in range(dmdk.shape[1]):
        m_nodes_hat.vector().set_local(dmdk[:,component][d2v]) 
        Jm_1.mult(m_nodes_hat.vector(), uhat.vector()) 
        # Multiply dmdc by Mass matrix.
        # M_M.mult(m_nodes_hat.vector(), m_hat.vector()) 
        # Jm.mult(m_hat.vector(),uhat.vector()) 
        Jacobian_true_1[:,component] = uhat.vector().get_local()
    u_ob_2  = observable_2.evalu(x[hp.STATE]).get_local()
    
    u_obs_2 = observable_2.evalu(x[hp.STATE]).get_local()
    Jm_2.observable.setLinearizationPoint(x)
    Jhelp3 = dl.Vector()
    for component in range(dmdk.shape[1]):
        m_nodes_hat.vector().set_local(dmdk[:,component][d2v])   
        Jm_2.mult(m_nodes_hat.vector(), Jhelp3) 
        Jacobian_true_2[:,component]= Jhelp3.get_local()
    
    
    u_true = u_obs_1
    u_obs = u_obs_2
    J_true = Jacobian_true_1
    J_obs  = Jacobian_true_2
    
    u_data = B.array() @ u_true
    A = u_obs - u_data
    #A = u_obs - u_true
    A[np.abs(A)<1e-10]=0
    print(np.sum(A))
    
    # plot u_true
    u.vector().set_local(u_true)
    
    print("B_pseudo_inverse.shape:", B_pesudo_inverse.shape)
    print("M_U.shape:", M_U.array().shape)
    M_obs = B_pesudo_inverse.T @ M_U.array() @ B_pesudo_inverse
    print("M_obs.shape:", M_obs.shape)
    
    #M_obs = M_U.array()
    
    # noise level


    ud = dl.Function(Vh[hp.STATE])
    ud.assign(u)

    # perturb state solution and create synthetic measurements ud
    # ud = u + ||u||/SNR * random.normal
    MAX = ud.vector().norm("linf")

    # Create noise vector and insert the numpy array
    noise = dl.Vector()
    M_U.init_vector(noise, 1)
    hp.parRandom.normal(noise_level * MAX, noise)

    bc0.apply(noise)

    ud.vector().axpy(1., noise)

    # plot
    hp.nb.multi1_plot([u, ud], ["State Field", "Synthetic observations"])
    # Add axis labels and title
    plt.scatter(targets[:,0], targets[:,1], color='white', marker='s', s=10, label='Observations')
    plt.legend()
    plt.xlabel("x-coordinate")
    plt.ylabel("z-coordinate")
    plt.title("Synthetic observations with measurement points")
    plt.show()
    
    # Generate Gaussian noise
    mean = 0
    std_dev = noise_level
    
    u_M_norm = np.dot(u_obs, M_obs @ u_obs)
    u_norm = np.dot(u_obs, u_obs)
    
    u_uT = np.outer(u_obs, u_obs)
    
    print("u_uT.shape:", u_uT.shape)
    print("u_norm:", u_norm)
    print("u_M_norm.shape:", u_M_norm)
    
    trace= np.trace(M_obs)
    print("trace of M_obs:", trace)
    
    beta =  std_dev #* np.sqrt(u_M_norm /M_obs.shape[0]) 
    #alpha = 0.05 #* np.sqrt(u_M_norm /trace) 
    # print("eigenvalues of Sigma_l:", np.linalg.eigvals(Sigma_l))
    Sigma_l = (beta**2) * np.eye(M_obs.shape[0])
    #Sigma_l = beta**2 *  np.linalg.pinv(M_obs)
    
    u_noisy = np.random.multivariate_normal(
    mean=u_obs,
    cov=Sigma_l)
    #u_obs =u_noisy
    print("||del u||:", np.linalg.norm(u_obs- u_noisy) / np.linalg.norm(u_obs))
    #exit()
    #Sigma_l = alpha**2 *  np.linalg.pinv(M_obs) * (u_M_norm /M_obs.shape[0])
    u_obs = u_noisy
    Sigma_l = torch.tensor(Sigma_l, dtype=torch.float32, device=device)
    
    dtype =torch.float32
    Mphi_r= torch.tensor(np.load('./saved_model/Mphi_r.npy'), dtype=dtype, device=device)
    phi_r = torch.tensor(np.load('./saved_model/phi_r.npy'), dtype=dtype, device=device)
    u_shift = torch.tensor(np.load('./saved_model/u_shift.npy'), dtype=dtype, device=device)
    
    
    
    Interpolation_matrix = torch.tensor(B.array(), dtype=dtype, device=device)
    
    #Interpolation_matrix = torch.tensor(np.eye(u_true.shape[0]), dtype=dtype, device=device)
    
    
    model_jacobian_full = torch.load("./saved_model/model_jacobian_full_8192_250.pth",map_location=device, weights_only=False)
    #model_jacobian_full = torch.load("./saved_model/model_without_jacobian_8192_500.pth",map_location=device, weights_only=False)
    # model_jacobian_full = torch.load("./saved_model/model_jacobian_full_8192_500.pth",map_location=device, weights_only=False)
  
    #compute_gradient(neural_network = model_jacobian_full,Decoder_matrix=phi_r,Encoder_matrix= Mphi_r, bias=u_shift, d = u_obs, interpolation_matrix=Interpolation_matrix, Noise_cov=Sigma_l,Sigma_p=Sigma_l )
    
    ###############################################################################
    # Make a list of gempy parameter which would be treated as a random variable
    ###############################################################################
    print(sp_coords_copy_test)
    
    test_list=[]
    
    std = 0.02  # 4*std = gap between two layers

    # Interface point 0
    mean_0 = torch.tensor(sp_coords_copy_test[0, 2], dtype=dtype, device=device)

    # Interface point 1
    mean_1 = torch.tensor(sp_coords_copy_test[1, 2], dtype=dtype, device=device)

    # Interface point 2
    mean_2 = torch.tensor(sp_coords_copy_test[2, 2], dtype=dtype, device=device)

    # Interface point 4 — wider prior (3*std)
    mean_4 = torch.tensor(sp_coords_copy_test[4, 2], dtype=dtype, device=device) 
    
    std_1  = torch.tensor(std,        dtype=dtype, device=device)
    std_4  = torch.tensor(3.0 * std,  dtype=dtype, device=device)
    print("\n")
    print("Deterministic inversion")
    print("\n") 
    mu_0 = torch.tensor([mean_0, mean_1, mean_2, mean_4] + shift, dtype=dtype, device=device, requires_grad= True)
    
    Sigma_p = torch.diag(torch.tensor([std_1, std_1, std_1, std_4], dtype=dtype, device = device)**2)

    test_list = [
        {
            "update": "interface_data",
            "id": torch.tensor([0]),
            "direction": "Z",
            "prior_distribution": "normal",
            "normal": {"mean": mu_0[0].detach(), "std": std_1}
        },
        {
            "update": "interface_data",
            "id": torch.tensor([1]),
            "direction": "Z",
            "prior_distribution": "normal",
            "normal": {"mean": mu_0[1].detach(), "std": std_1}
        },
        {
            "update": "interface_data",
            "id": torch.tensor([2]),
            "direction": "Z",
            "prior_distribution": "normal",
            "normal": {"mean": mu_0[2].detach(), "std": std_1}
        },
        {
            "update": "interface_data",
            "id": torch.tensor([4]),
            "direction": "Z",
            "prior_distribution": "normal",
            "normal": {"mean": mu_0[3].detach(), "std": std_4}
        },
    ]
    num_layers = len(test_list)
    
    
    
    obs_data = torch.tensor(u_obs, dtype=dtype, device=device)
    
    ###################
    
    
    compute_gradient(neural_network = model_jacobian_full,Decoder_matrix=phi_r,Encoder_matrix= Mphi_r, bias=u_shift, d = u_obs, interpolation_matrix=Interpolation_matrix, Noise_cov=beta, mean_p=mu_0, Sigma_p=Sigma_p )
    
    print("############################################################################")
    print(" Data generation range for the model parameters")

    print("############################################################################")
    print("\n")
    generate_final_model(geo_model=gempy_model, interpolation_input_=test_list, num_layers=num_layers, posterior_data=list(torch.tensor(mu_0, dtype=torch.float64)), slope=200, filename='posterior_model.png')
    generate_final_model(geo_model=gempy_model, interpolation_input_=test_list, num_layers=num_layers, posterior_data=list(torch.tensor(mu_0 + torch.sqrt(torch.diagonal(Sigma_p)), dtype=torch.float64)), slope=200, filename='posterior_model_plus_std.png')
    generate_final_model(geo_model=gempy_model, interpolation_input_=test_list, num_layers=num_layers, posterior_data=list(torch.tensor(mu_0 - torch.sqrt(torch.diagonal(Sigma_p)), dtype=torch.float64)), slope=200, filename='posterior_model_minus_std.png')
    print("\n")
    
    print("############################################################################")
    
    model = model_jacobian_full.to(device)
    model.eval()
    NN_output = model(mu_0)
    #print("NN_output:", NN_output)
    Jacobian_NN = calculate_jacobian_full_fwd(inputs= mu_0.unsqueeze(0), model=model)
   
    Jacobian_NN = Jacobian_NN.squeeze(0)
    
    
    output =torch.matmul(Interpolation_matrix, torch.matmul(NN_output.detach(), phi_r.T) + u_shift)
    Jacobian = torch.matmul(Interpolation_matrix, torch.matmul(phi_r, Jacobian_NN.detach()))
    
    output_mean =torch.matmul(Interpolation_matrix, u_shift)
    print("output mean magnitude", torch.norm(output_mean))
    print(" magnitude ",torch.norm(output))
    
    print("Jacobian infromation \n")
    rel_acc_true_U = 1 - torch.norm(torch.matmul(NN_output.detach(), phi_r.T) + u_shift - torch.tensor(u_obs_1, dtype=dtype, device= device)) / torch.norm(torch.tensor(u_obs_1, dtype=dtype, device= device))
    print("Relative Function  between true and surrogate using all the node \n",  rel_acc_true_U )
    
    rel_acc_true_U = 1 - torch.norm(output - torch.tensor(u_obs_2, dtype=dtype, device= device)) / torch.norm(torch.tensor(u_obs_2, dtype=dtype, device= device))
    print("Relative Function accuracy between true and surrogate using obs \n",  rel_acc_true_U )
    error_magnitude = torch.norm(output - torch.tensor(u_obs_2, dtype=dtype, device= device)) 
    print("Function error between true and surrogate using obs \n",  error_magnitude )
    
    rel_acc_true_J = 1 - torch.norm(torch.matmul(phi_r, Jacobian_NN.detach()) - torch.tensor(Jacobian_true_1, dtype=dtype, device= device), p='fro') / torch.norm(torch.tensor(Jacobian_true_1, dtype=dtype, device= device), p='fro')
    print("Relative Jacobian accuracy between true and surrogate using all the node \n",  rel_acc_true_J )
    
    rel_acc_true_J_obs = 1- torch.norm(Jacobian - torch.tensor(Jacobian_true_2, dtype=dtype, device= device), p='fro') / torch.norm(torch.tensor(Jacobian_true_2, dtype=dtype, device= device), p='fro')
    print( "Relative Jacobian accuracy between true and surrogate using all the obs\n", rel_acc_true_J_obs)
    
    
   
    # exit()
    #Jacobian = torch.tensor(J_obs, dtype=dtype, device = device)
    # A = Jacobian.T @ torch.linalg.inv(Sigma_l) @ Jacobian
    A = Jacobian.T @ Jacobian
    U, S, Vh = torch.linalg.svd(Jacobian , full_matrices=True)
    _, S_true, _ = torch.linalg.svd(torch.tensor(Jacobian_true_2, dtype=dtype, device= device), full_matrices=True)

    print("U shape:", U.shape)    # (m, m)
    print("S:", S)    # (min(m, n),)
    print("S_true", S_true)
    print("Vh shape:", Vh.shape)  # (n, n)
    
    
    
    beta =  beta
    output2 = torch.tensor(u_obs_2, dtype=dtype)
    #Sigma_post = torch.linalg.inv( torch.linalg.inv(Sigma_p) + Jacobian.T @ torch.linalg.inv(Sigma_l) @ Jacobian)
    Sigma_post = torch.linalg.inv( torch.linalg.inv(Sigma_p) + (Jacobian.T  @ Jacobian) / (beta**2) )
    #mu_post  = mu_0 + Sigma_post @ Jacobian.T @ torch.linalg.inv(Sigma_l) @ (obs_data - output)
    mu_post  = mu_0 + (Sigma_post @ Jacobian.T  @ (obs_data - output))/ (beta**2)
    #mu_post  = mu_0 + (Sigma_post @ Jacobian.T  @ (obs_data - output2))/ (beta**2)
    
    print("\n Posterior Covariance")
    print(Sigma_post)
    
    print("Eigen values of Covariance of posterior")
    print(torch.linalg.eigvals(Sigma_post))
    
    print("Prior mean:")
    print(mu_0)
    print("Prior std:")
    print(torch.sqrt(torch.diagonal(Sigma_p)))
    print("Posterior mean:")
    print(mu_post)
    print("Posterior std:")
    std_post = torch.sqrt(torch.diagonal(Sigma_post))
    print(std_post)
    
    
    print("############################################################################")
    print(" Data generation range for the model parameters using Gaussian approximation")
    print("############################################################################")
    print("\n")
    generate_final_model(geo_model=gempy_model, interpolation_input_=test_list, num_layers=num_layers, posterior_data=list(torch.tensor(mu_post, dtype=torch.float64)), slope=200, filename='posterior_model.png')
    generate_final_model(geo_model=gempy_model, interpolation_input_=test_list, num_layers=num_layers, posterior_data=list(torch.tensor(mu_post + std_post, dtype=torch.float64)), slope=200, filename='posterior_model_plus_std.png')
    generate_final_model(geo_model=gempy_model, interpolation_input_=test_list, num_layers=num_layers, posterior_data=list(torch.tensor(mu_post - std_post, dtype=torch.float64)), slope=200, filename='posterior_model_minus_std.png')
    print("\n")
    print("############################################################################")

    
    
    #print(obs_data.shape)
    dot = pyro.render_model(pyro_model, model_args=(test_list, num_layers, model, Interpolation_matrix,  u_shift, phi_r, obs_data ,Sigma_l, beta,  device),render_distributions=True)
    #pyro.set_rng_seed(42)

    prior = Predictive(pyro_model,num_samples=10000)(test_list, num_layers, model, Interpolation_matrix,   u_shift, phi_r, obs_data , Sigma_l, beta, device)
    #plt.figure(figsize=(8,10))
    data = az.from_pyro(prior=prior)
    #az.plot_trace(data.prior)
    
    ################################################################################
    # Posterior
    ################################################################################
    pyro.primitives.enable_validation(is_validate=True)
    nuts_kernel = NUTS(pyro_model, step_size=0.0085, adapt_step_size=True, target_accept_prob=0.7, max_tree_depth=10, init_strategy=init_to_mean)
    mcmc = MCMC(nuts_kernel, num_samples=4000, mp_context="spawn", warmup_steps=3000,num_chains=5, disable_validation=True)
    mcmc.run(test_list, num_layers, model,Interpolation_matrix,  u_shift, phi_r, obs_data ,Sigma_l, beta, device)
    posterior_samples = mcmc.get_samples(group_by_chain=False)
    samples = mcmc.get_samples(group_by_chain=True)
    
    summary_stats = summary(samples)
    #print(summary_stats) 
    # Convert to a Pandas DataFrame
    df = pd.DataFrame.from_dict(summary_stats, orient="index")
    # This transposes the dictionary so each parameter is a row

    # Now save to CSV
    df.to_csv("pyro_summary.csv")
    
    print("MCMC summary results")
            
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        mcmc.summary()  
        summary_output = buf.getvalue()

    
    print(summary_output)
    
    with open('./mcmc_summary_p.txt', 'w') as f:
                f.write(summary_output)
    
    list_parameter_mean = []
    list_parameter_mean_plus_std = []
    list_parameter_mean_minus_std = []
    for index, (key, values) in enumerate(posterior_samples.items()):
        print("Prior mean: ", test_list[index]["normal"]["mean"], "Prior std: ", test_list[index]["normal"]["std"])
        print("Posterior mean: ",torch.mean(values), "Posterior std: ",torch.std(values))
        list_parameter_mean.append(torch.mean(values).to(torch.float64))
        list_parameter_mean_plus_std.append(torch.mean(values).to(torch.float64) + torch.std(values).to(torch.float64))
        list_parameter_mean_minus_std.append(torch.mean(values).to(torch.float64) - torch.std(values).to(torch.float64))
    
    posterior_predictive = Predictive(pyro_model, posterior_samples)(test_list, num_layers, model, Interpolation_matrix,  u_shift, phi_r, obs_data ,Sigma_l, beta, device)
    plt.figure(figsize=(8,10))
    data = az.from_pyro(posterior=mcmc, prior=prior, posterior_predictive=posterior_predictive)
    az.plot_trace(data)
    filename_posteriro_plot = "./saved_model/posterior.png"
    plt.savefig(filename_posteriro_plot)
    plt.close()
    ###############################################TODO################################
    # Plot and save the file for each parameter
    ###################################################################################
    for i in range(len(test_list)):
        plt.figure(figsize=(8,10))
        az.plot_density(
        data=[data.posterior, data.prior],
        shade=.9,
        bw=0.006,  # increase bandwidth for smoother curve
        var_names=['mu_' +str(i+1)],
        data_labels=["Posterior Predictive", "Prior Predictive"],
        )
        plt.savefig("./saved_model/mu_"+ str(i)+".png")
        plt.close()
        
    
    print("############################################################################")
    print(" Data generation range for the model parameters using Pyro posterior samples")
    print("############################################################################")
    print("\n")
    generate_final_model(geo_model=gempy_model, interpolation_input_=test_list, num_layers=num_layers, posterior_data=list_parameter_mean, slope=200, filename='posterior_model.png')
    generate_final_model(geo_model=gempy_model, interpolation_input_=test_list, num_layers=num_layers, posterior_data=list_parameter_mean_plus_std, slope=200, filename='posterior_model_plus_std.png')
    generate_final_model(geo_model=gempy_model, interpolation_input_=test_list, num_layers=num_layers, posterior_data=list_parameter_mean_minus_std, slope=200, filename='posterior_model_minus_std.png')
    print("\n")
    print("############################################################################")
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
