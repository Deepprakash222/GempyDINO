"""
Bayesian inversion of the SD3LMP model with a trained neural-network surrogate.

Simplified version of the original inversion script. Every change is tagged
with a comment of the form   # [MOD n] ...   and explained in the summary:

  [MOD 1]  BUG FIX: torch.tensor(list) -> torch.stack(list) in pyro_model
  [MOD 2]  pyro_model simplified (only the "normal" prior is used)
  [MOD 3]  All experiment settings collected in one CONFIG block
  [MOD 4]  Setup split into small functions (mesh/PDE, targets, true data, noise)
  [MOD 5]  Duplicate / unused computations and debug prints removed
  [MOD 6]  Prior list built in a loop instead of four copy-pasted dicts
  [MOD 7]  GLA written as its own function
  [MOD 8]  Helper for the three geological plots (mean, +std, -std)
  [MOD 9]  Output files get a prefix, so prior/GLA/NUTS plots no longer overwrite each other
  [MOD 10] Unused imports removed
"""

import contextlib
import io
import random
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import ufl
import dolfin as dl
import hippylib as hp
import hippyflow as hf
import arviz as az
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive
from pyro.infer.autoguide import init_to_mean
from pyro.infer.mcmc.util import summary

from generate_samples2 import create_true_data, generate_final_model
from gradient import compute_gradient
from train_nn import calculate_jacobian_full_fwd
# [MOD 10] removed unused imports: json, MPI, EmpiricalMarginal, init_to_median,
#          init_to_value, and the wildcard `from helpers import *`
#          (re-add `from helpers import *` if generate_samples2 relies on it).


# =============================================================================
# [MOD 3] CONFIG: everything you change between runs lives here
# =============================================================================
SEED = 2025
DTYPE = torch.float32
NX = NY = 31

NOISE_LEVEL = 0.01                      # 0.01 or 0.05
SHIFT = np.array([0, 0, 0, -0.06])      # prior mean of mu_4 = 0.156
#SHIFT = np.array([0, 0, 0, -0.08333333 * 0.5])  # 0.2
# SHIFT = np.array([0, 0, 0, 0])                  # 0.3
# SHIFT = np.array([0, 0, 0,  0.08333333 * 0.5])  # 0.4
# SHIFT = np.array([0, 0, 0,  0.06])              # 0.444

PRIOR_TYPE = "normal"                   # "normal" or "uniform" (same mean and std)
PRIOR_STD = 0.02                        # mu_1..mu_3; mu_4 uses 3 * PRIOR_STD
INTERFACE_IDS = [0, 1, 2, 4]            # GemPy surface points used as parameters

BOREHOLE_X = [0.1, 0.3, 0.5, 0.7, 0.9]
BOREHOLE_TOP = [0.35, 0.5, 0.7, 0.5, 0.35]
BOREHOLE_POINTS = [100, 75, 50, 75, 100]

MODEL_DIR = "./saved_model"
NN_FILE = f"{MODEL_DIR}/model_jacobian_full_8192_250.pth"

NUTS_SETTINGS = dict(step_size=0.0085, adapt_step_size=True, target_accept_prob=0.7,
                     full_mass=True, max_tree_depth=10, init_strategy=init_to_mean)
MCMC_SETTINGS = dict(num_samples=500, warmup_steps=500, num_chains=5,
                     mp_context="spawn", disable_validation=True)


# =============================================================================
# Pyro model
# =============================================================================
def prior_distribution(p):
    """Return the Pyro distribution for one prior entry."""
    # [MOD 2] One place that maps the prior type to a distribution.
    #         Add new prior types here (e.g. "lognormal") without touching pyro_model.
    kind = p["prior_distribution"]
    if kind == "normal":
        return dist.Normal(p["normal"]["mean"], p["normal"]["std"])
    if kind == "uniform":
        return dist.Uniform(p["uniform"]["min"], p["uniform"]["max"])
    raise ValueError(f"Unknown prior_distribution '{kind}' (use 'normal' or 'uniform')")


def pyro_model(priors, NN_model, B, u_shift, phi, obs_data, beta):
    # [MOD 2] Normal and uniform priors are both supported via prior_distribution().
    #         Original issue fixed: the normal branch named sites by position
    #         (mu_1..mu_4) but the uniform branch by GemPy id (mu_0, mu_1, mu_2,
    #         mu_4), so switching prior type changed the names and broke the
    #         later plots. Names now always follow the position: mu_1..mu_n.
    #         Unknown types raise an error instead of being silently skipped.
    #         Unused arguments num_layers, Sigma_l and device are dropped.
    parameter = [pyro.sample(f"mu_{i + 1}", prior_distribution(p))
                 for i, p in enumerate(priors)]

    # [MOD 1] BUG FIX. Original: input_data = torch.tensor(parameter)
    #         torch.tensor() copies the values and DETACHES them from autograd,
    #         so NUTS saw no gradient from the likelihood (only from the prior).
    #         The sampler stays valid because the accept/reject step uses the
    #         full energy, but it is very inefficient: tiny step sizes, long
    #         trees, low n_eff and long run times. torch.stack keeps gradients.
    input_data = torch.stack(parameter)

    output = B @ (NN_model(input_data) @ phi.T + u_shift)
    with pyro.plate("likelihood", obs_data.shape[0]):
        pyro.sample("obs", dist.Normal(output, beta), obs=obs_data)


# =============================================================================
# [MOD 4] Setup functions
# =============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    torch.use_deterministic_algorithms(True)


def build_pde():
    mesh = dl.RectangleMesh(dl.Point(0.0, 0.0), dl.Point(1.0, 1.0), NX, NY)
    Vh_state = dl.FunctionSpace(mesh, "CG", 2)
    Vh_param = dl.FunctionSpace(mesh, "CG", 1)
    Vh = [Vh_state, Vh_param, Vh_state]

    def u_boundary(x, on_boundary):
        return on_boundary and (x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

    bc = dl.DirichletBC(Vh_state, dl.Expression("x[1]", degree=1), u_boundary)
    bc0 = dl.DirichletBC(Vh_state, dl.Constant(0.0), u_boundary)
    f = dl.Constant(0.0)

    def pde_varf(u, m, p):
        return m * ufl.inner(ufl.grad(u), ufl.grad(p)) * ufl.dx - f * p * ufl.dx

    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)
    u_trial, u_test = dl.TrialFunction(Vh_state), dl.TestFunction(Vh_state)
    M_U = dl.assemble(dl.inner(u_trial, u_test) * dl.dx)
    return mesh, Vh, pde, M_U


def borehole_targets():
    # Swap this function for a 20 x 20 grid to get the distributed acquisition.
    return np.vstack([
        np.column_stack((np.full(n, x), np.linspace(top, 0.9, n)))
        for x, top, n in zip(BOREHOLE_X, BOREHOLE_TOP, BOREHOLE_POINTS)
    ])


def true_solution(pde, Vh, observable, m_initial, dmdk, d2v):
    """Forward solve at the true parameter; returns observation and its Jacobian."""
    # [MOD 4] Replaces two near-identical loops (full state and observations).
    u = dl.Function(Vh[hp.STATE])
    m = dl.Function(Vh[hp.PARAMETER])
    p = dl.Function(Vh[hp.ADJOINT])
    dm = dl.Function(Vh[hp.PARAMETER])

    m.vector().set_local(m_initial[d2v])
    x = [u.vector(), m.vector(), p.vector()]
    pde.solveFwd(x[hp.STATE], x)

    Jm = hf.ObservableJacobian(observable)
    Jm.observable.setLinearizationPoint(x)
    u_obs = observable.evalu(x[hp.STATE]).get_local()

    # Output vector sized to the observable's output space (dim 0):
    # dU entries for the full-state observable, n_receivers for the boreholes.
    out = dl.Vector()
    observable.init_vector(out, 0)

    # Jacobian shape: (number of observations) x (number of parameters)
    Jacobian = np.zeros((u_obs.shape[0], dmdk.shape[1]))
    for k in range(dmdk.shape[1]):
        dm.vector().set_local(dmdk[:, k][d2v])
        Jm.mult(dm.vector(), out)
        Jacobian[:, k] = out.get_local()

    return u, u_obs, Jacobian


def plot_parameter_field(m_initial):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(m_initial.reshape((NX + 1, NY + 1)), cmap="viridis", origin="lower",
                   interpolation="bilinear", extent=[0, 1, 0, 1])
    fig.colorbar(im, ax=ax)
    ax.set(xlabel="x-coordinate", ylabel="z-coordinate", title="Parameter field $m(x,z)$")
    plt.tight_layout()
    plt.show()


def plot_state_with_targets(u, targets):
    # [MOD 5] The original plotted `ud = u + noise` with a different noise model
    #         (relative to max|u|, via hp.parRandom) than the one actually used
    #         in the inversion. It is replaced by the true state and the receivers.
    hp.nb.plot(u)
    plt.scatter(targets[:, 0], targets[:, 1], color="white", marker="s", s=10,
                label="Receivers")
    plt.legend()
    plt.xlabel("x-coordinate")
    plt.ylabel("z-coordinate")
    plt.title("State field with receiver locations")
    plt.show()


# =============================================================================
# [MOD 8] one helper instead of 4 x 3 copy-pasted generate_final_model calls
# =============================================================================
def plot_geology(gempy_model, priors, mean, std, prefix):
    for suffix, values in [("", mean), ("_plus_std", mean + std), ("_minus_std", mean - std)]:
        generate_final_model(geo_model=gempy_model, interpolation_input_=priors,
                             num_layers=len(priors),
                             posterior_data=list(values.detach().to(torch.float64)),
                             slope=200,
                             # [MOD 9] prefix added; original reused the same three
                             # file names for prior, GLA and NUTS (overwriting them)
                             filename=f"{prefix}_posterior_model{suffix}.png")


# =============================================================================
# [MOD 7] Gaussian linear approximation as its own function
# =============================================================================
def gla(model, mu_0, Sigma_p, B, phi_r, u_shift, obs_data, beta):
    NN_out = model(mu_0).detach()
    J_NN = calculate_jacobian_full_fwd(inputs=mu_0.unsqueeze(0), model=model).squeeze(0).detach()
    output = B @ (NN_out @ phi_r.T + u_shift)
    J = B @ (phi_r @ J_NN)
    Sigma_post = torch.linalg.inv(torch.linalg.inv(Sigma_p) + J.T @ J / beta**2)
    mu_post = mu_0 + Sigma_post @ J.T @ (obs_data - output) / beta**2
    return mu_post.detach(), Sigma_post.detach(), output, J, NN_out, J_NN


def rel_acc(approx, true):
    return (1 - torch.linalg.norm(approx - true) / torch.linalg.norm(true)).item()


# =============================================================================
# Main
# =============================================================================
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # [MOD 5] MPS branch removed: it printed "Using MPS" but always used the CPU.
    print(f"Using {device}")
    to_t = lambda a: torch.as_tensor(np.asarray(a), dtype=DTYPE, device=device)

    # ---- forward problem and receivers -------------------------------------
    mesh, Vh, pde, M_U = build_pde()
    d2v = dl.dof_to_vertex_map(Vh[hp.PARAMETER])
    targets = borehole_targets()
    B_dl = hp.assemblePointwiseObservation(Vh[hp.STATE], targets)

    # ---- true model (GemPy) -------------------------------------------------
    m_initial, dmdk, sp_coords, gempy_model = create_true_data(
        mesh=mesh, shift=SHIFT, nodes=(NX + 1) * (NY + 1), filename=None)
    m_initial, dmdk = 2 * m_initial, 2 * dmdk
    plot_parameter_field(m_initial)

    # ---- true state and Jacobians (full field and at receivers) ------------
    u, u_full, J_full = true_solution(pde, Vh, hf.LinearStateObservable(pde, hf.StateSpaceIdentityOperator(M_U)),
                                      m_initial, dmdk, d2v)
    _, u_obs_true, J_obs = true_solution(pde, Vh, hf.LinearStateObservable(pde, B_dl),
                                         m_initial, dmdk, d2v)
    # [MOD 5] removed: u_data/A consistency check, M_obs, u_uT, trace and the
    #         B pseudo-inverse (only used for printed diagnostics)
    plot_state_with_targets(u, targets)

    # ---- synthetic noisy data ----------------------------------------------
    beta = NOISE_LEVEL
    u_obs = np.random.multivariate_normal(mean=u_obs_true,
                                          cov=beta**2 * np.eye(len(u_obs_true)))
    print("relative noise ||u_noisy - u||/||u||:",
          np.linalg.norm(u_obs - u_obs_true) / np.linalg.norm(u_obs_true))
    obs_data = to_t(u_obs)

    # ---- surrogate ----------------------------------------------------------
    Mphi_r = to_t(np.load(f"{MODEL_DIR}/Mphi_r.npy"))
    phi_r = to_t(np.load(f"{MODEL_DIR}/phi_r.npy"))
    u_shift = to_t(np.load(f"{MODEL_DIR}/u_shift.npy"))
    B = to_t(B_dl.array())
    model = torch.load(NN_FILE, map_location=device, weights_only=False).to(device)
    model.eval()  # [MOD 2] called once here instead of inside every pyro_model call

    # ---- prior --------------------------------------------------------------
    # [MOD 6] built in a loop; replaces mean_0..mean_4 and four hand-written dicts
    mu_0 = (to_t(sp_coords[INTERFACE_IDS, 2]) + to_t(SHIFT)).requires_grad_(True)
    prior_std = to_t([PRIOR_STD] * 3 + [3 * PRIOR_STD])
    Sigma_p = torch.diag(prior_std**2)
    priors = []
    for j, i in enumerate(INTERFACE_IDS):
        entry = {"update": "interface_data", "id": torch.tensor([i]), "direction": "Z",
                 "prior_distribution": PRIOR_TYPE}
        if PRIOR_TYPE == "normal":
            entry["normal"] = {"mean": mu_0[j].detach(), "std": prior_std[j]}
        else:  # uniform on [mean - a, mean + a] with the same std: a = sqrt(3) * std
            half = np.sqrt(3.0) * prior_std[j]
            entry["uniform"] = {"min": mu_0[j].detach() - half, "max": mu_0[j].detach() + half}
        priors.append(entry)

    compute_gradient(neural_network=model, Decoder_matrix=phi_r, Encoder_matrix=Mphi_r,
                     bias=u_shift, d=u_obs, interpolation_matrix=B, Noise_cov=beta,
                     mean_p=mu_0, Sigma_p=Sigma_p)
    # compute_gradient (gradient.py) picks its own device (MPS on a Mac) and
    # moves the network there in place, so bring it back to this script's device.
    model.to(device)
    model.eval()
    plot_geology(gempy_model, priors, mu_0, prior_std, prefix="prior")

    # ---- GLA ----------------------------------------------------------------
    t0 = datetime.now()
    mu_post, Sigma_post, output, J, NN_out, J_NN = gla(model, mu_0, Sigma_p, B, phi_r,
                                                      u_shift, obs_data, beta)
    print(f"GLA time: {datetime.now() - t0}")

    print("Surrogate accuracy at the true parameter:")
    print("  function, full field :", rel_acc(NN_out @ phi_r.T + u_shift, to_t(u_full)))
    print("  function, receivers  :", rel_acc(output, to_t(u_obs_true)))
    print("  Jacobian, full field :", rel_acc(phi_r @ J_NN, to_t(J_full)))
    print("  Jacobian, receivers  :", rel_acc(J, to_t(J_obs)))
    # [MOD 5] removed: unused A = J^T J, SVD of J and printed shapes, output2

    std_post = torch.sqrt(torch.diagonal(Sigma_post))
    print("Prior mean :", mu_0.detach(), "\nPrior std  :", prior_std)
    print("GLA mean   :", mu_post, "\nGLA std    :", std_post)
    print("GLA covariance:\n", Sigma_post)
    plot_geology(gempy_model, priors, mu_post, std_post, prefix="gla")

    # ---- NUTS ---------------------------------------------------------------
    model_args = (priors, model, B, u_shift, phi_r, obs_data, beta)
    prior_pred = Predictive(pyro_model, num_samples=10000)(*model_args)
    # [MOD 5] removed pyro.render_model(...): its result was never used

    pyro.primitives.enable_validation(is_validate=True)
    mcmc = MCMC(NUTS(pyro_model, **NUTS_SETTINGS), **MCMC_SETTINGS)
    t0 = datetime.now()
    mcmc.run(*model_args)
    print(f"NUTS time: {datetime.now() - t0}")

    samples = mcmc.get_samples(group_by_chain=True)
    pd.DataFrame.from_dict(summary(samples), orient="index").to_csv("pyro_summary.csv")
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        mcmc.summary()
        summary_text = buf.getvalue()
    print(summary_text)
    with open("./mcmc_summary_p.txt", "w") as f:
        f.write(summary_text)

    flat = mcmc.get_samples()
    nuts_mean = torch.stack([flat[f"mu_{j + 1}"].mean() for j in range(len(priors))])
    nuts_std = torch.stack([flat[f"mu_{j + 1}"].std() for j in range(len(priors))])
    print("NUTS mean  :", nuts_mean, "\nNUTS std   :", nuts_std)

    # ---- plots --------------------------------------------------------------
    post_pred = Predictive(pyro_model, flat)(*model_args)
    data = az.from_pyro(posterior=mcmc, prior=prior_pred, posterior_predictive=post_pred)
    az.plot_trace(data)
    plt.savefig(f"{MODEL_DIR}/posterior.png")
    plt.close()
    for j in range(len(priors)):
        az.plot_density(data=[data.posterior, data.prior], shade=0.9, bw=0.006,
                        var_names=[f"mu_{j + 1}"],
                        data_labels=["Posterior", "Prior"])
        plt.savefig(f"{MODEL_DIR}/mu_{j + 1}.png")  # [MOD 9] was mu_{j}: off by one
        plt.close()

    plot_geology(gempy_model, priors, nuts_mean, nuts_std, prefix="nuts")


if __name__ == "__main__":
    print("Script started...")
    start = datetime.now()
    main()
    print(f"Script ended. Elapsed time: {datetime.now() - start}")