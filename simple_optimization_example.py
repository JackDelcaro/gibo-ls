
import argparse
import yaml
import torch
import copy

from src import config
from src.loop import loop

import pickle

torch.set_default_dtype(torch.float64)

def constr_fcn(x):
    """
    Returns >= 0 if x is inside the cloud, < 0 if x is outside.
    The cloud is defined by summing multiple 2D Gaussian bumps
    and subtracting a threshold.
    
    Args:
      x: (N, 2) tensor of points [x1, x2].
    
    Returns:
      c: (N,) tensor. c[i] >= 0 => inside the cloud, c[i] < 0 => outside.
    """
    # Define a few "bump" centers and widths
    # Feel free to add/remove bumps and tweak the parameters
    bumps = [
        # center=(x1_center, x2_center), alpha=width factor
        {'center': torch.tensor([-0.5,  2.0]), 'alpha':  5.0},
        {'center': torch.tensor([0.0,  2.5]), 'alpha':  6.0},
        {'center': torch.tensor([-1.0, 2.3]), 'alpha':  6.0},
        {'center': torch.tensor([-0.3,  1.4]), 'alpha':  4.5},
    ]

    # This threshold sets how large the sum of Gaussians must be
    # to be considered "inside" the cloud.
    threshold = 0.69
    
    # Accumulate the sum of Gaussian bumps
    total_bumps = torch.zeros_like(x[:, 0])  # shape: (N,)
    for b in bumps:
        center = b['center']
        alpha  = b['alpha']
        # Squared distance from center
        dist_sq = (x[:, 0] - center[0])**2 + (x[:, 1] - center[1])**2
        # Gaussian bump = exp(-alpha * distance^2)
        total_bumps += torch.exp(-alpha * dist_sq)
    
    # The constraint function = sum_of_bumps - threshold
    # >= 0 => inside, < 0 => outside
    return total_bumps - threshold

#  Define the cost function
def objective_fcn(x, noise, constr_fcn):
    """
    Compute a objective_fcn-like cost function.
    Input:
      x: a tensor of shape (N, 2)
    Output:
      a tensor of shape (N,)
    """
    x1 = x[:, 0]
    x2 = x[:, 1]
    term1 = 3 * (1 - x1)**2 * torch.exp(-x1**2 - (x2 + 1)**2)
    term2 = -10 * (x1/5 - x1**3 - x2**5) * torch.exp(-x1**2 - x2**2)
    term3 = -(1/3) * torch.exp(-(x1 + 1)**2 - x2**2)
    f_val = term1 + term2 + term3
    result = f_val + noise * torch.randn_like(term1)
    nan_res = torch.full_like(f_val, float("nan"))

    return torch.where(constr_fcn(x) > 0, nan_res, result)

save_results = False
if __name__ == "__main__":

    default_config_file = r"configs/gibo_default.yaml"

    parser = argparse.ArgumentParser(
        description="Run optimization of synthetic functions."
    )
    parser.add_argument("-c", "--config", type=str, help="Path to config file.", default=default_config_file)
    
    args = parser.parse_args()
    print(args.config)

    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    
    # Additional configuration settings
    cfg['optimizer_config']['marginal_likelihood'] = 'loo-pll'

    dim = 2
    cfg['max_objective_calls'] = 50
    cfg['optimizer_config']['num_samples_before_gp_training'] = (dim+1)*2
    cfg['optimizer_config']['model_config']['N_max'] = dim*10
    cfg['optimizer_config']['model_config']['ard_num_dims'] = dim
    cfg['optimizer_config']['max_samples_per_iteration'] = dim+1

    cfg['optimizer_config']['hyperparameter_config']['hypers']['likelihood.noise'] = 1e-2

    # cfg['optimizer_config']['lin_search_routine_config']['enable_plotting'] = True
    # cfg['optimizer_config']['lin_search_routine'] = None
    cfg['optimizer_config']['optimizer_torch_config']['lr'] = 0.5
    cfg['optimizer_config']['generate_initial_data'] = None
    # cfg['optimizer_config']['generate_initial_data_config']['bounds'] = torch.tensor([[-2.0, -2.0], [2.0, 2.0]], dtype=torch.float64)

    original_cfg = copy.deepcopy(cfg)
    
    cfg = config.insert(cfg, config.insertion_config)
    cfg = config.evaluate_hyperpriors(cfg)

    noise = 0
    objective_fcn_handle = lambda x: objective_fcn(x, noise, constr_fcn)
    
    sample_point = torch.tensor([1.0, 2.15], dtype=torch.float64).unsqueeze(dim=0)
    optimization_data = loop(
        params_init         = sample_point,
        max_iterations      = cfg['max_iterations'],
        max_objective_calls = cfg['max_objective_calls'],
        objective           = objective_fcn_handle,
        Optimizer           = cfg['method'],
        optimizer_config    = cfg['optimizer_config'],
        verbose             = cfg['optimizer_config']['verbose'],
    )

    import pandas as pd
    df = pd.DataFrame(optimization_data['iteration_data'])
    print(df)

    if save_results:
        # Create a dictionary with both the iteration data and the original config.
        results_to_save = {
            "df": df,
            "cfg": original_cfg,
        }
        # Save the dictionary to a pickle file.
        with open('data/saved_results.pkl', "wb") as f:
            pickle.dump(results_to_save, f)