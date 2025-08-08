
import argparse
import yaml
import time
import inspect
import torch
import botorch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import copy
import os
import pickle

from src import config
from src.loop import loop
from src.utils import get_best_point, plot_gp_1D, reconstruct_optimization_state, compute_numerical_derivative
from src.vdp_model import VDPNaN, VDPUCB

from gpytorch.constraints import GreaterThan, Interval
from src.acquisition_function import GradientInformation
from matplotlib.animation import PillowWriter

import pandas as pd

torch.set_default_dtype(torch.float64)
# torch.manual_seed(69)

## PLOT FUNCTION
import matplotlib.colors as mcolors
def darken_color(color, factor=0.8):
    """
    Darkens the given color by multiplying its RGB values by the given factor.
    A factor less than 1.0 will produce a darker color.
    """
    rgb = mcolors.to_rgb(color)
    return tuple(max(0, c * factor) for c in rgb)

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

      
def plot_gp_line_search(df, idx, cfg, objective_fn, axes, feas_gp_model, virt_gp_model, vdp_function_handle):
    global use_latex

    row = df.loc[idx]

    input_scaling_coeffs = cfg['optimizer_config']['objfcn_input_scaling_coeffs']
    if input_scaling_coeffs is None:
        input_scaling_coeffs = torch.ones_like(torch.tensor(row['sample_point']))
    start_point = torch.tensor(row["gd_starting_point"])/input_scaling_coeffs
    direction = torch.tensor(row["gradient"])/input_scaling_coeffs
    virt_gp_train_x = torch.tensor(row["gp_train_virtualpoints_x"])/input_scaling_coeffs
    virt_gp_train_y = torch.tensor(row["gp_train_virtualpoints_y"])

    # Define a custom acquisition function that “sees” t in [0,1].
    # It wraps the original acquisition function from the caller.
    class LineAcquisitionFunction(botorch.acquisition.AcquisitionFunction):
        def __init__(self, original_acqf, p: torch.Tensor, direction: torch.Tensor):
            # Use the same model as the original acquisition function.
            super().__init__(model=original_acqf.model)
            self.original_acqf = original_acqf
            self.start_point = start_point
            self.direction = direction

        def forward(self, t: torch.Tensor) -> torch.Tensor:
            # Map the 1D input t to the full search space along the line.
            x = self.start_point + t * self.direction
            return self.original_acqf(x)
    
    if row['type'] == 'nan_gp_line_search':
        acquisition_function = cfg['optimizer_config']['nan_it_routine_config']['acquisition_function']
        acqf_config = cfg['optimizer_config']['nan_it_routine_config']['acqf_config']
    else:
        acquisition_function = cfg['optimizer_config']['lin_search_routine_config']['acquisition_function']
        acqf_config = cfg['optimizer_config']['lin_search_routine_config']['acqf_config']

    sig = inspect.signature(acquisition_function)
    if "best_f" in sig.parameters and sig.parameters["best_f"].default is inspect.Parameter.empty:
        if acqf_config is None:
            acqf_config = {}
        virt_gp_train_t = ((virt_gp_train_x - start_point)* direction).sum(dim=1)/(direction * direction).sum()
        orthogonal_components = virt_gp_train_x - (start_point + virt_gp_train_x.unsqueeze(1) * direction)
        direction_mask = (torch.sum(torch.abs(orthogonal_components), dim = 1) < 1e-10) & (virt_gp_train_t <= 1) & (virt_gp_train_t >= 0)
        virt_gp_train_y_line = virt_gp_train_y[direction_mask]
        acqf_config["best_f"] = virt_gp_train_y_line[~torch.isnan(virt_gp_train_y_line) & ~torch.isinf(virt_gp_train_y_line)].max()
        
    acq_func_original = acquisition_function(
            virt_gp_model,
            **(acqf_config if acqf_config is not None else {})
        )
    line_acq_func = LineAcquisitionFunction(acq_func_original, start_point, direction)

    if row['type'] == 'nan_gp_line_search':
        t_min_diff = cfg['optimizer_config']['nan_it_routine_config']['t_min_diff'] if 't_min_diff' in cfg['optimizer_config']['nan_it_routine_config'] else None
    else:
        t_min_diff = cfg['optimizer_config']['lin_search_routine_config']['t_min_diff'] if 't_min_diff' in cfg['optimizer_config']['lin_search_routine_config'] else None
    
    plot_gp_1D(feas_gp_model, virt_gp_model, start_point, direction, virt_gp_train_y,
               line_acqf=line_acq_func, vdp_function=vdp_function_handle,
               t_min_diff=t_min_diff,
               axes=axes, use_latex=use_latex, plot_grad=True,
               plot_interactive=False, plot_legend=False
               )
    
    # Create a grid of t values in [0,1]
    t_vals = torch.linspace(-0.05, 1.05, 100)
    # Map t to full-dimensional points: x = start_point + t * direction
    x_line_plot = start_point + t_vals.unsqueeze(1) * direction  # shape: (t_grid, D)
    # Evaluate the true objective along the line
    true_vals = objective_fn(x_line_plot * input_scaling_coeffs).detach().squeeze()

    # Get the current figure
    axes[0].plot(t_vals, true_vals, 'k--', label='True objective')
    axes[1].plot(t_vals, compute_numerical_derivative(t_vals*direction.norm(), true_vals), 'k--', label='True derivative')
    
    # Set legend
    handles, labels = axes[0].get_legend_handles_labels()
    new_order = [1, 2, 6, 3, 4, 5, 0, 7, 8]
    reordered_handles = [handles[i] for i in new_order]
    reordered_labels  = [labels[i]  for i in new_order]
    axes[0].legend(
        reordered_handles,
        reordered_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.37),
        ncol=3,
        fancybox=True,
        shadow=False,
        fontsize=10,
    )
    axes[1].legend(fontsize=10)
    axes[2].legend(fontsize=10)
    
    global save_animation
    if not save_animation:
        plt.draw()
        plt.pause(0.01)


def plot_gp_evaluation(df, idx, cfg, objective_fn, axes, feas_gp_model, virt_gp_model, vdp_function_handle):
    global use_latex

    row = df.loc[idx]

    input_scaling_coeffs = cfg['optimizer_config']['objfcn_input_scaling_coeffs']
    if input_scaling_coeffs is None:
        input_scaling_coeffs = torch.ones_like(torch.tensor(row['sample_point']))
    row_idx = torch.nonzero(torch.tensor((df.index <= idx) & (df['type'] == 'iteration')))[-1].item()
    start_point = torch.tensor(df.loc[row_idx]["gd_starting_point"])/input_scaling_coeffs
    end_point = torch.tensor(df.loc[row_idx]["sample_point"])/input_scaling_coeffs

    direction = end_point - start_point
    virt_gp_train_x = torch.tensor(row["gp_train_virtualpoints_x"])/input_scaling_coeffs
    virt_gp_train_y = torch.tensor(row["gp_train_virtualpoints_y"])

    def nan_fcn(x):
        return torch.full([x.shape[0], 1], torch.tensor(float('nan')))

    plot_gp_1D(feas_gp_model, virt_gp_model, start_point, direction, virt_gp_train_y,
               vdp_function=vdp_function_handle, line_acqf=nan_fcn, use_latex=use_latex, plot_grad=True,
               axes=axes, plot_interactive=False, plot_legend=False)
    
    # Create a grid of t values in [0,1]
    t_vals = torch.linspace(-0.05, 1.05, 100)
    # Map t to full-dimensional points: x = start_point + t * direction
    x_line_plot = start_point + t_vals.unsqueeze(1) * direction  # shape: (t_grid, D)
    # Evaluate the true objective along the line
    true_vals = objective_fn(x_line_plot * input_scaling_coeffs).detach().squeeze()

    # Get the current figure
    axes[0].plot(t_vals, true_vals, 'k--', label='True objective')
    axes[1].plot(t_vals, compute_numerical_derivative(t_vals*direction.norm(), true_vals), 'k--', label='True derivative')
    
    # Set legend
    handles, labels = axes[0].get_legend_handles_labels()
    new_order = [1, 2, 6, 3, 4, 5, 0, 7, 8]
    reordered_handles = [handles[i] for i in new_order]
    reordered_labels  = [labels[i]  for i in new_order]
    axes[0].legend(
        reordered_handles,
        reordered_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.37),
        ncol=3,
        fancybox=True,
        shadow=False,
        fontsize=10,
    )
    axes[1].legend(fontsize=10)

    global save_animation
    if not save_animation:
        plt.draw()
        plt.pause(0.01)

def update_contour_plot(df, idx, cfg, ax_left, curr_it_point, curr_it_line, curr_search_line, curr_eval_points, curr_line_points, curr_eval_nanpoints, curr_line_nanpoints):
    global acqf_contourf
    global use_latex

    row = df.loc[idx]
    curr_type = row['type']
    
    input_scaling_coeffs = cfg['optimizer_config']['objfcn_input_scaling_coeffs']
    if input_scaling_coeffs is None:
        input_scaling_coeffs = torch.ones_like(torch.tensor(row['sample_point']))
    
    last_it_idx = torch.nonzero(torch.tensor((df.index <= idx) & (df['type'] == 'iteration')))[-1].item()
    start_point = torch.tensor(df.loc[last_it_idx]["gd_starting_point"])/input_scaling_coeffs
    end_point = torch.tensor(df.loc[last_it_idx]["sample_point"])/input_scaling_coeffs
    direction = end_point - start_point
    
    virt_gp_train_x = torch.tensor(row["gp_train_virtualpoints_x"])/input_scaling_coeffs
    virt_gp_train_y = torch.tensor(row["gp_train_virtualpoints_y"])
    virt_gp_train_isnan = torch.tensor(row["gp_train_virtualpoints_isnan"], dtype=torch.bool)
    
    indexes = (df.index > 0) & (df.index <= idx) & (df['type'] == 'iteration') & (df['gd_starting_point'].apply(lambda x: len(x) > 0))
    if any(indexes):
        iteration_points = np.vstack(df[indexes]['gd_starting_point'])/input_scaling_coeffs
        iteration_points = torch.vstack([iteration_points, end_point])
    if curr_type == "iteration" and idx > 0:
        iteration_points = torch.vstack([iteration_points, torch.tensor(df.iloc[idx]['sample_point'])/input_scaling_coeffs])
    elif idx < torch.nonzero(torch.tensor(df['type'] == 'iteration'))[1].item():
        iteration_points = torch.tensor(df.iloc[0]['sample_point'])/input_scaling_coeffs
    
    iteration_points_line = iteration_points
    if curr_type == "evaluation":
        # Look backwards from the current index until a non-'evaluation' row is found
        last_non_eval_type = None
        for i in range(idx - 1, -1, -1):
            if df.loc[i]['type'] != "evaluation":
                last_non_eval_type = df.loc[i]['type']
                break
        if last_non_eval_type in ["gp_line_search", "nan_gp_line_search"]:
            virt_gp_train_t = ((virt_gp_train_x - start_point)* direction).sum(dim=1)/(direction * direction).sum()
            orthogonal_components = virt_gp_train_x - (start_point + virt_gp_train_t.unsqueeze(1) * direction)
            direction_mask = (torch.sum(torch.abs(orthogonal_components), dim = 1) < 1e-10) & (virt_gp_train_t <= 1) & (virt_gp_train_t >= 0)
            best_point, best_point_value = get_best_point(sample_points=virt_gp_train_x[direction_mask], observations=virt_gp_train_y[direction_mask])
            iteration_points_line = iteration_points.clone()
            iteration_points = torch.vstack([iteration_points, best_point])

    it_mask = torch.any(torch.sum(torch.abs(virt_gp_train_x.unsqueeze(1) - iteration_points.unsqueeze(0)), dim=2) < 1e-9, dim=1)

    virt_gp_train_t = ((virt_gp_train_x - start_point)* direction).sum(dim=1)/(direction * direction).sum()
    orthogonal_components = virt_gp_train_x - (start_point + virt_gp_train_t.unsqueeze(1) * direction)
    direction_mask = (torch.sum(torch.abs(orthogonal_components), dim = 1) < 1e-10) & (virt_gp_train_t <= 1) & (virt_gp_train_t >= 0)
    
    line_virt_gp_train_x_nan = virt_gp_train_x[~it_mask & direction_mask & virt_gp_train_isnan]
    line_virt_gp_train_x = virt_gp_train_x[~it_mask & direction_mask & ~virt_gp_train_isnan]
    eval_virt_gp_train_x_nan = virt_gp_train_x[~it_mask & ~direction_mask & virt_gp_train_isnan]
    eval_virt_gp_train_x = virt_gp_train_x[~it_mask & ~direction_mask & ~virt_gp_train_isnan]
    
    if line_virt_gp_train_x.shape[0] > 0:
        curr_line_points.set_offsets(np.column_stack((line_virt_gp_train_x[:,0].flatten().numpy(), line_virt_gp_train_x[:,1].flatten().numpy())))
    else:
        curr_line_points.set_offsets(np.empty((0, 2)))

    if line_virt_gp_train_x_nan.shape[0] > 0:
        curr_line_nanpoints.set_offsets(np.column_stack((line_virt_gp_train_x_nan[:,0].flatten().numpy(), line_virt_gp_train_x_nan[:,1].flatten().numpy())))
    else:
        curr_line_nanpoints.set_offsets(np.empty((0, 2)))
    
    if eval_virt_gp_train_x.shape[0] > 0:
        curr_eval_points.set_offsets(np.column_stack((eval_virt_gp_train_x[:,0].flatten().numpy(), eval_virt_gp_train_x[:,1].flatten().numpy())))
    else:
        curr_eval_points.set_offsets(np.empty((0, 2)))
    
    if eval_virt_gp_train_x_nan.shape[0] > 0:
        curr_eval_nanpoints.set_offsets(np.column_stack((eval_virt_gp_train_x_nan[:,0].flatten().numpy(), eval_virt_gp_train_x_nan[:,1].flatten().numpy())))
    else:
        curr_eval_nanpoints.set_offsets(np.empty((0, 2)))

    curr_it_point.set_offsets(np.column_stack((iteration_points[:,0].flatten().numpy(), iteration_points[:,1].flatten().numpy())))
    curr_it_line.set_data(iteration_points_line[:,0].flatten().numpy(), iteration_points_line[:,1].flatten().numpy())

    if curr_type == "evaluation":
        
        original_delta = cfg['optimizer_config']['optimize_acqf_config']['delta']
        
        if 'lin_search_routine_config' in cfg['optimizer_config']:
            lin_search_routine_config = cfg['optimizer_config']['lin_search_routine_config']
            curr_lr_coeff_idx = torch.nonzero(torch.tensor((df.index <= idx) & (df['learning_rate'].apply(lambda x: isinstance(x, float) > 0)), dtype=torch.bool))[-1].item()
            original_lr_coeff_idx = torch.nonzero(torch.tensor((df.index <= idx) & (df['learning_rate'].apply(lambda x: isinstance(x, float) > 0)), dtype=torch.bool))[0].item()
            delta_multiplier = df.iloc[curr_lr_coeff_idx]['learning_rate']/df.iloc[original_lr_coeff_idx]['learning_rate']
            delta = original_delta * delta_multiplier
            if 'delta_min' in lin_search_routine_config and delta < lin_search_routine_config['delta_min']:
                delta = lin_search_routine_config['delta_min']
        else:
            delta = original_delta
        
        curr_bounds = iteration_points[-1].clone() + torch.tensor([[-delta], [delta]])

        _, prev_virt_gp_model, _ = reconstruct_optimization_state(df, idx-1, copy.deepcopy(cfg))
        acquisition_fcn = GradientInformation(prev_virt_gp_model)
        acquisition_fcn.update_theta_i(iteration_points[-1].unsqueeze(0).clone())
        
        n_grid = 15
        x1_lin = torch.linspace(curr_bounds[0][0].item(), curr_bounds[1][0].item(), n_grid)
        x2_lin = torch.linspace(curr_bounds[0][1].item(), curr_bounds[1][1].item(), n_grid)
        X1, X2 = torch.meshgrid(x1_lin, x2_lin, indexing='ij')
        grid = torch.cat([X1.reshape(-1, 1), X2.reshape(-1, 1)], dim=1)

        # Compute the true cost function
        with torch.no_grad():
            acqf_values = acquisition_fcn(grid.unsqueeze(1)).view_as(X1).numpy()
        vmin = np.nanmin(acqf_values) - 0.1 * (np.nanmax(acqf_values) - np.nanmin(acqf_values))
        vmax = np.nanmax(acqf_values) + 0.1 * (np.nanmax(acqf_values) - np.nanmin(acqf_values))
        levels = np.linspace(vmin, vmax, 30)
        
        if acqf_contourf is not None:
            # for coll in acqf_contourf.collections:
            #     coll.remove()
            acqf_contourf.remove()
            acqf_contourf = None
        acqf_contourf = ax_left.contourf(X1.numpy(), X2.numpy(), acqf_values, levels=levels, cmap='binary', vmin=vmin, vmax=vmax, alpha=0.5)

        curr_bounds_points = torch.tensor([[curr_bounds[0][0], curr_bounds[0][1]], [curr_bounds[1][0], curr_bounds[0][1]], [curr_bounds[1][0], curr_bounds[1][1]], [curr_bounds[0][0], curr_bounds[1][1]], [curr_bounds[0][0], curr_bounds[0][1]]])
        curr_search_line.set_data(curr_bounds_points[:,0].flatten().numpy(), curr_bounds_points[:,1].flatten().numpy())

    else:
        if acqf_contourf is not None:
            # for coll in acqf_contourf.collections:
            #     coll.remove()
            # acqf_contourf = None            
            acqf_contourf.remove()
            acqf_contourf = None
        # curr_search_line.set_data([], [])

def update_figure(frame_idx, df, cfg, objective_fcn_handle, ax_left, ax_obj, ax_grad, ax_acq, curr_it_point, curr_it_line, curr_search_line, curr_eval_points, curr_line_points, curr_eval_nanpoints, curr_line_nanpoints):

    curr_x1_lims = ax_left.get_xlim()
    curr_x2_lims = ax_left.get_ylim()
    curr_x1_center = np.mean(curr_x1_lims)
    curr_x2_center = np.mean(curr_x2_lims)
    curr_x1_diff = np.diff(curr_x1_lims)
    curr_x2_diff = np.diff(curr_x2_lims)
    all_sampled_points = np.concatenate(df['sample_point'].to_numpy(), axis=0)
    all_observations = np.concatenate(df['observation'].to_numpy(), axis=0)
    best_idx = np.nanargmax(all_observations)
    best_x1 = all_sampled_points[best_idx, 0].item()
    best_x2 = all_sampled_points[best_idx, 1].item()
    new_x1_center = curr_x1_center*0.25 + 0.75*best_x1
    new_x2_center = curr_x2_center*0.25 + 0.75*best_x2
    new_x1_diff = 0.5 * curr_x1_diff
    new_x2_diff = 0.5 * curr_x2_diff
    if frame_idx == 0:
        all_sampled_x1 = all_sampled_points[:,0].flatten()
        all_sampled_x2 = all_sampled_points[:,1].flatten()
        max_sampled_x1, min_sampled_x1 = all_sampled_x1.max().item(), all_sampled_x1.min().item()
        max_sampled_x2, min_sampled_x2 = all_sampled_x2.max().item(), all_sampled_x2.min().item()
        diff_x1 = max_sampled_x1 - min_sampled_x1
        diff_x2 = max_sampled_x2 - min_sampled_x2
        if diff_x1 > diff_x2:
            diff = diff_x1
        else:
            diff = diff_x2
        center_point_x1 = 0.5*min_sampled_x1 + 0.5*max_sampled_x1
        center_point_x2 = 0.5*min_sampled_x2 + 0.5*max_sampled_x2
        plot_margin = 0.3
        min_plot_x1, max_plot_x1 = center_point_x1 - 0.5*diff - plot_margin, center_point_x1 + 0.5*diff + plot_margin
        min_plot_x2, max_plot_x2 = center_point_x2 - 0.5*diff - plot_margin, center_point_x2 + 0.5*diff + plot_margin
        ax_left.set_xlim(min_plot_x1, max_plot_x1)
        ax_left.set_ylim(min_plot_x2, max_plot_x2)

        curr_search_line.set_data([], [])

    if frame_idx == torch.nonzero(torch.tensor(df['type'] == 'iteration'))[-2] or frame_idx == torch.nonzero(torch.tensor(df['type'] == 'iteration'))[-1]:
    # if frame_idx == torch.nonzero(torch.tensor(df['type'] == 'iteration'))[-3]:
        ax_left.set_xlim(new_x1_center - 0.5*new_x1_diff, new_x1_center + 0.5*new_x1_diff)
        ax_left.set_ylim(new_x2_center - 0.5*new_x2_diff, new_x2_center + 0.5*new_x2_diff)
    
    update_contour_plot(df, frame_idx, cfg, ax_left, curr_it_point, curr_it_line, curr_search_line, curr_eval_points, curr_line_points, curr_eval_nanpoints, curr_line_nanpoints)

    ax_obj.clear()
    ax_grad.clear()
    ax_acq.clear()

    if frame_idx >= torch.nonzero(torch.tensor(df['type'] == 'iteration'))[1].item():
        feas_gp_model, virt_gp_model, vdp_function_handle = reconstruct_optimization_state(df, frame_idx, copy.deepcopy(cfg))
        ax_obj.set_visible(True)
        ax_grad.set_visible(True)
        ax_acq.set_visible(True)
        if df.loc[frame_idx, 'type'] == 'gp_line_search' or df.loc[frame_idx, 'type'] == 'nan_gp_line_search':
            plot_gp_line_search(df, frame_idx, cfg, objective_fcn_handle, ax_right, feas_gp_model, virt_gp_model, vdp_function_handle)
        elif frame_idx > 0 and df.loc[frame_idx, 'type'] != 'init_point':
            plot_gp_evaluation(df, frame_idx, cfg, objective_fcn_handle, ax_right, feas_gp_model, virt_gp_model, vdp_function_handle)                
    else:
        ax_obj.set_visible(False)
        ax_grad.set_visible(False)
        ax_acq.set_visible(False)

    plt.suptitle(f"Iteration {df.iloc[frame_idx]['iteration_number']}, Obj eval: {df.iloc[frame_idx]['evaluation_number']}, Mode: {df.iloc[frame_idx]['type']}",
            fontsize=14, x=0.4)
        
    plt.draw()

use_latex = False
acqf_contourf = None
save_animation = False
if __name__ == "__main__": 

    # Open and load the pickle file.
    with open('data/saved_results.pkl', 'rb') as f:
        loaded_data = pickle.load(f)

    # Extract the configuration.
    cfg = loaded_data["cfg"]
    df = loaded_data["df"]

    cfg = config.insert(cfg, config.insertion_config)
    cfg = config.evaluate_hyperpriors(cfg)

    if use_latex:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    fig = plt.figure(figsize=(17.3, 8.6))
    plt.ion()

    # Create a GridSpec with 3 rows and 2 columns.
    # The left column (index 0) will host a subplot that spans all rows,
    # while the right column (index 1) will have three subplots, one per row.
    gs = gridspec.GridSpec(3, 2, width_ratios=[3, 2])

    # Left subplot: spans all rows in the first column
    ax_left = fig.add_subplot(gs[:, 0])
    ax_left.set_title('Objective function', fontsize=14)
    ax_left.set_xlabel('$x_1$', fontsize=14)
    ax_left.set_ylabel('$x_2$', fontsize=14)

    n_grid = 500
    all_sampled_points = torch.tensor(np.concatenate(df['sample_point'].to_numpy(), axis=0))
    # all_observations = torch.tensor(np.concatenate(df['observation'].to_numpy(), axis=0))
    # best_idx = np.nanargmax(all_observations.numpy())
    # best_x1 = all_sampled_points[best_idx, 0].item()
    # best_x2 = all_sampled_points[best_idx, 1].item()
    all_sampled_x1 = all_sampled_points[:,0].flatten()
    all_sampled_x2 = all_sampled_points[:,1].flatten()
    max_sampled_x1, min_sampled_x1 = all_sampled_x1.max().item(), all_sampled_x1.min().item()
    max_sampled_x2, min_sampled_x2 = all_sampled_x2.max().item(), all_sampled_x2.min().item()
    diff_x1 = max_sampled_x1 - min_sampled_x1
    diff_x2 = max_sampled_x2 - min_sampled_x2
    if diff_x1 > diff_x2:
        diff = diff_x1
    else:
        diff = diff_x2
    center_point_x1 = 0.5*min_sampled_x1 + 0.5*max_sampled_x1
    center_point_x2 = 0.5*min_sampled_x2 + 0.5*max_sampled_x2
    plot_margin = 0.3
    min_plot_x1, max_plot_x1 = center_point_x1 - 0.5*diff - plot_margin, center_point_x1 + 0.5*diff + plot_margin
    min_plot_x2, max_plot_x2 = center_point_x2 - 0.5*diff - plot_margin, center_point_x2 + 0.5*diff + plot_margin

    # x1_lin = torch.linspace(df['sample_point'][0][0,0].item() - 2.0, df['sample_point'][0][0,0].item() + 1.0, n_grid)
    # x2_lin = torch.linspace(df['sample_point'][0][0,1].item() - 1.0, df['sample_point'][0][0,1].item() + 1.0, n_grid)
    x1_lin = torch.linspace(min_plot_x1, max_plot_x1, n_grid)
    x2_lin = torch.linspace(min_plot_x2, max_plot_x2, n_grid)
    X1, X2 = torch.meshgrid(x1_lin, x2_lin, indexing='ij')
    grid = torch.cat([X1.reshape(-1, 1), X2.reshape(-1, 1)], dim=1)

    # Compute the true cost function
    noise = 0
    objective_fcn_handle = lambda x: objective_fcn(x, noise, constr_fcn)
    true_cost = objective_fcn_handle(grid).view_as(X1).numpy()

    vmin = np.nanmin(true_cost)
    vmax = np.nanmax(true_cost)
    levels = np.linspace(vmin, vmax, 30)

    red_points_color  = '#c1121f'
    blue_points_color = '#669bbc'

    cp1 = ax_left.contourf(X1.numpy(), X2.numpy(), true_cost, levels=levels, cmap='viridis', vmin=vmin, vmax=vmax)
    cb = fig.colorbar(cp1, ax=ax_left, location='left', pad=0.1)
    curr_it_line, = ax_left.plot([], [], color='k', linestyle='--', label='Step')
    curr_search_line, = ax_left.plot([], [], color='white', linestyle=':', label='Evaluation search box')
    curr_it_point = ax_left.scatter(df['sample_point'][0][0,0].item(), df['sample_point'][0][0,1].item(), s=70, color=red_points_color, edgecolors='white', linewidth=1.5, label='Iteration point', zorder=5)
    curr_eval_points = ax_left.scatter([], [], s=45, c=blue_points_color, edgecolors='white', linewidth=1.5, label='Evaluated points', zorder=6)
    curr_eval_nanpoints = ax_left.scatter([], [], s=45, c=blue_points_color, marker='x', label='Evaluated (nan) points', zorder=7)
    curr_line_points = ax_left.scatter([], [], s=45, color=darken_color(blue_points_color, 0.7), edgecolors='white', linewidth=1.5,label='Linesearch points', zorder=8)
    curr_line_nanpoints = ax_left.scatter([], [], s=45, color=darken_color(blue_points_color, 0.7), marker='x', label='Linesearch (nan) points', zorder=9)
    
    # Set legend
    handles, labels = ax_left.get_legend_handles_labels()
    new_order = [0, 1, 2, 3, 4, 5, 6]
    reordered_handles = [handles[i] for i in new_order]
    reordered_labels  = [labels[i]  for i in new_order]
    ax_left.legend(
        reordered_handles,
        reordered_labels,
        loc='upper center',
        ncol=3,
        fancybox=True,
        shadow=False,
        fontsize=10,
    )

    # Right subplots: one in each row of the second column
    ax_obj  = fig.add_subplot(gs[0, 1])
    ax_grad = fig.add_subplot(gs[1, 1], sharex=ax_obj)
    ax_acq  = fig.add_subplot(gs[2, 1], sharex=ax_obj)
    ax_right = [ax_obj, ax_grad, ax_acq]

    update_animation = lambda idx: update_figure(idx, df, cfg, objective_fcn_handle, ax_left, ax_obj, ax_grad, ax_acq, curr_it_point, curr_it_line, curr_search_line, curr_eval_points, curr_line_points, curr_eval_nanpoints, curr_line_nanpoints)
    
    feas_gp_model, virt_gp_model, vdp_function_handle = reconstruct_optimization_state(df, 0, copy.deepcopy(cfg))
    plot_gp_evaluation(df, 0, cfg, objective_fcn_handle, ax_right, feas_gp_model, virt_gp_model, vdp_function_handle)
    plt.tight_layout()
    plt.show()
    plt.pause(0.01)

    ani = animation.FuncAnimation(fig, update_animation, frames=len(df), interval=1000)

    if save_animation:
        writer = PillowWriter(fps=0.75)  # Adjust fps as needed
        ani.save("data/saved_animation.gif", writer=writer)

    plt.show()

    plt.ioff()
    plt.show()
