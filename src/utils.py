import torch
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
import copy
from src.vdp_model import VDPNaN, VDPUCB
from collections import OrderedDict

def get_best_point(sample_points: Union[List, torch.tensor], observations: Union[List, torch.tensor], return_index: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns the best point and its value."""
    if not torch.is_tensor(sample_points):
        observation_tensor = torch.vstack(observations)
        sample_points_tensor = torch.vstack(sample_points)
    else:
        observation_tensor = observations
        sample_points_tensor = sample_points

    non_nan_mask = ~torch.isnan(observation_tensor)
    non_nan_values = observation_tensor[non_nan_mask]
    max_value, max_index_in_non_nan = torch.max(non_nan_values, dim=0)
    max_index_in_original = torch.nonzero(non_nan_mask)[max_index_in_non_nan].item()
    best_point = sample_points_tensor[max_index_in_original]
    best_point_value = observation_tensor[max_index_in_original]

    if not return_index:
        return best_point.clone(), best_point_value.clone()
    else:
        return best_point.clone(), best_point_value.clone(), max_index_in_original

def _mul_broadcast_shape(*shapes, error_msg=None):
    """Compute dimension suggested by multiple tensor indices (supports broadcasting)"""

    # Pad each shape so they have the same number of dimensions
    num_dims = max(len(shape) for shape in shapes)
    shapes = tuple([1] * (num_dims - len(shape)) + list(shape) for shape in shapes)

    # Make sure that each dimension agrees in size
    final_size = []
    for size_by_dim in zip(*shapes):
        non_singleton_sizes = tuple(size for size in size_by_dim if size != 1)
        if len(non_singleton_sizes):
            if any(size != non_singleton_sizes[0] for size in non_singleton_sizes):
                if error_msg is None:
                    raise RuntimeError("Shapes are not broadcastable for mul operation")
                else:
                    raise RuntimeError(error_msg)
            final_size.append(non_singleton_sizes[0])
        # In this case - all dimensions are singleton sizes
        else:
            final_size.append(1)

    return torch.Size(final_size)

def _match_batch_dims(x1, *args):
    batch_shape = x1.shape[:-2]
    # Make sure the batch shapes agree for training/test data
    output = [x1]
    for x2 in args:
        if batch_shape != x1.shape[:-2]:
            batch_shape = _mul_broadcast_shape(batch_shape, x1.shape[:-2])
            x1 = x1.expand(*batch_shape, *x1.shape[-2:])
        if batch_shape != x2.shape[:-2]:
            batch_shape = _mul_broadcast_shape(batch_shape, x2.shape[:-2])
            x1 = x1.expand(*batch_shape, *x1.shape[-2:])
            x2 = x2.expand(*batch_shape, *x2.shape[-2:])
        output.append(x2)
    return output


def _match_dtype(x_in, *args):
    in_dtype = x_in.dtype
    output = []
    for arg in args:
        out = arg.to(in_dtype)
        output.append(out)
    return output


def unit_vector(v):
    """ Returns the unit vector of the vector.  """
    return v / torch.norm(v)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return torch.acos(torch.clamp(torch.dot(v1_u.float(), v2_u.float()), -1.0, 1.0))


def _outer_product(v1, v2):
    return torch.einsum('p,q->pq', v1, v2)


def get_history_field(history, field: str):
    """
    Extracts the data for the given field from the history list.
    
    Each entry in self.history is expected to be a dict. If an entry doesn't
    have the specified key, it is skipped.
    
    Parameters:
        field (str): The field name to extract from each history record.
    
    Returns:
        List of all elements in corresponding to the field name
    """
    collected = []
    for row in history:
        if field in row:
            collected.append(row[field])
    
    if not collected:
        return None  # or you could return an empty tensor if preferred
    
    return collected

def min_distance_constraint(X: torch.Tensor, points: torch.Tensor, min_distance: float) -> torch.Tensor:
    """
    Computes the minimum Euclidean distance from each candidate in X to any point in `points`,
    and subtracts a predefined minimum distance threshold (min_distance).

    Each candidate is required to be at least min_distance away (in Euclidean distance)
    from every point in `points`. The function returns, for each candidate, the difference 
    between its minimum distance to any point and min_distance. A non-negative value indicates 
    that the candidate satisfies the constraint.

    Args:
        X (torch.Tensor): A tensor of candidate points with shape (num_restarts, q, d),
                          where d is the number of dimensions.
        points (torch.Tensor): A tensor of reference (already-sampled) points with shape (n, d),
                               where n is the number of points.

    Returns:
        torch.Tensor: A tensor of shape (num_restarts, q) where each element is given by
                      (min_distance(candidate, points) - min_distance). The candidate satisfies
                      the constraint if the value is >= 0.

    Note:
        The variable min_distance (a float) must be defined in the surrounding scope.
    """

    original_x_len = len(X.shape)
    if original_x_len == 1:
        X = X.unsqueeze(0).unsqueeze(0)
    elif original_x_len == 2:
        X = X.unsqueeze(0)

    # Compute the difference between each candidate and each sampled point.
    # X has shape (num_restarts, q, d) and we reshape points to (1, 1, n, d)
    diff = X.unsqueeze(2) - points.unsqueeze(0).unsqueeze(0)  # shape: (num_restarts, q, n, d)
    
    # Compute the Euclidean distance along the last dimension (d)
    euclidean_dist = torch.norm(diff, dim=-1)  # shape: (num_restarts, q, n)
    
    # For each candidate, get the minimum distance to any sampled point
    min_dist, _ = euclidean_dist.min(dim=2)  # shape: (num_restarts, q)
    
    # Return the constraint value (should be >= 0 for feasibility)
    result = min_dist - min_distance

    if original_x_len == 1:
        result = result.squeeze()
    elif original_x_len == 2:
        result = result.squeeze(1)

    return result

def compute_safe_intervals(points: torch.Tensor, eps: float, interval_start: float=0, interval_end: float=1):
    """
    Given a tensor of points and a distance eps, and an interval [interval_start, interval_end],
    computes the intervals within [interval_start, interval_end] where every point is at least eps
    away from each given point. Points outside [interval_start, interval_end] are ignored.
    
    Args:
        points (torch.Tensor): 1D tensor of points.
        eps (float): safety margin distance.
        interval_start (float): lower bound of the interval.
        interval_end (float): upper bound of the interval.
    
    Returns:
        List of [start, end] intervals representing safe regions.
    """
    # Filter out points that lie outside the given interval
    points = points[((points + eps) >= interval_start) & ((points - eps) <= interval_end)]
    
    # If no points remain, the entire interval is safe
    if points.numel() == 0:
        return [[interval_start, interval_end]]
    
    # Compute the unsafe intervals for each point
    lefts = torch.clamp(points - eps, min=interval_start)
    rights = torch.clamp(points + eps, max=interval_end)
    intervals = torch.stack([lefts, rights], dim=1)
    
    # Sort intervals by their starting points
    sorted_intervals = intervals[torch.argsort(intervals[:, 0])]
    
    # Merge overlapping intervals
    merged_intervals = []
    current_left, current_right = sorted_intervals[0, 0].item(), sorted_intervals[0, 1].item()
    for i in range(1, sorted_intervals.size(0)):
        left, right = sorted_intervals[i, 0].item(), sorted_intervals[i, 1].item()
        if left <= current_right:
            # Extend the current interval
            current_right = max(current_right, right)
        else:
            # Save the current interval and start a new one
            merged_intervals.append([current_left, current_right])
            current_left, current_right = left, right
    merged_intervals.append([current_left, current_right])
    
    # Compute the complement (safe) intervals in [interval_start, interval_end]
    safe_intervals = []
    last_end = interval_start
    tol = 1e-8
    for start, end in merged_intervals:
        if start > last_end + tol:
            safe_intervals.append([last_end, start])
        last_end = end
    if last_end < interval_end - tol:
        safe_intervals.append([last_end, interval_end])
    
    return safe_intervals


def plot_gp_1D(feas_gp, virt_gp, start_point, direction, replacement_points, plot_grad=False, line_acqf=None, vdp_function=None, it=None, t_min_diff=None, fig_num=None, use_latex=False, axes=None, plot_interactive=True, plot_legend=True):
    
    if use_latex:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    virt_gp_allpoints_x = virt_gp.train_xs.clone()
    virt_gp_allpoints_t = ((virt_gp_allpoints_x - start_point)* direction).sum(dim=1)/(direction * direction).sum()
    virt_gp_allpoints_y = virt_gp.train_ys.clone()
    feas_gp_allpoints_x = feas_gp.train_xs.clone()
    feas_gp_allpoints_t = ((feas_gp_allpoints_x - start_point)* direction).sum(dim=1)/(direction * direction).sum()
    feas_gp_allpoints_y = feas_gp.train_ys.clone()

    # Select only points on the line from start_point to start_point + direction
    orthogonal_components = virt_gp_allpoints_x - (start_point + virt_gp_allpoints_t.unsqueeze(1) * direction)
    virt_mask = (torch.sum(torch.abs(orthogonal_components), dim = 1) < 1e-10) & (virt_gp_allpoints_t <= 1) & (virt_gp_allpoints_t >= 0)
    orthogonal_components = feas_gp_allpoints_x - (start_point + feas_gp_allpoints_t.unsqueeze(1) * direction)
    feas_mask = (torch.sum(torch.abs(orthogonal_components), dim = 1) < 1e-10) & (feas_gp_allpoints_t <= 1) & (feas_gp_allpoints_t >= 0)

    virt_gp_allpoints_x = virt_gp_allpoints_x[virt_mask]
    virt_gp_allpoints_t = virt_gp_allpoints_t[virt_mask]
    virt_gp_allpoints_y = virt_gp_allpoints_y[virt_mask]
    replacement_points  = replacement_points[virt_mask]
    feas_gp_allpoints_x = feas_gp_allpoints_x[feas_mask]
    feas_gp_allpoints_t = feas_gp_allpoints_t[feas_mask]
    feas_gp_allpoints_y = feas_gp_allpoints_y[feas_mask]

    # Split the points into feasible and virtual
    mask = torch.isnan(virt_gp_allpoints_y)
    virt_gp_feaspoints_x = virt_gp_allpoints_x[~mask]
    virt_gp_feaspoints_t = virt_gp_allpoints_t[~mask]
    virt_gp_feaspoints_y = virt_gp_allpoints_y[~mask]
    virt_gp_virtpoints_x = virt_gp_allpoints_x[mask]
    virt_gp_virtpoints_t = virt_gp_allpoints_t[mask]
    virt_gp_virtpoints_y = replacement_points[mask]

    ## EVALUATE GPs ON THE LINE

    # Create a grid of t values in [0,1]
    t_plot = torch.linspace(-0.05, 1.05, 100)
    # Map t to full-dimensional points: x = start_point + t * direction
    x_line_plot = start_point + t_plot.unsqueeze(1) * direction  # shape: (t_grid, D)
    
    virt_gp.eval()
    feas_gp.eval()
    with torch.no_grad():
        virt_post = virt_gp.posterior(x_line_plot, observation_noise=True).distribution
        virt_gp_mean = virt_post.mean.squeeze()
        virt_gp_std  = virt_post.stddev.squeeze()

        feas_post = feas_gp.posterior(x_line_plot, observation_noise=True).distribution
        feas_gp_mean = feas_post.mean.squeeze()
        feas_gp_std  = feas_post.stddev.squeeze()

        if plot_grad is True:
            derivative_mean, derivative_var = virt_gp.posterior_derivative(x_line_plot)
            norm_direction = direction / direction.norm()
            derivative_mean = (derivative_mean @ norm_direction.transpose(0,1)).squeeze()
            derivative_std = torch.sqrt((norm_direction @ derivative_var @ norm_direction.transpose(0,1)).squeeze())

    # Compute the acquisition function values along the line
    if line_acqf is not None:
        acq_plot = line_acqf(t_plot.unsqueeze(1).unsqueeze(1)).detach().squeeze()

    # Compute the virtual point replacement function values along the line
    if vdp_function is not None:
        virtualpointsline_plot = vdp_function(x_line_plot).squeeze()
    
    # Plot the GPs and the acquisition function
    
    if axes is not None:
        fig_num = None
        if (line_acqf is None) and (plot_grad is False):
            ax_obj = axes
        elif (line_acqf is not None) and (plot_grad is False):
            ax_obj = axes[0]
            ax_acqf = axes[1]
        elif (line_acqf is None) and (plot_grad is True):
            ax_obj = axes[0]
            ax_grad = axes[1]
        elif (line_acqf is not None) and (plot_grad is True):
            ax_obj = axes[0]
            ax_grad = axes[1]
            ax_acqf = axes[2]

    
    if axes is None and fig_num is None:
        fig_num = 1

    # Use a fixed figure number
    if fig_num is not None and plt.fignum_exists(fig_num):
        fig = plt.figure(fig_num)
        fig.clear()
        if (line_acqf is not None) and (plot_grad is True):
            ax_obj = fig.add_subplot(311)
            ax_grad = fig.add_subplot(312, sharex=ax_obj)
            ax_acqf = fig.add_subplot(313, sharex=ax_obj)
        elif (line_acqf is not None) and (plot_grad is False):
            ax_obj = fig.add_subplot(211)
            ax_acqf = fig.add_subplot(212, sharex=ax_obj)
        elif (line_acqf is None) and (plot_grad is True):
            ax_obj = fig.add_subplot(211)
            ax_grad = fig.add_subplot(212, sharex=ax_obj)
        else:
            ax_obj = fig.add_subplot(111)
    elif fig_num is not None:
        if (line_acqf is not None) and (plot_grad is True):
            fig, (ax_obj, ax_grad, ax_acqf) = plt.subplots(3, 1, figsize=(8, 10), sharex=True, num=fig_num)
        elif (line_acqf is not None) and (plot_grad is False):
            fig, (ax_obj, ax_acqf) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, num=fig_num)
        elif (line_acqf is None) and (plot_grad is True):
            fig, (ax_obj, ax_grad) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, num=fig_num)
        else:
            fig, ax_obj = plt.subplots(1, 1, figsize=(8, 6), num=fig_num)

    plt.ion()
    ax_obj.clear()
    if line_acqf is not None:
        ax_acqf.clear()
    if plot_grad is True:
        ax_grad.clear()

    # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # colors = ["#0072BD", "#D95319"]
    colors = ["#1461f0", "#ed6a5a"]
    # colors = ["#307acf", "#eb2d3a"]
    
    if vdp_function is not None:
        ax_obj.plot(t_plot, virtualpointsline_plot, linestyle=':', color=colors[0], label="Virt points line")
    ax_obj.plot(t_plot, virt_gp_mean, color=colors[0], label="Virt GP Mean")
    ax_obj.fill_between(
        t_plot, 
        virt_gp_mean - 2.0 * virt_gp_std, 
        virt_gp_mean + 2.0 * virt_gp_std, 
        color=colors[0], alpha=0.2, label="Virt GP 95\% CI"
    )
    ax_obj.plot(t_plot, feas_gp_mean, color=colors[1], label="Feas GP Mean")
    ax_obj.fill_between(
        t_plot, 
        feas_gp_mean - 2.0 * feas_gp_std, 
        feas_gp_mean + 2.0 * feas_gp_std, 
        color=colors[1], alpha=0.2, label="Feas GP 95\% CI"
    )

    # Plot the sampled points (from history) on the t axis.
    ax_obj.scatter(feas_gp_allpoints_t, feas_gp_allpoints_y, color=colors[1], s=60, marker='o', label="Feas GP Points", zorder=5)
    ax_obj.scatter(virt_gp_feaspoints_t, virt_gp_feaspoints_y, color=colors[0], s=20, marker='o', label="Virt GP Real Points", zorder=6)
    ax_obj.scatter(virt_gp_virtpoints_t, virt_gp_virtpoints_y, color=colors[0], s=30, marker='x', label="Virt GP Virt Points", zorder=7)

    ax_obj.set_ylabel("Objective", fontsize=14)
    ax_obj.grid(True, linestyle='--', alpha=0.7)

    if (line_acqf is None) and (plot_grad is False):
        ax_obj.set_xlabel(r'$t$ along search direction', fontsize=14)
    else:
        ax_obj.tick_params(labelbottom=False)

    # Plot the gradient
    if plot_grad is True:        
        ax_grad.grid()
        ax_grad.plot(t_plot, derivative_mean, color=colors[0], label='Derivative GP Mean')
        ax_grad.fill_between(t_plot.flatten(), 
                        derivative_mean - 2*derivative_std, 
                        derivative_mean + 2*derivative_std, 
                        alpha=0.3, color=colors[0], label='Derivative GP Confidence')
        ax_grad.grid(True, linestyle='--', alpha=0.7)
        ax_grad.legend(fontsize=12)
        ax_grad.set_ylabel('Objective Gradient', fontsize=14)
        if line_acqf is None:
            ax_grad.set_xlabel(r'$t$ along search direction', fontsize=14)
        else:
            ax_grad.tick_params(labelbottom=False)
    
    # Plot the acquisition function
    if line_acqf is not None:
        t_max = 1.0
        ax_acqf.plot(t_plot, acq_plot, 'g', label="Acq. function")
        if t_min_diff is not None:
            mask = torch.any(torch.abs(t_plot.unsqueeze(1) - virt_gp_allpoints_t.unsqueeze(0)) < t_min_diff, dim=1)
            acq_plot_tmp = acq_plot.clone()
            if virt_gp_allpoints_y.isnan().any():
                t_max = torch.min(virt_gp_allpoints_t[virt_gp_allpoints_y.isnan()]).item()
            acq_plot_tmp[(t_plot < t_max) & (~mask) & (t_plot > 0)] = torch.tensor(float('nan'))
            ax_acqf.plot(t_plot, acq_plot_tmp, 'g', alpha=0.2, linewidth=10, label="Excluded zones")
        else:
            acq_plot_tmp = acq_plot.clone()
            if virt_gp_allpoints_y.isnan().any():
                t_max = torch.min(virt_gp_allpoints_t[virt_gp_allpoints_y.isnan()]).item()
            acq_plot_tmp[(t_plot < t_max) & (t_plot > 0)] = torch.tensor(float('nan'))
            ax_acqf.plot(t_plot, acq_plot_tmp, 'g', alpha=0.2, linewidth=10, label="Excluded zones")
        ax_acqf.set_ylabel("Acq. function", fontsize=14)
        ax_acqf.legend(fontsize=12)
        ax_acqf.grid(True, linestyle='--', alpha=0.7)
        ax_acqf.set_xlabel(r'$t$ along search direction', fontsize=14)

    if it is not None:
        plt.suptitle(f"Iteration {it+1}", fontsize=14)

    # Set legend
    if plot_legend:
        handles, labels = ax_obj.get_legend_handles_labels()
        if vdp_function is not None:
            new_order = [1, 2, 6, 3, 4, 5, 0, 7]
        else:
            new_order = [0, 1, 5, 2, 3, 4, 6]
        reordered_handles = [handles[i] for i in new_order]
        reordered_labels  = [labels[i]  for i in new_order]
        ax_obj.legend(
            reordered_handles,
            reordered_labels,
            loc='upper center',
            ncol=3,
            fancybox=True,
            shadow=False,
            fontsize=10,
        )

    ax_obj.set_xlim(t_plot.min().item(), t_plot.max().item())
    
    if axes is None:
        plt.tight_layout()
    
    if plot_interactive:
        plt.show()
        plt.pause(0.001)

def reconstruct_optimization_state(df, idx, cfg):
    """
    Reconstructs the GP model from the given row of the optimization data.
    """
    row = df.loc[idx]

    input_scaling_coeffs = cfg['optimizer_config']['objfcn_input_scaling_coeffs']
    if input_scaling_coeffs is None:
        input_scaling_coeffs = torch.ones_like(torch.tensor(row['sample_point']))
    sample_point = torch.tensor(row["sample_point"], dtype=input_scaling_coeffs.dtype)/input_scaling_coeffs
    observation_value = torch.tensor(row["observation"], dtype=sample_point.dtype)
    start_point_idx = torch.nonzero(torch.tensor((df.index <= idx) & (df['gd_starting_point'].apply(lambda x: len(x) > 0))))[-1].item()
    start_point = torch.tensor(df.loc[start_point_idx]["gd_starting_point"], dtype=sample_point.dtype)/input_scaling_coeffs
    
    # Get GP points
    feas_gp_train_x = torch.tensor(row["gp_train_feaspoints_x"], dtype=sample_point.dtype)/input_scaling_coeffs
    feas_gp_train_y = torch.tensor(row["gp_train_feaspoints_y"], dtype=sample_point.dtype)
    virt_gp_train_x = torch.tensor(row["gp_train_virtualpoints_x"], dtype=sample_point.dtype)/input_scaling_coeffs
    virt_gp_train_y = torch.tensor(row["gp_train_virtualpoints_y"], dtype=sample_point.dtype)
    virt_gp_train_isnan = torch.tensor(row["gp_train_virtualpoints_isnan"], dtype=torch.bool)
    
    gp_model_cls = cfg['optimizer_config']['Model']

    ## Reconstruct feasibe GP model

    feas_gp_model = gp_model_cls(D=sample_point.shape[1],  is_double=(sample_point.dtype == torch.float64),
                                 **copy.deepcopy(cfg['optimizer_config']['model_config']))
    virt_gp_model = gp_model_cls(D=sample_point.shape[1],  is_double=(sample_point.dtype == torch.float64),
                                 **copy.deepcopy(cfg['optimizer_config']['model_config']))
    if row['type'] == 'gp_line_search' or row['type'] == 'nan_gp_line_search':
        prev_feas_gp_model = gp_model_cls(D=sample_point.shape[1],  is_double=(sample_point.dtype == torch.float64),
                                    **copy.deepcopy(cfg['optimizer_config']['model_config']))
    
    # Initialize hyperparameters if provided.
    if cfg['optimizer_config']['hyperparameter_config'] is not None and "hypers" in cfg['optimizer_config']['hyperparameter_config'] and cfg['optimizer_config']['hyperparameter_config']['hypers'] is not None:
        hypers = {k: v for k, v in cfg['optimizer_config']['hyperparameter_config']["hypers"].items() if v is not None}
        feas_gp_model.initialize(**hypers)
        virt_gp_model.initialize(**hypers)
        if row['type'] == 'gp_line_search' or row['type'] == 'nan_gp_line_search':
            prev_feas_gp_model.initialize(**hypers)

    if cfg['optimizer_config']['hyperparameter_config'] is not None and "no_noise_optimization" in cfg['optimizer_config']['hyperparameter_config'] and cfg['optimizer_config']['hyperparameter_config'].get("no_noise_optimization", False):
        feas_gp_model.likelihood.noise_covar.raw_noise.requires_grad = False
        virt_gp_model.likelihood.noise_covar.raw_noise.requires_grad = False
        if row['type'] == 'gp_line_search' or row['type'] == 'nan_gp_line_search':
            prev_feas_gp_model.likelihood.noise_covar.raw_noise.requires_grad = False

    if cfg['optimizer_config']['hyperparameter_config'] is not None and "no_mean_optimization" in cfg['optimizer_config']['hyperparameter_config'] and cfg['optimizer_config']['hyperparameter_config'].get("no_mean_optimization", False):
        feas_gp_model.mean_module.raw_constant.requires_grad = False
        virt_gp_model.mean_module.raw_constant.requires_grad = False
        if row['type'] == 'gp_line_search' or row['type'] == 'nan_gp_line_search':
            prev_feas_gp_model.mean_module.raw_constant.requires_grad = False

    # Assign the hyperparameters
    gp_hypers_idx = torch.nonzero(torch.tensor((df.index <= idx) & (df['gp_lengthscales'].apply(lambda x: len(x) > 0))))[-1].item()
    feas_gp_model.covar_module.base_kernel.lengthscale = torch.tensor(df.loc[gp_hypers_idx]["gp_lengthscales"], dtype=sample_point.dtype)
    feas_gp_model.covar_module.outputscale             = torch.tensor(df.loc[gp_hypers_idx]["gp_outputscale"],  dtype=sample_point.dtype)
    feas_gp_model.likelihood.noise                     = torch.tensor(df.loc[gp_hypers_idx]["gp_noise"],        dtype=sample_point.dtype)
    feas_gp_model.mean_module.constant                 = torch.tensor(df.loc[gp_hypers_idx]["gp_mean"],         dtype=sample_point.dtype)

    # Assign the hyperparameters
    if row['type'] == 'gp_line_search' or row['type'] == 'nan_gp_line_search':
        prev_row = df.loc[idx-1]
        prev_feas_gp_model.covar_module.base_kernel.lengthscale = torch.tensor(prev_row["gp_lengthscales"], dtype=sample_point.dtype)
        prev_feas_gp_model.covar_module.outputscale             = torch.tensor(prev_row["gp_outputscale"],  dtype=sample_point.dtype)
        prev_feas_gp_model.likelihood.noise                     = torch.tensor(prev_row["gp_noise"],        dtype=sample_point.dtype)
        prev_feas_gp_model.mean_module.constant                 = torch.tensor(prev_row["gp_mean"],         dtype=sample_point.dtype)

    # Copy the hyperparameters to the virtual GP model
    virt_gp_model.load_state_dict(feas_gp_model.state_dict())

    if any(key.startswith("in_transform_") for key in row.index):
        keys = [key for key in row.index if key.startswith("in_transform_")]
        mask = (df.index <= idx) & (df[keys[0]].apply(lambda x: len(x) > 0))

        if any(mask):
            in_dict_idx = torch.nonzero(torch.tensor(mask, dtype=torch.bool))[-1].item()
            in_dict = OrderedDict(
                (key[len("in_transform_"):], value if torch.is_tensor(value) else torch.tensor(value))
                for key, value in df.iloc[in_dict_idx].items() if key.startswith("in_transform_")
            )
            feas_gp_model.input_transform.load_state_dict(in_dict)
            virt_gp_model.input_transform.load_state_dict(in_dict)
            if row['type'] == 'gp_line_search' or row['type'] == 'nan_gp_line_search':
                prev_feas_gp_model.input_transform.load_state_dict(in_dict)

    if any(key.startswith("out_transform_") for key in row.index):
        keys = [key for key in row.index if key.startswith("out_transform_")]
        mask = (df.index <= idx) & (df[keys[0]].apply(lambda x: len(x) > 0))

        if any(mask):
            out_dict_idx = torch.nonzero(torch.tensor(mask, dtype=torch.bool))[-1].item()
            out_dict = OrderedDict(
                (key[len("out_transform_"):], value if torch.is_tensor(value) else torch.tensor(value))
                for key, value in df.iloc[out_dict_idx].items() if key.startswith("out_transform_")
            )
            feas_gp_model.outcome_transform.load_state_dict(out_dict)
            virt_gp_model.outcome_transform.load_state_dict(out_dict)
            if row['type'] == 'gp_line_search' or row['type'] == 'nan_gp_line_search':
                prev_feas_gp_model.outcome_transform.load_state_dict(out_dict)

    # Append training data to GP
    feas_gp_model.append_train_data(feas_gp_train_x, feas_gp_train_y, unlimited=True, update_inout_transforms=False)
    if row['type'] == 'gp_line_search' or row['type'] == 'nan_gp_line_search':
        prev_feas_gp_model.append_train_data(feas_gp_train_x, feas_gp_train_y, unlimited=True, update_inout_transforms=False)

    # Update GP prediction strategy
    feas_gp_model.posterior(sample_point)
    if row['type'] == 'gp_line_search' or row['type'] == 'nan_gp_line_search':
        prev_feas_gp_model.posterior(sample_point)

    ## Recontruct virtualpoint replacement function
    vdp_config = cfg['optimizer_config']['vdp_config']
    if vdp_config is not None and vdp_config.get("active", False):
        if vdp_config.get("method", None) == "ucb":
            if row['type'] == 'gp_line_search' or row['type'] == 'nan_gp_line_search':
                vdp_function = lambda x, x_curr: VDPUCB(x, x_curr, vdp_config["beta"], prev_feas_gp_model)
            else:
                vdp_function = lambda x, x_curr: VDPUCB(x, x_curr, vdp_config["beta"], feas_gp_model)
        else:
            vdp_function = lambda x, _: VDPNaN(x)
    else:
        vdp_function = lambda x, _: VDPNaN(x)

    if row['type'] == 'gp_line_search' or row['type'] == 'nan_gp_line_search':
        direction = torch.tensor(row["gradient"], dtype=sample_point.dtype)/input_scaling_coeffs
        virt_gp_train_t = ((virt_gp_train_x - start_point)* direction).sum(dim=1)/(direction * direction).sum()
        orthogonal_components = virt_gp_train_x - (start_point + virt_gp_train_t.unsqueeze(1) * direction)
        direction_mask = (torch.sum(torch.abs(orthogonal_components), dim = 1) < 1e-10) & (virt_gp_train_t <= 1) & (virt_gp_train_t >= 0)
        vdp_sel_point, _ = get_best_point(sample_points=virt_gp_train_x[direction_mask], observations=virt_gp_train_y[direction_mask])
        vdp_function_handle = lambda x: vdp_function(x, vdp_sel_point)
    elif  row['type'] == 'evaluation':
        # Look backwards from the current index until a non-'evaluation' row is found
        last_non_eval_type = None
        for i in range(idx - 1, -1, -1):
            if df.loc[i]['type'] != "evaluation":
                last_non_eval_type = df.loc[i]['type']
                break
        if last_non_eval_type in ["gp_line_search", "nan_gp_line_search"]:
            direction = torch.tensor(df.iloc[i]["gradient"], dtype=sample_point.dtype)/input_scaling_coeffs
            virt_gp_train_t = ((virt_gp_train_x - start_point)* direction).sum(dim=1)/(direction * direction).sum()
            orthogonal_components = virt_gp_train_x - (start_point + virt_gp_train_t.unsqueeze(1) * direction)
            direction_mask = (torch.sum(torch.abs(orthogonal_components), dim = 1) < 1e-10) & (virt_gp_train_t <= 1) & (virt_gp_train_t >= 0)
            best_point, best_point_value = get_best_point(sample_points=virt_gp_train_x[direction_mask], observations=virt_gp_train_y[direction_mask])
            vdp_function_handle = lambda x: vdp_function(x, best_point)
        else:
            vdp_sel_point = sample_point.detach() if not torch.isnan(observation_value) else start_point
            vdp_function_handle = lambda x: vdp_function(x, vdp_sel_point)
    else:
        vdp_sel_point = sample_point.detach() if not torch.isnan(observation_value) else start_point
        vdp_function_handle = lambda x: vdp_function(x, vdp_sel_point)

    # Append virtual GP training data
    tmp_virt_gp_train_y = virt_gp_train_y.clone()
    tmp_virt_gp_train_y[virt_gp_train_isnan] = torch.tensor(float("nan"), dtype=sample_point.dtype)
    virt_gp_model.append_train_data_vdp(virt_gp_train_x, tmp_virt_gp_train_y, vdp_function_handle, unlimited=True, update_inout_transforms=False)

    # Update GP prediction strategy
    virt_gp_model.posterior(sample_point)  

    # Append training data to GP
    return feas_gp_model, virt_gp_model, vdp_function_handle


def compute_numerical_derivative(X, y):
    """Compute the numerical derivative using finite differences in torch."""
    # Flatten the tensors
    X_flat = X.flatten()
    y_flat = y.flatten()
    
    # Calculate differences in y and X values
    # If using PyTorch 1.8 or newer, you can use torch.diff:
    dy = torch.diff(y_flat)
    dx = torch.diff(X_flat)
    
    # Alternatively, if torch.diff isn't available, use slicing:
    # dy = y_flat[1:] - y_flat[:-1]
    # dx = X_flat[1:] - X_flat[:-1]
    
    # Compute derivative as dy/dx
    derivative = dy / dx
    
    # Append the last derivative value to keep the same length
    derivative = torch.cat((derivative, derivative[-1].unsqueeze(0)))
    return derivative