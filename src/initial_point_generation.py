import math
import torch
from typing import Callable, Tuple
from torch.quasirandom import SobolEngine

def lhs_in_bounds(bounds: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Generate initial conditions using Latin Hypercube Sampling within given bounds.

    Args:
        bounds: A 2 x D tensor of lower and upper bounds for each dimension.
        num_samples: Number of samples to generate.

    Returns:
        A num_samples x D tensor of initial points sampled using LHS.
    """
    dim = bounds.size(-1)  # Get the dimensionality
    # Using SobolEngine (scrambled) as a proxy for LHS
    sobol = SobolEngine(dimension=dim, scramble=True)
    lhs_samples = sobol.draw(num_samples)  # Generate samples in [0, 1]
    lower_bounds, upper_bounds = bounds[0], bounds[1]
    scaled_samples = lower_bounds + (upper_bounds - lower_bounds) * lhs_samples
    return scaled_samples

def generate_init_lhs_points(objective: Callable[[torch.Tensor], torch.Tensor],
                             bounds: torch.Tensor,
                             num_samples: int,
                             batch_size: int = 1,
                             verbose: bool = False,
                             init_points_fixed: torch.Tensor = torch.empty(0)) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate initial points using LHS within the specified bounds,
    with the option to include fixed initial points.
    
    Parameters:
      objective: A callable that evaluates a tensor of points.
      bounds: A Dx2 tensor specifying the lower and upper bounds.
      num_samples: Total number of points to generate.
      batch_size: Batch size for evaluating the objective.
      verbose: If True, print progress messages.
      init_points_fixed: A tensor of shape (N, D) of fixed initial points.
                         If empty, all points are generated via LHS.
    
    Returns:
      A tuple (init_points, init_observations) where:
        init_points: A (num_samples x D) tensor of points.
        init_observations: A tensor of evaluations of the objective at these points.
    """

    # Check if fixed points have been provided.
    if init_points_fixed.numel() == 0:
        # No fixed points provided: generate all points using LHS.
        init_points = lhs_in_bounds(bounds, num_samples)
    else:
        fixed_num = init_points_fixed.shape[0]
        if fixed_num > num_samples:
            raise ValueError("Number of fixed initial points exceeds num_samples.")
        # Generate only the remaining points via LHS.
        lhs_points = lhs_in_bounds(bounds, num_samples - fixed_num)
        # Concatenate fixed points with the generated LHS points.
        init_points = torch.cat([init_points_fixed, lhs_points], dim=0)
    
    init_observations = torch.empty(num_samples)

    # Process the points in batches.
    num_batches = math.ceil(num_samples / batch_size)
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, num_samples)
        if verbose:
            print(f"    [Initial Points]: Sampling from point {start+1} to {end}")
        batch_points = init_points[start:end]
        # Evaluate the objective on the entire batch at once.
        init_observations[start:end] = objective(batch_points)
    
    return init_points, init_observations