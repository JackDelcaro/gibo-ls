# ------------------------------------------------------------------------------
# Optimizer Loop with Call Counting and Iteration Control
# 
# This script defines a loop function that orchestrates an optimization process
# using a variety of optimizer algorithms. The function can terminate based on
# either a fixed number of iterations or a maximum number of objective function
# calls. The script includes a decorator to count the number of times the 
# objective function is called, allowing for detailed logging and stopping 
# criteria based on sample complexity.
# 
# Main Components:
# - `call_counter`: A decorator to count how many times a function is called.
# - `loop`: The main function that performs the optimization.
#
# The optimization process can be configured using the `optimizer_config` 
# dictionary and allows for flexible stopping criteria.
# ------------------------------------------------------------------------------

from typing import Optional, Dict, Callable, Union, Tuple

import numpy as np
import torch
import time
import scipy.io

# from src.environment_api import EnvironmentObjective
from src.optimizers import (
    VDPBayesianGradientAscent,
)


def call_counter(func) -> Callable:
    """Decorate a function and "substitute" it function with a wrapper that
        count its calls.

    Args:
        func: Function which calls should be counted.

    Returns:
        Helper function which has attributes _calls and _func.
    """

    def helper(*args, **kwargs):
        num_rows = 1  # Default increment
        if args:
            thetas = args[0]
            if isinstance(thetas, torch.Tensor):
                # If a 1D tensor, treat it as a single row; otherwise, use the first dimension.
                num_rows = 1 if thetas.dim() == 1 else thetas.shape[0]
            elif isinstance(thetas, np.ndarray):
                # If a 1D array, treat it as a single row; otherwise, use the first dimension.
                num_rows = 1 if thetas.ndim == 1 else thetas.shape[0]

        helper._calls += num_rows
        return func(*args, **kwargs)

    # Attach the original function to the helper for reference
    helper._func = func
    # Initialize the call counter to zero
    helper._calls = 0
    # Return the wrapped function
    return helper


def loop(
    params_init: torch.tensor,
    max_iterations: Optional[int],
    max_objective_calls: Optional[int],
    objective: Callable[[torch.Tensor], torch.Tensor],
    Optimizer: Union[VDPBayesianGradientAscent],
    optimizer_config: Optional[Dict],
    verbose=False,
    save_process=False,
    out_filename=None,
) -> Tuple[list, list, list]:
    """Connects parameters with objective and optimizer.

    Args:
        params_init:
        max_iterations: Stopping criterion for optimization after maximum of
            iterations (update steps of parameters).
        max_objective_calls: Stopping criterion for optimization after maximum
            of function calls of objective function.
        objective: Objective function to be optimized (search for maximum).
        Optimizer: One of the implemented optimizers.
        optimizer_config: Configuration dictionary for optimizer.
        verbose: If True an output is logged.

    Returns:
        Tuple of
            - list of parameters history
            - list of objective function calls in every iteration
            - list of observation history
    """
    def get_iteration_data():

        # Determine the union of keys in order of appearance
        list_of_dicts = optimizer.history
        ordered_keys = []
        for d in list_of_dicts:
            for key in d.keys():
                if key not in ordered_keys:
                    ordered_keys.append(key)

        # Create a new list of dictionaries with missing keys filled with [] and convert torch tensors to numpy arrays
        filled_list = []
        for d in list_of_dicts:
            new_dict = {}
            for key in ordered_keys:
                if key in d:
                    value = d[key]
                    # Convert torch tensors to numpy arrays
                    if isinstance(value, torch.Tensor):
                        value = value.detach().cpu().numpy()
                    new_dict[key] = value
                else:
                    new_dict[key] = []  # fill missing keys with an empty list
            filled_list.append(new_dict)

        return filled_list

    def save_iteration_data():

        optimization_data = {
            'iteration_data': iteration_data,
        }

        with open(out_filename, "wb") as file:
            scipy.io.savemat(file, {'optimization_data': optimization_data})
            file.flush()  # Ensure the file is flushed
            file.close()  # Explicitly close the file

    start_time = time.time()

    # List to track the number of objective calls at each iteration
    calls_in_iteration = []
    
    # Wrap the objective function with the call counter decorator
    objective_w_counter = call_counter(objective)
    
    # Initialize the optimizer with the provided parameters and configuration
    optimizer = Optimizer(params_init, objective_w_counter, **optimizer_config)

    if max_iterations:
        # If max_iterations is specified, run the loop for the given number of iterations
        for iteration in range(max_iterations):
            if verbose:
                print(f"--- Iteration {iteration+1} ---")
            optimizer()   # Perform one optimization iteration
            # Record the number of objective calls after this iteration
            calls_in_iteration.append(objective_w_counter._calls)
            
    elif max_objective_calls:
        # If max_objective_calls is specified, run until this number of calls is reached
        iteration = 0
        while objective_w_counter._calls < max_objective_calls:
            if verbose:
                print(
                    f"--- Iteration {iteration+1} ({objective_w_counter._calls} objective calls so far) ---"
                )
            optimizer() # Perform one optimization iteration
            iteration += 1
            # Record the number of objective calls after this iteration
            calls_in_iteration.append(objective_w_counter._calls)
            
            if save_process:
                iteration_data = get_iteration_data()
                save_iteration_data()
            
            if optimizer.terminal_condition_reached:
                break
    
    
    elapsed_time = time.time() - start_time

    if verbose:
        # Print the total number of objective function calls after completion
        print(
            f"\nObjective function was called {objective_w_counter._calls} times (sample complexity).\n"
        )

    iteration_data = get_iteration_data()
    optimization_data = {
        'iteration_data': iteration_data,
        'optimization_time': elapsed_time,
    }

    if save_process:
        save_iteration_data()

    # Return the history of parameters, objective calls, and observations
    return optimization_data