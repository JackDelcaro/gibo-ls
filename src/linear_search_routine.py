import torch
from typing import Callable, Tuple, Dict
import botorch
import time
from src.acquisition_function import optimize_acqf_botorch
from src.utils import get_best_point, compute_safe_intervals, plot_gp_1D
import inspect
import copy



def gp_line_search(
    objective_fn: Callable[[torch.Tensor], torch.Tensor],
    start_point: torch.Tensor,
    start_point_value: torch.Tensor,
    end_point: torch.Tensor,
    end_point_value: torch.Tensor,
    caller,  # an instance of VDPBayesianGradientAscent
    max_evaluations: int = 8,
    n_seed: int = 2,
    t_min_diff: float = None,
    acquisition_function: botorch.acquisition.AcquisitionFunction = botorch.acquisition.analytic.UpperConfidenceBound,
    acqf_config: Dict = None,
    optimize_acqf: Callable = optimize_acqf_botorch,
    optimize_acqf_config: Dict[str, torch.Tensor] = None,
    optimize_hyperparameters: bool = False,
    verbose: bool = False,
    enable_plotting: bool = False,
    run_type: str = "gp_line_search",
) -> Tuple[torch.Tensor, torch.Tensor]:

    # Initialize the direction vector.
    direction = end_point - start_point

    # Current 1D data
    gp_1D_data_t = torch.tensor([0.0, 1.0], dtype=start_point.dtype)
    gp_1D_data_x = torch.vstack([start_point, end_point])
    gp_1D_data_y = torch.vstack([start_point_value, end_point_value]).squeeze()

    sig = inspect.signature(acquisition_function)
    if "best_f" in sig.parameters and sig.parameters["best_f"].default is inspect.Parameter.empty:
        update_best_f = True
    else:
        update_best_f = False
    
    if acqf_config is None:
        acqf_config = {}
    if optimize_acqf_config is None:
        optimize_acqf_config = {}

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

    # --- Build the acquisition function along the line ---
    # Create the original acquisition function using the current virtual GP model.
    acq_func_original = acquisition_function(
        caller.virtualdatapoints_model,
        **(acqf_config if acqf_config is not None else {})
    )
    # Wrap it in our line acquisition function.
    line_acq_func = LineAcquisitionFunction(acq_func_original, start_point, direction)

    t_seed = torch.linspace(0.0, 1.0, n_seed + 2, dtype=start_point.dtype)[1:-1].unsqueeze(1)
    t_max = 1.0

    # Loop over the maximum number of line search iterations.
    for i in range(max_evaluations):
        algo_start_time = time.time()

        # --- Update best_f for the acquisition function if needed ---
        if update_best_f:
            if gp_1D_data_y.shape[0] == 0:
                raise ValueError("BO cannot be initialized with no seed points when best_f is required. Consider using generate_initial_data.")
            acqf_config["best_f"] = gp_1D_data_y[~torch.isnan(gp_1D_data_y) & ~torch.isinf(gp_1D_data_y)].max()

        # Define 1D bounds for t.
        t_bounds = torch.tensor([[0.0], [t_max]], dtype=start_point.dtype, device=start_point.device)

        # Optimize the line acquisition function.        
        acqf_optimize_start_time = time.time()
        if i < n_seed:
            t_sample, acqf_value = t_seed[i].unsqueeze(0), line_acq_func(t_seed[i].squeeze()).detach()
            acqf_value = acqf_value.squeeze()
        elif t_min_diff is not None:
            safe_intervals = compute_safe_intervals(gp_1D_data_t, t_min_diff, interval_end=t_max)
            if safe_intervals: # if there exist at least one interval which points are at least t_min_diff distant from any point in gp_1D_data_t
                safe_t_samples = torch.empty([len(safe_intervals), 1], dtype=start_point.dtype)
                safe_acqf_values = torch.empty(len(safe_intervals), dtype=start_point.dtype)
                for ii, interval in enumerate(safe_intervals):
                    safe_t_samples[ii], safe_acqf_values[ii] = optimize_acqf(line_acq_func, bounds=torch.tensor(interval).unsqueeze(1), **optimize_acqf_config)
                best_idx = torch.argmax(safe_acqf_values)
                t_sample, acqf_value = safe_t_samples[best_idx].unsqueeze(0), safe_acqf_values[best_idx]
            else:
                if verbose:
                    print(f"    [Line search iter {i+1}]: could not find safe intervals, resorting to classical acquisition function maximization")
                t_sample, acqf_value = optimize_acqf(line_acq_func, t_bounds, **optimize_acqf_config)
        else:
            t_sample, acqf_value = optimize_acqf(line_acq_func, t_bounds, **optimize_acqf_config)
        acqf_optimize_elapsed_time = time.time() - acqf_optimize_start_time

        # If sample point is at the start or end of the line, return the best point and end linear subroutine.
        if t_sample.squeeze().item() == 0.0 or t_sample.squeeze().item() == t_max:
            if verbose:
                print(f"    [Line search iter {i+1}]: acquisition function maximization leads to {'start_point' if t_sample.squeeze().item() == 0.0 else 'end_point'}. Returning to main optimization.")
            best_point, best_point_value, best_point_idx  = get_best_point(sample_points=gp_1D_data_x, observations=gp_1D_data_y, return_index=True)
            return best_point, best_point_value, gp_1D_data_t[best_point_idx].clone()
        
        # Map the optimized t to a full-dimensional sample point.
        sample_point = start_point + t_sample * direction

        # --- Evaluate the objective at the new candidate ---
        obj_eval_start_time = time.time()
        observation_value = objective_fn(sample_point * caller.objfcn_input_scaling_coeffs)
        obj_eval_elapsed_time = time.time() - obj_eval_start_time
        
        if verbose:
            print(f"    [Line search iter {i+1}]: t: {t_sample.item(): .2g}, objective evaluation: {observation_value.item(): .2g}")

        # Get the best non-NaN point
        gp_1D_data_t = torch.cat([t_sample.squeeze(0), gp_1D_data_t])
        gp_1D_data_x = torch.vstack([sample_point, gp_1D_data_x])
        gp_1D_data_y = torch.cat([observation_value, gp_1D_data_y])
        best_point, best_point_value, best_point_idx = get_best_point(sample_points=gp_1D_data_x, observations=gp_1D_data_y, return_index=True)
        best_point_t = gp_1D_data_t[best_point_idx].clone()

        if gp_1D_data_y.isnan().any():
            t_max = torch.min(gp_1D_data_t[gp_1D_data_y.isnan()]).item()

        # --- Update the GP training data in both models ---
        gp_feastraining_x, gp_feastraining_y = caller.onlyfeaspoints_model.append_train_data(sample_point, observation_value, unlimited=True, update_inout_transforms=False)
        gp_virtualtraining_x, gp_virtualtraining_y = caller.virtualdatapoints_model.append_train_data_vdp(
            sample_point,
            observation_value,
            lambda x: caller.vdp_function(x, best_point),
            unlimited=True,
            update_inout_transforms=False
        )

        if enable_plotting:
            plot_gp_1D(caller.onlyfeaspoints_model, caller.virtualdatapoints_model, start_point, direction, gp_virtualtraining_y, line_acqf=line_acq_func, vdp_function=lambda x: caller.vdp_function(x, best_point), it=i, t_min_diff=t_min_diff, plot_grad=True)

        # --- GP Hyperparameter Optimization (if enabled) ---
        if optimize_hyperparameters and caller.optimize_hyperparameters and caller.virtualdatapoints_model.N >= caller.num_samples_before_gp_training:
            
            in_dict, out_dict = caller.onlyfeaspoints_model.update_inout_transforms()
            _, _ = caller.virtualdatapoints_model.update_inout_transforms_vdp(input_dict=in_dict, outcome_dict=out_dict)
            update_inout_trans_history = True
            
            mll = caller.marginal_likelihood(
                caller.onlyfeaspoints_model.likelihood, caller.onlyfeaspoints_model
            )
            gp_hypers_fitting_start_time = time.time()
            botorch.fit.fit_gpytorch_model(mll)
            gp_hypers_fitting_time = time.time() - gp_hypers_fitting_start_time

            # Synchronize the two models.
            caller.virtualdatapoints_model.load_state_dict(caller.onlyfeaspoints_model.state_dict())
            caller.virtualdatapoints_model.eval()
            caller.onlyfeaspoints_model.eval()

            if verbose:
                print(f"    [Line search iter {i+1}] GP hypers optimized in {gp_hypers_fitting_time:.2f} s.")
        else:
            update_inout_trans_history = False
            gp_hypers_fitting_time = []

        gp_update_start_time = time.time()
        caller.virtualdatapoints_model.posterior(sample_point.clone().detach())
        gp_update_elapsed_time = time.time() - gp_update_start_time

        # --- Log the iteration in the caller's history ---
        history_entry = {
            "type": run_type,
            "iteration_number": caller.iteration_number,
            "evaluation_number": objective_fn._calls,
            "sample_point": sample_point.clone() * caller.objfcn_input_scaling_coeffs,
            "observation": observation_value.clone(),
            "acqf_opt_time": acqf_optimize_elapsed_time,
            "gp_update_time": gp_update_elapsed_time,
            "acquisition_function": acqf_value.clone(),
            "objective_eval_time": obj_eval_elapsed_time,
            "optimization_algo_time": time.time() - algo_start_time - obj_eval_elapsed_time,
            "gp_hyperparam_tuning_time": gp_hypers_fitting_time,
            "gp_lengthscales": caller.onlyfeaspoints_model.covar_module.base_kernel.lengthscale.detach().flatten().clone(),
            "gp_outputscale": caller.onlyfeaspoints_model.covar_module.outputscale.detach().clone(),
            "gp_noise": caller.onlyfeaspoints_model.likelihood.noise.detach().clone(),
            "gp_mean": caller.onlyfeaspoints_model.mean_module.constant.detach().clone(),
            "gd_starting_point": start_point.clone() * caller.objfcn_input_scaling_coeffs,
            "gradient": direction.clone() * caller.objfcn_input_scaling_coeffs,
            "gp_train_feaspoints_x": gp_feastraining_x * caller.objfcn_input_scaling_coeffs,
            "gp_train_feaspoints_y": gp_feastraining_y,
            "gp_train_virtualpoints_x": gp_virtualtraining_x * caller.objfcn_input_scaling_coeffs,
            "gp_train_virtualpoints_y": gp_virtualtraining_y,
            "gp_train_virtualpoints_isnan": caller.virtualdatapoints_model.train_ys.clone().isnan(),
        }
        if update_inout_trans_history and hasattr(caller.onlyfeaspoints_model, "input_transform"):
            in_dict = copy.deepcopy(caller.onlyfeaspoints_model.input_transform.state_dict())
            history_entry.update({
                "in_transform_" + key: value 
                for key, value in in_dict.items()
            })
        if update_inout_trans_history and hasattr(caller.onlyfeaspoints_model, "outcome_transform"):
            out_dict = copy.deepcopy(caller.onlyfeaspoints_model.outcome_transform.state_dict())
            history_entry.update({
                "out_transform_" + key: value 
                for key, value in out_dict.items()
            })        
        caller.history.append(history_entry)
    
    return best_point, best_point_value, best_point_t
