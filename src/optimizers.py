from typing import Tuple, Dict, Callable, Iterator, Union, Optional, List, Any

from abc import ABC, abstractmethod

import numpy as np
import time
import torch
import gpytorch
import botorch
import copy

from src.model import DerivativeExactGPSEModel
from src.acquisition_function import GradientInformation
from src.model import DerivativeExactGPSEModel
from src.vdp_model import VDPNaN, VDPUCB
from src.utils import get_best_point, get_history_field
from scipy.optimize import minimize

import warnings
from linear_operator.utils.warnings import NumericalWarning
from botorch.exceptions.errors import ModelFittingError

class AbstractOptimizer(ABC):
    """Abstract optimizer class.

    Sets a default optimizer interface.

    Attributes:
        params_init: Starting parameter configuration for the optimization.
        objective: Objective to optimize, can be a function or a
            EnvironmentObjective.
        param_args_ignore: Which parameters should not be optimized.
        optimizer_config: Configuration file for the optimizer.
    """

    def __init__(
        self,
        params_init: torch.Tensor,
        objective: Callable[[torch.Tensor], torch.Tensor],
        param_args_ignore: List[int] = None,
        **optimizer_config: Dict,
    ):
        """Inits the abstract optimizer."""
        # Optionally add batchsize to parameters.
        if len(params_init.shape) == 1:
            params_init = params_init.reshape(1, -1)
        self.sample_point = params_init.clone()
        self.param_args_ignore = param_args_ignore
        self.objective = objective
        self.terminal_condition_reached = False

    def __call__(self):
        """Call method of optimizers."""
        self.step()

    @abstractmethod
    def step(self) -> None:
        """One parameter update step."""
        pass


class VDPBayesianGradientAscent(AbstractOptimizer):
    """Optimizer for Bayesian gradient ascent.

    Also called gradient informative Bayesian optimization (GIBO).

    Attributes:
        params_init: Starting parameter configuration for the optimization.
        objective: Objective to optimize, can be a function or a
            EnvironmentObjective.
        max_samples_per_iteration: Maximum number of samples that are supplied
            by acquisition function before updating the parameters.
        OptimizerTorch: Torch optimizer to update parameters, e.g. SGD or Adam. --------CHECK: algorithm used for the gradient step
        optimizer_torch_config: Configuration dictionary for torch optimizer.
        lr_schedular: Optional learning rate schedular, mapping iterations to
            learning rates. --------CHECK: this is a dictionary like {0: 0.1, 10: 0.01, 20: 0.001} where {iteration:learningrate, ...}
        Model: Gaussian process model, has to supply Jacobian information.
        model_config: Configuration dictionary for the Gaussian process model.
        hyperparameter_config: Configuration dictionary for hyperparameters of
            Gaussian process model.
        optimize_acqf: Function that optimizes the acquisition function.
        optimize_acqf_config: Configuration dictionary for optimization of
            acquisition function.
        bounds: Search bounds for optimization of acquisition function.
        delta: Defines search bounds for optimization of acquisition function
            indirectly by defining it within a distance of delta from the
            current parameter constellation.
        epsilon_diff_acq_value: Difference between acquisition values. Sampling
            of new data points with acquisition function stops when threshold of
            this epsilon value is reached.
        generate_initial_data: Function to generate initial data for Gaussian
            process model.
        normalize_gradient: Algorithmic extension, normalize the gradient
            estimate with its L2 norm and scale the remaining gradient direction
            with the trace of the lengthscale matrix.
        standard_deviation_scaling: Scale gradient with its variance, inspired
            by an augmentation of random search.
        verbose: If True an output is logged.
    """

    def __init__(
        self,
        params_init: torch.Tensor,
        objective: Callable[[torch.Tensor], torch.Tensor],
        max_samples_per_iteration: int,
        OptimizerTorch: torch.optim.Optimizer,
        optimizer_torch_config: Optional[Dict],
        lr_schedular: Optional[Dict[int, float]],
        Model: DerivativeExactGPSEModel,
        model_config: Optional[
            Dict[
                str,
                Union[int, float, torch.nn.Module, gpytorch.priors.Prior],
            ]
        ],
        hyperparameter_config: Optional[Dict[str, bool]],
        optimize_acqf: Callable[[GradientInformation, torch.Tensor], torch.Tensor],
        optimize_acqf_config: Dict[str, Union[torch.Tensor, int, float]],
        epsilon_diff_acq_value: Optional[Union[int, float]],
        generate_initial_data: Optional[
            Callable[[Callable[[torch.Tensor], torch.Tensor]], torch.Tensor]
        ] = None,
        generate_initial_data_config: Dict = {},
        marginal_likelihood: Union[gpytorch.mlls.ExactMarginalLogLikelihood, gpytorch.mlls.LeaveOneOutPseudoLikelihood] = None,
        vdp_config: Optional[Dict] = None,
        normalize_gradient: bool = False,
        standard_deviation_scaling: bool = False,
        objfcn_input_scaling_coeffs: torch.Tensor = None,
        opt_bounds: torch.Tensor = None,
        num_samples_before_gp_training: int = None,
        verbose: bool = True,
        nan_it_routine = None,
        nan_it_routine_config: Optional[Dict] = None,
        lin_search_routine = None,
        lin_search_routine_config: Optional[Dict] = {},
        terminal_condition: Dict = {},
    ) -> None:
        """Inits optimizer Bayesian gradient ascent with Virtual Data Points."""
        super(VDPBayesianGradientAscent, self).__init__(params_init, objective)

        # Store the dimensionality of the parameters
        self.D = self.sample_point.shape[-1]

        # Store whether to normalize the gradient and scale it by standard deviation
        self.normalize_gradient = normalize_gradient
        self.standard_deviation_scaling = standard_deviation_scaling
        if objfcn_input_scaling_coeffs is not None:
            if torch.is_tensor(objfcn_input_scaling_coeffs):
                self.objfcn_input_scaling_coeffs = objfcn_input_scaling_coeffs.view([1, objfcn_input_scaling_coeffs.numel()])
            else:
                self.objfcn_input_scaling_coeffs = torch.tensor(objfcn_input_scaling_coeffs, dtype=self.sample_point.dtype)
        else:
            # self.normalize_costfcn = False
            self.objfcn_input_scaling_coeffs = torch.ones([1, self.D], dtype=self.sample_point.dtype)
        # Rescale initial point
        self.sample_point = self.sample_point / self.objfcn_input_scaling_coeffs

        self.use_opt_bounds = False
        if opt_bounds is not None:
            self.use_opt_bounds = True
            if not torch.is_tensor(opt_bounds):
                self.opt_bounds = torch.tensor(opt_bounds, dtype=self.sample_point.dtype)
            else:
                self.opt_bounds = opt_bounds
            self.opt_bounds = self.opt_bounds / self.objfcn_input_scaling_coeffs

        # Initialize the gradient tensor for parameters
        self.sample_point.grad = torch.zeros_like(self.sample_point)        

        # Initialize the PyTorch optimizer with the given parameters and configuration
        # The optimizer will be used to perform the gradient step of the self.sample_point value (self.sample_point = self.sample_point - lr*gradient)
        self.optimizer_torch = OptimizerTorch([self.sample_point], **optimizer_torch_config)
        
        # Store the learning rate scheduler configuration
        if not isinstance(lr_schedular, dict) and hasattr(lr_schedular, '__len__') and len(lr_schedular) == 2:
            lr_dict = {int(key): float(value) for key, value in zip(lr_schedular[0], lr_schedular[1])}
            lr_schedular = lr_dict
        self.lr_schedular = lr_schedular

        # Store the threshold for acquisition value differences to control sampling
        self.epsilon_diff_acq_value = epsilon_diff_acq_value
        
        # Initialize the Gaussian process model with the configuration
        self.onlyfeaspoints_model    = Model(self.D, is_double=params_init.dtype==torch.float64, **copy.deepcopy(model_config))
        self.virtualdatapoints_model = Model(self.D, is_double=params_init.dtype==torch.float64, **copy.deepcopy(model_config))
        
        # Initialize model hyperparameters if provided
        if hyperparameter_config["hypers"]:
            hypers = dict(
                filter(
                    lambda item: item[1] is not None,
                    hyperparameter_config["hypers"].items(),
                )
            )
            self.onlyfeaspoints_model.initialize(**hypers)
            self.virtualdatapoints_model.initialize(**hypers)

        if hyperparameter_config is not None and "no_noise_optimization" in hyperparameter_config and hyperparameter_config.get("no_noise_optimization", False):
            self.onlyfeaspoints_model.likelihood.noise_covar.raw_noise.requires_grad = False
            self.virtualdatapoints_model.likelihood.noise_covar.raw_noise.requires_grad = False

        if hyperparameter_config is not None and "no_mean_optimization" in hyperparameter_config and hyperparameter_config.get("no_mean_optimization", False):
            self.onlyfeaspoints_model.mean_module.raw_constant.requires_grad = False
            self.virtualdatapoints_model.mean_module.raw_constant.requires_grad = False

        # Store whether to optimize hyperparameters during training
        self.optimize_hyperparameters = hyperparameter_config["optimize_hyperparameters"]

        # Initialize acquisition function and its optimizer with configuration
        self.acquisition_fcn = GradientInformation(self.virtualdatapoints_model)
        self.optimize_acqf = lambda acqf, acqf_bounds: optimize_acqf(
            acqf,
            acqf_bounds,
            **{k: v for k, v in optimize_acqf_config.items() if k not in {'bounds', 'delta'}}
        )

        # Set the acqf_bounds for acquisition function optimization
        self.acqf_bounds = optimize_acqf_config['bounds'] if 'bounds' in optimize_acqf_config else None
        # Delta defines the search bounds relative to current parameters if bounds are not explicitly set
        self.delta = optimize_acqf_config['delta'] if 'delta' in optimize_acqf_config else None
        self.update_acqf_bounds = (self.acqf_bounds is None) or (self.acqf_bounds['lower_bound'] is None or self.acqf_bounds['upper_bound'] is None)
        
        # Maximum number of samples per iteration for the acquisition function
        self.max_samples_per_iteration = max_samples_per_iteration
        # Verbose flag to control logging output
        self.verbose = verbose

        # Initialize virtual data point function
        if vdp_config is not None and 'active' in vdp_config and vdp_config['active'] == True:
            if 'method' in vdp_config and vdp_config['method'] == 'ucb':
                self.vdp_function = lambda x, x_curr_iteration: VDPUCB(x, x_curr_iteration, vdp_config['beta'], self.onlyfeaspoints_model)
            else:
                self.vdp_function = lambda x: VDPNaN(x)
        else:
            self.vdp_function = lambda x: VDPNaN(x)

        if num_samples_before_gp_training is None:
            self.num_samples_before_gp_training = (self.max_samples_per_iteration+1)*3
        else:
            self.num_samples_before_gp_training = num_samples_before_gp_training

        # Set marginal likelihood
        if marginal_likelihood == None:
            self.marginal_likelihood = gpytorch.mlls.ExactMarginalLogLikelihood
        else:
            self.marginal_likelihood = marginal_likelihood

        if nan_it_routine is None:
            self.nan_it_routine = None
        else:
            self.nan_it_routine = lambda obj, start_p, start_p_val, end_p, end_p_val: nan_it_routine(obj, start_p, start_p_val, end_p, end_p_val, self, run_type="nan_gp_line_search", **nan_it_routine_config)
        
        if lin_search_routine is None:
            self.lin_search_routine = None
        else:
            self.lin_search_routine = lambda obj, start_p, start_p_val, end_p, end_p_val: lin_search_routine(obj, start_p, start_p_val, end_p, end_p_val, self, run_type="gp_line_search", **{k: v for k, v in lin_search_routine_config.items() if k not in {'lr_multiplier', 'delta_min'}})
            if 'lr_multiplier' in lin_search_routine_config:
                self.lin_search_routine_lr_multiplier = lin_search_routine_config['lr_multiplier']
            else:
                self.lin_search_routine_lr_multiplier = 0.0
            if 'delta_min' in lin_search_routine_config:
                self.delta_min = lin_search_routine_config['delta_min']
            else:
                self.delta_min = None


        if terminal_condition:
            self.check_for_terminal_condition = True
            self.terminal_condition_config = terminal_condition
        else:
            self.check_for_terminal_condition = False

        
        # Explicitly initialize the history dictionary        
        self.history: List[Dict[str, Any]] = []

        if (generate_initial_data_config is not None) and ('start_from_init_best' in generate_initial_data_config) and (generate_initial_data_config['start_from_init_best'] is True) and (generate_initial_data is not None):
            self.start_from_init_best = True
        else:
            self.start_from_init_best = False

        # Initialize training data.
        if generate_initial_data is not None:

            if (generate_initial_data_config is not None) and ('bounds' in generate_initial_data_config) and (generate_initial_data_config['bounds'] is not None) and (not torch.is_tensor(generate_initial_data_config['bounds'])):
                generate_initial_data_config['bounds'] = torch.tensor(generate_initial_data_config['bounds'], dtype=self.sample_point.dtype)

            # Generate initial data and scale inputs.
            obj_eval_start_time = time.time()
            train_x_init, train_y_init = generate_initial_data(self.objective, **{k: v for k, v in generate_initial_data_config.items() if k not in {'start_from_init_best'}})
            train_x_init = train_x_init / self.objfcn_input_scaling_coeffs
            obj_eval_elapsed_time = time.time() - obj_eval_start_time

            original_train_x_init = train_x_init.clone()
            original_train_y_init = train_y_init.clone()

            # Get best point
            best_point, best_point_value, best_idx = get_best_point(sample_points=train_x_init, observations=train_y_init, return_index=True)

            # Get distances wrt best point
            distances = torch.norm(train_x_init - best_point, dim=1)
            
            # Get the sorted indices based on distance
            sorted_indices = torch.argsort(distances)

            # Reorder train_x_init and train_y_init accordingly
            train_x_init = train_x_init[sorted_indices]
            train_y_init = train_y_init[sorted_indices]

            self.onlyfeaspoints_model.eval()
            self.virtualdatapoints_model.eval()

            # Append initial training data to both models
            gp_feastraining_x, gp_feastraining_y = self.onlyfeaspoints_model.append_train_data(train_x_init, train_y_init, unlimited=True)
            gp_virtualtraining_x, gp_virtualtraining_y = self.virtualdatapoints_model.append_train_data_vdp(train_x_init, train_y_init, lambda x: self.vdp_function(x, best_point), unlimited=True)
            
            # Append initial training data to history
            for i in range(train_x_init.shape[0]):
                history_entry = {
                    "type": "init_point",
                    "iteration_number": 0,
                    "evaluation_number": self.objective._calls - train_x_init.shape[0] + 1 + i,
                    "sample_point": original_train_x_init[i, :].clone().unsqueeze(0) * self.objfcn_input_scaling_coeffs,
                    "observation": original_train_y_init[i].clone().unsqueeze(0),
                    "objective_eval_time": obj_eval_elapsed_time / train_x_init.shape[0],
                    "acqf_opt_time": 0,
                    "gp_update_time": 0,
                    "optimization_algo_time": 0,
                    "gp_hyperparam_tuning_time": 0,
                    "gp_lengthscales": (self.onlyfeaspoints_model.covar_module.base_kernel.lengthscale.detach().flatten().clone() if i == train_x_init.shape[0] - 1 else []),
                    "gp_outputscale": (self.onlyfeaspoints_model.covar_module.outputscale.detach().clone() if i == train_x_init.shape[0] - 1 else []),
                    "gp_noise": (self.onlyfeaspoints_model.likelihood.noise.detach().clone() if i == train_x_init.shape[0] - 1 else []),
                    "gp_mean": (self.onlyfeaspoints_model.mean_module.constant.detach().clone() if i == train_x_init.shape[0] - 1 else []),
                    "gp_train_feaspoints_x": (gp_feastraining_x * self.objfcn_input_scaling_coeffs if i == train_x_init.shape[0] - 1 else []),
                    "gp_train_feaspoints_y": (gp_feastraining_y if i == train_x_init.shape[0] - 1 else []),
                    "gp_train_virtualpoints_x": (gp_virtualtraining_x * self.objfcn_input_scaling_coeffs if i == train_x_init.shape[0] - 1 else []),
                    "gp_train_virtualpoints_y": (gp_virtualtraining_y if i == train_x_init.shape[0] - 1 else []),
                    "gp_train_virtualpoints_isnan": (self.virtualdatapoints_model.train_ys.clone().isnan() if i == train_x_init.shape[0] - 1 else []),
                }
                self.history.append(history_entry)

            if self.start_from_init_best:
                self.best_init_point = best_point.clone()
                self.best_init_point_value = best_point_value.clone()

        self.last_gd_starting_point = torch.full_like(self.sample_point, float('nan'))
        self.last_gd_starting_point_value = torch.tensor(float('nan'))
        self.last_est_gradient = torch.full_like(self.sample_point, float('nan'))
        self.update_lr = False
        
        # Initialize iteration counter
        self.iteration_number = 0
        
    def step(self) -> None:
        self.iteration_number += 1

        algo_start_time = time.time()

        # Sample with new params from objective and add this to train data.
        # Optionally forget old points (if N > N_max).
        obj_eval_start_time = time.time()
        if (self.iteration_number > 1) or (not self.start_from_init_best):
            observation_value = self.objective(self.sample_point * self.objfcn_input_scaling_coeffs)
        else:
            self.sample_point.copy_(self.best_init_point)
            observation_value = self.best_init_point_value
        obj_eval_elapsed_time = time.time() - obj_eval_start_time

        if self.iteration_number == 1 and (not observation_value.isnan()):
            # Initialize last values for first iteration
            self.last_gd_starting_point = self.sample_point.clone()
            self.last_gd_starting_point_value = observation_value.clone()
        elif self.iteration_number == 1 and observation_value.isnan():
            raise ValueError("First iteration must have a non-nan observation value.")
        
        if (self.iteration_number > 1) or (not self.start_from_init_best):
            gp_onlyfeas_training_x, gp_onlyfeas_training_y = self.onlyfeaspoints_model.append_train_data(self.sample_point, observation_value, update_inout_transforms=False)
        else:
            gp_onlyfeas_training_x, gp_onlyfeas_training_y = self.onlyfeaspoints_model.train_xs.clone(), self.virtualdatapoints_model.train_ys.clone()

        if self.verbose:
            posterior = self.onlyfeaspoints_model.posterior(self.sample_point)
            with warnings.catch_warnings():
                # Ignore NumericalWarning warning since posterior.mvn.variance could be too small and cause issues
                warnings.simplefilter("ignore", category=NumericalWarning)
                print(f"    Sampled point: [{', '.join(f'{x:.2g}' for x in (self.sample_point*self.objfcn_input_scaling_coeffs).flatten())}]. Objective value: {observation_value.item(): .2g}")
                print(f"    GP mean at sampled point (with prev GP hypers): {posterior.mvn.mean.item(): .2g} and variance {posterior.mvn.variance.item(): .2g}. Prediction error: {abs(observation_value.item()-posterior.mvn.mean.item()): .2g}")     

        # Only optimize model hyperparameters if N >= num_samples_before_gp_training
        if self.optimize_hyperparameters and self.virtualdatapoints_model.N >= self.num_samples_before_gp_training: # check rinominare in min_sampples_for_gp_training
            
            in_dict, out_dict = self.onlyfeaspoints_model.update_inout_transforms()
            _, _ = self.virtualdatapoints_model.update_inout_transforms_vdp(input_dict=in_dict, outcome_dict=out_dict)

            mll = self.marginal_likelihood(self.onlyfeaspoints_model.likelihood, self.onlyfeaspoints_model)

            gp_hypers_fitting_start_time = time.time()
            try:
                mll = botorch.fit.fit_gpytorch_mll(mll)
            except ModelFittingError as err:
                warnings.warn(str(err), RuntimeWarning)
            # botorch.fit.fit_gpytorch_model(mll)
            gp_hypers_fitting_time = time.time() - gp_hypers_fitting_start_time

            # Update the virtual data points model with the hyperparameters of the only feasible points model
            self.virtualdatapoints_model.load_state_dict(self.onlyfeaspoints_model.state_dict())

            # Set models to eval mode
            self.virtualdatapoints_model.eval()
            self.onlyfeaspoints_model.eval()

            if self.verbose:
                print(f"    Time for GP hyperparameter optimization: {gp_hypers_fitting_time:.2f} s.")
                print(f"    GP lengthscales: [{', '.join(f'{x:.2g}' for x in self.onlyfeaspoints_model.covar_module.base_kernel.lengthscale.detach().flatten())}], outputscale: {self.onlyfeaspoints_model.covar_module.outputscale.detach().numpy().item(): .2g}, noise {self.onlyfeaspoints_model.likelihood.noise.detach().numpy().item(): .2g}")
        else:
            gp_hypers_fitting_time = []

        # Update the virtual data points model with the current sampled points (vdp must be computed using the current onlyfeaspoints model)
        if (self.iteration_number > 1) or (not self.start_from_init_best):
            gp_virtualtraining_x, gp_virtualtraining_y = self.virtualdatapoints_model.append_train_data_vdp(self.sample_point, observation_value, lambda x: self.vdp_function(x, self.sample_point.detach() if not torch.isnan(observation_value) else self.last_gd_starting_point.detach()), update_inout_transforms=False)
        else:
            gp_virtualtraining_x, gp_virtualtraining_y = self.virtualdatapoints_model.train_xs.clone(), self.virtualdatapoints_model.train_ys.clone()

        history_entry = {
            "type": "iteration",
            "iteration_number": self.iteration_number,
            "evaluation_number": self.objective._calls,
            "sample_point": self.sample_point.clone() * self.objfcn_input_scaling_coeffs,
            "observation": observation_value.clone(),
            "objective_eval_time": obj_eval_elapsed_time,
            "gp_hyperparam_tuning_time": gp_hypers_fitting_time,
            "gp_lengthscales": self.onlyfeaspoints_model.covar_module.base_kernel.lengthscale.detach().flatten().clone(),
            "gp_outputscale": self.onlyfeaspoints_model.covar_module.outputscale.detach().clone(),
            "gp_noise": self.onlyfeaspoints_model.likelihood.noise.detach().clone(),
            "gp_mean": self.onlyfeaspoints_model.mean_module.constant.detach().clone(),
            "gd_starting_point": self.last_gd_starting_point.clone() * self.objfcn_input_scaling_coeffs,
            "gradient": -(self.last_est_gradient.clone() * self.objfcn_input_scaling_coeffs),
            "learning_rate": self.optimizer_torch.param_groups[0]['lr'],
            "gp_train_feaspoints_x": gp_onlyfeas_training_x * self.objfcn_input_scaling_coeffs,
            "gp_train_feaspoints_y": gp_onlyfeas_training_y,
            "gp_train_virtualpoints_x": gp_virtualtraining_x * self.objfcn_input_scaling_coeffs,
            "gp_train_virtualpoints_y": gp_virtualtraining_y,
            "gp_train_virtualpoints_isnan": self.virtualdatapoints_model.train_ys.clone().isnan(),
        }
        if hasattr(self.onlyfeaspoints_model, "input_transform"):
            in_dict = copy.deepcopy(self.onlyfeaspoints_model.input_transform.state_dict())
            history_entry.update({
                "in_transform_" + key: value 
                for key, value in in_dict.items()
            })
        if hasattr(self.onlyfeaspoints_model, "outcome_transform"):
            out_dict = copy.deepcopy(self.onlyfeaspoints_model.outcome_transform.state_dict())
            history_entry.update({
                "out_transform_" + key: value 
                for key, value in out_dict.items()
            })        
        self.history.append(history_entry)
        prev_history_idx = len(self.history) - 1

        remove_time_start_time = time.time()
        # If linear search subroutine is defined, use it to rebase the iteration point (sample_point)
        if self.lin_search_routine is not None and self.iteration_number > 1:
            gp_update_start_time = time.time()
            self.virtualdatapoints_model.posterior(self.sample_point)
            gp_update_elapsed_time = time.time() - gp_update_start_time

            self.history[-1]["gp_update_time"] = gp_update_elapsed_time

            best_recovery_point, best_recovery_point_value, best_recovery_t = self.lin_search_routine(self.objective, self.last_gd_starting_point, self.last_gd_starting_point_value, self.sample_point.detach().clone(), observation_value.clone())
            self.sample_point.copy_(best_recovery_point)
            observation_value.copy_(best_recovery_point_value)
            self.update_lr = True
            print(f"    [Linear routine]: rebasing sample_point to: [{', '.join(f'{x:.2g}' for x in (self.sample_point.detach().clone()*self.objfcn_input_scaling_coeffs).flatten())}]")
        
        # If sampled point is nan, activate nan_recovery_subroutine
        if torch.isnan(observation_value) and self.nan_it_routine and not self.lin_search_routine:
            best_recovery_point, best_recovery_point_value, best_recovery_t = self.nan_it_routine(self.objective, self.last_gd_starting_point, self.last_gd_starting_point_value, self.sample_point.detach().clone(), observation_value.clone())
            self.sample_point.copy_(best_recovery_point)
            observation_value.copy_(best_recovery_point_value)
            self.update_lr = True
            print(f"    [Recovery routine]: rebasing sample_point to: [{', '.join(f'{x:.2g}' for x in (self.sample_point.detach().clone()*self.objfcn_input_scaling_coeffs).flatten())}]")
        remove_time = time.time() - remove_time_start_time
        
        # Update the learning rate
        if self.update_lr:
            m = self.lin_search_routine_lr_multiplier
            self.optimizer_torch.param_groups[0]['lr'] = self.optimizer_torch.param_groups[0]['lr'] * (1.0 - m*(1.0 - best_recovery_t.item()))
            self.delta = self.delta * (1.0 - m*(1.0 - best_recovery_t.item()))
            if self.delta_min is not None and self.delta < self.delta_min:
                self.delta = self.delta_min
            self.update_lr = False
            self.history[-1]["learning_rate"] = self.optimizer_torch.param_groups[0]['lr']
            print(f"    [Linear routine]: learning rate updated to {self.optimizer_torch.param_groups[0]['lr']: .2g} (best t: {best_recovery_t.item(): .2g})")

        # Check for terminal condition
        if self.check_for_terminal_condition:
            if 'lr_below' in self.terminal_condition_config and self.optimizer_torch.param_groups[0]['lr'] < self.terminal_condition_config['lr_below']:
                self.terminal_condition_reached = True
        
        # Stay local around current parameters.
        if self.update_acqf_bounds:
            self.acqf_bounds = torch.tensor([[-self.delta], [self.delta]]) + self.sample_point.detach().clone()

        # Update the acquisition function with the current parameters
        self.acquisition_fcn.update_theta_i(self.sample_point.detach().clone())

        acq_value_old = None
        for i in range(self.max_samples_per_iteration):        

            # Update the prediction strategy of GPyTorch's posterior with the current sampled points
            # this is necessary for the acquisition function get_L_lower and get_K_XX_inv methods
            gp_update_start_time = time.time()
            self.virtualdatapoints_model.posterior(self.sample_point)
            gp_update_elapsed_time = time.time() - gp_update_start_time

            if self.lin_search_routine is not None and self.iteration_number > 1 and i == 0:
                self.history[prev_history_idx]["optimization_algo_time"] = time.time() - algo_start_time - remove_time - obj_eval_elapsed_time
            else:
                self.history[-1]["gp_update_time"] = gp_update_elapsed_time
                self.history[-1]["optimization_algo_time"] = time.time() - algo_start_time - obj_eval_elapsed_time
            algo_start_time = time.time()

            # Optimize acquistion function and get new observation.
            acqf_optimize_start_time = time.time()
            new_x, acq_value = self.optimize_acqf(self.acquisition_fcn, self.acqf_bounds)
            acqf_optimize_elapsed_time = time.time() - acqf_optimize_start_time

            obj_eval_start_time = time.time()
            new_y = self.objective(new_x * self.objfcn_input_scaling_coeffs)
            obj_eval_elapsed_time = time.time() - obj_eval_start_time

            # Update training points
            gp_onlyfeas_training_x, gp_onlyfeas_training_y = self.onlyfeaspoints_model.append_train_data(new_x, new_y, update_inout_transforms=False)
            gp_virtualtraining_x, gp_virtualtraining_y = self.virtualdatapoints_model.append_train_data_vdp(new_x, new_y, lambda x: self.vdp_function(x, self.sample_point.detach() if not torch.isnan(observation_value) else self.last_gd_starting_point.detach()), update_inout_transforms=False)
            
            update_acqf_start_time = time.time()
            self.acquisition_fcn.update_K_xX_dx()
            update_acqf_elapsed_time = time.time() - update_acqf_start_time

            history_entry = {
                "type": "evaluation",
                "iteration_number": self.iteration_number,
                "evaluation_number": self.objective._calls,
                "sample_point": new_x.clone() * self.objfcn_input_scaling_coeffs,
                "observation": new_y.clone(),
                "acquisition_function": acq_value.clone(),
                "objective_eval_time": obj_eval_elapsed_time,
                "acqf_opt_time": acqf_optimize_elapsed_time + update_acqf_elapsed_time,
                "gp_train_feaspoints_x": gp_onlyfeas_training_x * self.objfcn_input_scaling_coeffs,
                "gp_train_feaspoints_y": gp_onlyfeas_training_y,
                "gp_train_virtualpoints_x": gp_virtualtraining_x * self.objfcn_input_scaling_coeffs,
                "gp_train_virtualpoints_y": gp_virtualtraining_y,
                "gp_train_virtualpoints_isnan": self.virtualdatapoints_model.train_ys.clone().isnan(),
            }
            self.history.append(history_entry)

            # Stop sampling if differece of values of acquired points is smaller than a threshold.
            # Equivalent to: variance of gradient did not change larger than a threshold.
            if self.epsilon_diff_acq_value is not None:
                if acq_value_old is not None:
                    diff = acq_value - acq_value_old
                    if diff < self.epsilon_diff_acq_value:
                        if self.verbose:
                            print(f"    [Acq function]: Stop sampling after {i+1} samples, since gradient certainty is {diff.item(): .2g}.")
                        break
                acq_value_old = acq_value.clone()

        with torch.no_grad():

            # If the point is NaN, reset to the previous good point
            if observation_value.isnan():

                self.sample_point.copy_(self.last_gd_starting_point)
                print(f"    Rebasing GD starting point to: [{', '.join(f'{x:.2g}' for x in (self.sample_point.detach().clone()*self.objfcn_input_scaling_coeffs).flatten())}]")
            
            else:
                self.last_gd_starting_point_value = observation_value.clone()
                self.last_gd_starting_point = self.sample_point.detach().clone()

            self.optimizer_torch.zero_grad(set_to_none=False)
            gp_update_start_time = time.time()
            mean_d, variance_d = self.virtualdatapoints_model.posterior_derivative(self.sample_point)
            gp_update_elapsed_time = time.time() - gp_update_start_time
            est_gradient = -mean_d.view(1, self.D).detach().clone()

            if self.use_opt_bounds:
                # Get bounds
                lower_bounds = self.opt_bounds[0]
                upper_bounds = self.opt_bounds[1]

                # Set est_gradient to zero where gradient is positive and touching the upper bound (self.sample_point = self.sample_point - lr*gradient)
                est_gradient = torch.where((est_gradient < 0) & (self.sample_point[0, :] >= upper_bounds).view(1, self.D), torch.tensor(0.0, dtype=est_gradient.dtype), est_gradient)
                # Set est_gradient to zero where gradient is negative and touching the lower bound
                est_gradient = torch.where((est_gradient > 0) & (self.sample_point[0, :] <= lower_bounds).view(1, self.D), torch.tensor(0.0, dtype=est_gradient.dtype), est_gradient)

            if self.normalize_gradient:
                lengthscale = self.onlyfeaspoints_model.covar_module.base_kernel.lengthscale.detach()
                est_gradient = torch.nn.functional.normalize(est_gradient) * lengthscale
                
            if self.standard_deviation_scaling:
                est_gradient = est_gradient / torch.diag(variance_d.view(self.D, self.D))
            
            if self.lr_schedular:
                if self.iteration_number in self.lr_schedular:
                    self.optimizer_torch.param_groups[0]['lr'] = self.lr_schedular[self.iteration_number]
                    self.history[-1]["learning_rate"] = self.optimizer_torch.param_groups[0]['lr']
                    print(f"    [Linear routine]: resetting learning rate to {self.optimizer_torch.param_groups[0]['lr']: .2g} (iteration: {self.iteration_number})")
            
            self.sample_point.grad[:] = est_gradient  # Define as gradient ascent.

            if self.verbose:
                print(f"    Estimated gradient: [{', '.join(f'{x:.2g}' for x in (-est_gradient*self.objfcn_input_scaling_coeffs).flatten())}]")
            
            # Update the gradient
            self.optimizer_torch.step() # Computes the gradient step as sample_point - lr*sample_point.grad
        
        # Saturate sample_point to upper and lower bounds
        if self.use_opt_bounds:
            lower_bounds = self.opt_bounds[0]
            upper_bounds = self.opt_bounds[1]

            # Perform the saturation/clamping operation using vectorized operations
            below_lower_bound = self.sample_point[0, :] <= lower_bounds
            above_upper_bound = self.sample_point[0, :] >= upper_bounds

            # Saturate parameters to upper and lower bounds bounds and mark them
            self.sample_point[0, below_lower_bound] = lower_bounds[below_lower_bound]
            self.sample_point[0, above_upper_bound] = upper_bounds[above_upper_bound]

            # Print consolidated information if verbose
            if self.verbose:
                if below_lower_bound.any():
                    print(f"    After GD, params below lower bounds: [{', '.join(f'{x+1:d}' for x in below_lower_bound.nonzero().flatten())}]. Saturated to respective lower bounds: [{', '.join(f'{x:.2g}' for x in (lower_bounds[below_lower_bound]*self.objfcn_input_scaling_coeffs[0, below_lower_bound]).flatten())}].")
                if above_upper_bound.any():
                    print(f"    After GD, params above upper bounds: [{', '.join(f'{x+1:d}' for x in above_upper_bound.nonzero().flatten())}]. Saturated to respective upper bounds: [{', '.join(f'{x:.2g}' for x in (upper_bounds[above_upper_bound]*self.objfcn_input_scaling_coeffs[0, above_upper_bound]).flatten())}].")
 
        if self.verbose:
            posterior = self.onlyfeaspoints_model.posterior(self.sample_point)
            with warnings.catch_warnings():
                # Ignore NumericalWarning warning since posterior.mvn.variance could be too small and cause issues
                warnings.simplefilter("ignore", category=NumericalWarning)
                print(f"    Next point to sample: [{', '.join(f'{x:.2g}' for x in (self.sample_point*self.objfcn_input_scaling_coeffs).flatten())}], predicted mean {posterior.mvn.mean.item(): .2g} and stddev {posterior.mvn.stddev.item(): .2g}.")

        self.last_est_gradient = est_gradient

        self.history[-1]["gp_update_time"] = gp_update_elapsed_time
        self.history[-1]["optimization_algo_time"] = time.time() - algo_start_time - obj_eval_elapsed_time
