from typing import Tuple

import torch
import gpytorch
import botorch
import numpy as np

from src.cholesky import one_step_cholesky, schur_complement_inverse
from botorch.generation import gen_candidates_torch
from torch.quasirandom import SobolEngine
from scipy.optimize import minimize

class GradientInformation(botorch.acquisition.AnalyticAcquisitionFunction):
    '''Acquisition function to sample points for gradient information.

    Attributes:
        model: Gaussian process model that supplies the Jacobian (e.g. DerivativeExactGPSEModel).
    '''

    def __init__(self, model):
        '''Inits acquisition function with model.'''
        super().__init__(model)

    def update_theta_i(self, theta_i: torch.Tensor):
        '''Updates the current parameters.

        This leads to an update of K_xX_dx.

        Args:
            theta_i: New parameters.
        '''

        if not torch.is_tensor(theta_i):
            theta_i = torch.tensor(theta_i)

        # Normalize theta
        if hasattr(self.model, "input_transform") and self.model.input_transform is not None:
            # Disable learning of coefficients to avoid changing the input transform.
            learn_coeffs = self.model.input_transform.learn_coefficients
            self.model.input_transform.learn_coefficients = False
            theta_i = self.model.transform_inputs(theta_i)
            self.model.input_transform.learn_coefficients = learn_coeffs
        
        self.theta_i = theta_i
        self.update_K_xX_dx()

    def update_K_xX_dx(self):
        '''When new x is given update K_xX_dx.'''
        # Pre-compute large part of K_xX_dx.

        X = self.model.train_inputs[0]
        if hasattr(self.model, "input_transform") and self.model.input_transform is not None and self.model.training:
            # Disable learning of coefficients to avoid changing the input transform.
            learn_coeffs = self.model.input_transform.learn_coefficients
            self.model.input_transform.learn_coefficients = False
            X = self.model.transform_inputs(X)
            self.model.input_transform.learn_coefficients = learn_coeffs
        
        x = self.theta_i.view(-1, self.model.D)
        self.K_xX_dx_part = self._get_KxX_dx(x, X)

    def _get_KxX_dx(self, x, X) -> torch.Tensor:
        '''Computes the analytic derivative of the kernel K(x,X) w.r.t. x.

        Args:
            x: (n x D) Test points.
            X: (N x D) Data points.

        Returns:
            (n x D x N) The derivative of K(x,X) w.r.t. x.
        '''
        N = X.shape[0]
        n = x.shape[0]
        K_xX = self.model.covar_module(x, X).evaluate()
        lengthscale = self.model.covar_module.base_kernel.lengthscale.detach()
        return (
            -torch.eye(self.model.D, device=X.device, dtype=X.dtype)
            / lengthscale ** 2
            @ (
                (x.view(n, 1, self.model.D) - X.view(1, N, self.model.D))
                * K_xX.view(n, N, 1)
            ).transpose(1, 2)
        )

    # TODO: nicer batch-update for batch of thetas.
    @botorch.utils.transforms.t_batch_mode_transform(expected_q=1)
    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        '''Evaluate the acquisition function on the candidate set thetas.

        Args:
            thetas: A (b) x D-dim Tensor of (b) batches with a d-dim theta points each.

        Returns:
            A (b)-dim Tensor of acquisition function values at the given theta points.
        '''
        sigma_n = self.model.likelihood.noise_covar.noise
        D = self.model.D
        X = self.model.train_inputs[0]
        x = self.theta_i.view(-1, D)

        # If there is an active input transform, all data must be transformed before entering 
        # the covariance module
        if hasattr(self.model, "input_transform") and self.model.input_transform is not None:
            # Disable learning of coefficients to avoid changing the input transform.
            learn_coeffs = self.model.input_transform.learn_coefficients
            self.model.input_transform.learn_coefficients = False
            thetas = self.model.transform_inputs(thetas)
            if self.model.training:
                X = self.model.transform_inputs(X)
            self.model.input_transform.learn_coefficients = learn_coeffs

        variances = []
        for theta in thetas:
            theta = theta.view(-1, D)
            # Compute K_Xθ, K_θθ (do not forget to add noise).
            K_Xθ = self.model.covar_module(X, theta).evaluate()
            K_θθ = self.model.covar_module(theta).evaluate() + sigma_n * torch.eye(
                K_Xθ.shape[-1]
            ).to(theta)

            try:
                K_XX_inv = schur_complement_inverse(
                    A11_inv=self.model.get_KXX_inv(),
                    A12=K_Xθ,
                    A21=K_Xθ.transpose(-1, -2),
                    A22=K_θθ,
                    )
            except:
                print(f"    [Acquisition function]: error in Schur complement inverse, resorting to classical inverse. "
                      "To avoid numerical instabilities, consider increasing the lower bound of the GP noise or "
                      "decreasing the upper bounds of lenthscales or outputscale.")
                A = torch.cat([torch.cat([self.model.get_KXX(), K_Xθ], dim=-1), torch.cat([K_Xθ.transpose(-1, -2), K_θθ], dim=-1)], dim=-2)
                K_XX_inv = torch.inverse(A)

            # get K_xX_dx
            K_xθ_dx = self._get_KxX_dx(x, theta)
            K_xX_dx = torch.cat([self.K_xX_dx_part, K_xθ_dx], dim=-1)

            # Compute_variance.
            variance_d = -K_xX_dx @ K_XX_inv @ K_xX_dx.transpose(1, 2)
            variances.append(torch.trace(variance_d.view(D, D)).view(1))

        return -torch.cat(variances, dim=0)



def optimize_acqf_botorch(acq_func: botorch.acquisition.AcquisitionFunction, 
                          bounds: torch.Tensor,
                          q: int = 1,
                          num_restarts: int = 5,
                          raw_samples: int = 64,
                          retry_on_optimization_warning: bool = False,
                          verbose: bool = False,
                          **kwargs) -> torch.Tensor:
    '''Wrapper function for botorch.optim.optimize_acqf.

    For instance for expected improvement (botorch.acquisition.analytic.ExpectedImprovement).

    Args:
        acq_function: An AcquisitionFunction.
        q: number of candidates.
        num_restarts: The number of starting points for multistart acquisition function optimization.
        raw_samples: The number of samples for initialization. This is required if batch_initial_conditions is not specified.
        **kwargs: Additional keyword arguments to pass to botorch.optim.optimize_acqf.

    Returns:
        A q x D-dim tensor of generated candidates.
    '''


    candidates, acq_value = botorch.optim.optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        retry_on_optimization_warning=retry_on_optimization_warning,
        **kwargs,
    )

    if verbose:
        print(f"    [Acq function]: Optimized acquisition function value: {acq_value.item(): .2g}.")

    # Observe new values.
    new_x = candidates.detach()
    return new_x, acq_value

def lhs_initial_conditions(bounds: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Generate initial conditions using Latin Hypercube Sampling within given bounds.

    Args:
        bounds: A 2 x D tensor of lower and upper bounds for each dimension.
        num_samples: Number of samples to generate.

    Returns:
        A num_samples x D tensor of initial points sampled using LHS.
    """
    dim = bounds.size(-1)  # Get the dimensionality
    sobol = SobolEngine(dimension=dim, scramble=True)
    lhs_samples = sobol.draw(num_samples)  # Generate samples in [0, 1]

    # Scale samples to fit within the provided bounds
    lower_bounds, upper_bounds = bounds[0], bounds[1]
    scaled_samples = lower_bounds + (upper_bounds - lower_bounds) * lhs_samples
    return scaled_samples

def optimize_acqf_scipy(
    acq_func: botorch.acquisition.AcquisitionFunction,
    bounds: torch.Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
    optimizer_method: str = "L-BFGS-B",
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimizes the acquisition function using custom optimization with LHS initial conditions.

    Args:
        acq_func: The acquisition function to optimize.
        bounds: A 2 x D tensor of lower and upper bounds for each dimension.
        q: The number of candidates.
        num_restarts: The number of starting points for multistart acquisition function optimization.
        raw_samples: The number of samples for LHS initialization.
        optimizer_method: The optimization method to use (e.g., "L-BFGS-B", "Powell", "TNC").

    Returns:
        A two-element tuple containing:
            - a q x D-dim tensor of generated candidates.
            - a tensor of associated acquisition values.
    """
    if bounds.size()[1] == 1:
        bounds = [(bounds[0, i].item(), bounds[1, i].item()) for i in range(bounds.size()[1])]

    # Generate initial conditions using LHS
    raw_initial_conditions = lhs_initial_conditions(bounds, raw_samples)

    # Step 2: Evaluate the acquisition function at these initial conditions
    raw_acq_values = torch.tensor([acq_func(x.unsqueeze(0)).item() for x in raw_initial_conditions])

    # Step 3: Select the top `num_restarts` initial conditions based on acquisition values
    _, topk_indices = torch.topk(raw_acq_values, k=num_restarts, largest=True)
    top_initial_conditions = raw_initial_conditions[topk_indices]

    # Step 4: Convert the bounds to scipy-compatible format
    scipy_bounds = [(bounds[0, i].item(), bounds[1, i].item()) for i in range(bounds.size()[1])]

    # Step 5: Optimize the acquisition function starting from the `num_restarts` best initial points
    _, best_lhs_index  = torch.max(raw_acq_values, 0)  # Get best LHS point based on acq_values
    best_solution = raw_initial_conditions[best_lhs_index]  # Return as a single batch tensor
    best_value = -raw_acq_values[best_lhs_index].item()

    def scipy_acq_func(x: np.ndarray) -> float:
        """Wrapper for the acquisition function compatible with scipy.optimize."""
        x_tensor = torch.tensor(x, dtype=torch.float64)
        return -acq_func(x_tensor.unsqueeze(0)).item()  # Minimize the negative acquisition value

    # Run optimization for each of the `num_restarts` initial conditions
    opt_success = []
    for init_point in top_initial_conditions:
        # Optimize using scipy minimize
        result = minimize(
            fun=scipy_acq_func,
            x0=init_point.numpy(),
            method=optimizer_method,
            bounds=scipy_bounds,
            options={"disp": False},
        )

        if result.fun < best_value:
            best_value = result.fun
            best_solution = torch.tensor(result.x.copy())

        opt_success.append(result.success)

    if verbose:
        print(f"    [Acqf optimization]: {sum(opt_success)} out of {num_restarts} iterations were successful.")

    best_value = - best_value

    # Step 6: Return the best candidate and acquisition value
    return best_solution.unsqueeze(0), torch.tensor([best_value], dtype=torch.float64)

def optimize_acqf_normalized_scipy(
    acq_func: botorch.acquisition.AcquisitionFunction,
    bounds: torch.Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
    optimizer_method: str = "L-BFGS-B",
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimizes the acquisition function using custom optimization with LHS initial conditions.

    Args:
        acq_func: The acquisition function to optimize.
        bounds: A 2 x D tensor of lower and upper bounds for each dimension.
        q: The number of candidates.
        num_restarts: The number of starting points for multistart acquisition function optimization.
        raw_samples: The number of samples for LHS initialization.
        optimizer_method: The optimization method to use (e.g., "L-BFGS-B", "Powell", "TNC").

    Returns:
        A two-element tuple containing:
            - a q x D-dim tensor of generated candidates.
            - a tensor of associated acquisition values.
    """

    # Step 1: Set bounds
    lower_bounds, upper_bounds = bounds[0], bounds[1]
    normalized_bounds = torch.tensor([[0.0] * bounds.size(1), [1.0] * bounds.size(1)])
    # Convert the bounds to scipy-compatible format
    scipy_bounds = [(normalized_bounds[0, i].item(), normalized_bounds[1, i].item()) for i in range(normalized_bounds.size()[1])]

    # Step 2: Generate initial conditions using LHS
    normalized_raw_initial_conditions = lhs_initial_conditions(normalized_bounds, raw_samples)

    # Step 3: Evaluate the acquisition function at these initial conditions
    raw_acq_values = torch.tensor([acq_func((x * (upper_bounds - lower_bounds) + lower_bounds).unsqueeze(0)).item() for x in normalized_raw_initial_conditions])
    
    # Normalize the acquisition values to [0, 1]
    min_acq_value, max_acq_value = raw_acq_values.min(), raw_acq_values.max()
    if max_acq_value.item() - min_acq_value.item() > 1e-6:
        acqf_normalization_mult = 1.0 / (max_acq_value.item() - min_acq_value.item())
    else:
        acqf_normalization_mult = 1.0
    
    normalized_raw_acq_values = (raw_acq_values - min_acq_value) * acqf_normalization_mult

    # Step 4: Select the top `num_restarts` initial conditions based on acquisition values
    _, topk_indices = torch.topk(normalized_raw_acq_values, k=num_restarts, largest=True)
    top_normalized_initial_conditions = normalized_raw_initial_conditions[topk_indices]

    # Step 5: Optimize the acquisition function starting from the `num_restarts` best initial points
    _, best_lhs_index  = torch.max(normalized_raw_acq_values, 0)  # Get best LHS point based on acq_values
    best_normalized_solution = normalized_raw_initial_conditions[best_lhs_index]  # Return as a single batch tensor
    best_normalized_value = -normalized_raw_acq_values[best_lhs_index].item()

    def scipy_acq_func(x_normalized: np.ndarray) -> float:
        """Wrapper for the acquisition function compatible with scipy.optimize."""
        x_tensor = torch.tensor(x_normalized, dtype=torch.float64)
        x_unnormalized = x_tensor * (upper_bounds - lower_bounds) + lower_bounds

        f_value = acq_func(x_unnormalized.unsqueeze(0)).item()
        f_value_normalized = (f_value - min_acq_value) * acqf_normalization_mult

        return -f_value_normalized  # Minimize the negative acquisition value

    # Run optimization for each of the `num_restarts` initial conditions
    opt_success = []
    for init_point in top_normalized_initial_conditions:
        # Optimize using scipy minimize
        result = minimize(
            fun=scipy_acq_func,
            x0=init_point.numpy(),
            method=optimizer_method,
            bounds=scipy_bounds,
            options={"disp": False},
        )

        if result.fun < best_normalized_value:
            best_normalized_value = result.fun
            best_normalized_solution = torch.tensor(result.x.copy())

        opt_success.append(result.success)

    if verbose:
        print(f"    [Acqf optimization]: {sum(opt_success)} out of {num_restarts} iterations were successful.")

    best_normalized_value = - best_normalized_value

    # Step 6: Unnormalize `best_solution` and 'best_value'
    best_solution = best_normalized_solution * (upper_bounds - lower_bounds) + lower_bounds
    best_value = best_normalized_value / acqf_normalization_mult + min_acq_value

    return best_solution.unsqueeze(0), torch.tensor([best_value], dtype=torch.float64)
