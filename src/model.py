import botorch.models.transforms
import botorch.models.transforms.outcome
import torch
import gpytorch
import botorch
import warnings


class ExactGPSEModel(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    """An exact Gaussian process (GP) model with a squared exponential (SE) kernel.

    ExactGP: The base class of gpytorch for any Gaussian process latent function to be
        used in conjunction with exact inference.
    GPyTorchModel: The easiest way to use a GPyTorch model in BoTorch.
        This adds all the api calls that botorch expects in its various modules.

    Attributes:
        train_x: (N x D) The training features X.
        train_y: (N x 1) The training targets y.
        lengthscale_constraint: Constraint for lengthscale of SE-kernel, gpytorch.constraints.
        lengthscale_hyperprior: Hyperprior for lengthscale of SE-kernel, gpytorch.priors.
        outputscale_constraint: Constraint for outputscale of SE-kernel, gpytorch.constraints.
        outputscale_hyperprior: Hyperprior for outputscale of SE-kernel, gpytorch.priors.
        noise_constraint: Constraint for noise, gpytorch.constraints.
        noise_hyperprior: Hyperprior for noise, gpytorch.priors.
        ard_num_dims: Set this if you want a separate lengthscale for each input dimension.
            Should be D if train_x is a N x D matrix.
        prior_mean: Value for constant mean.
    """

    _num_outputs = 1  # To inform GPyTorchModel API.

    def __init__(
        self,
        D: int,
        input_transform = None,
        outcome_transform = None,
        N_max: int = None,
        N_max_nan: int  = None,
        train_x: torch.Tensor = None,
        train_y: torch.Tensor = None,
        lengthscale_constraint = None,
        lengthscale_hyperprior = None,
        outputscale_constraint = None,
        outputscale_hyperprior = None,
        noise_constraint = None,
        noise_hyperprior = None,
        ard_num_dims = None,
        prior_mean = 0,
        is_double = False,
    ):
        if train_x is None:
            train_x, train_y = (torch.empty(0, D), torch.empty(0))

        self.N_max = N_max
        self.N_max_nan = N_max_nan
        self.D = D
        self.train_xs = train_x.clone()
        self.train_ys = train_y.clone()
        self.N = 0 if train_x is None else self.train_xs.shape[0]

        # Init input and outcome transforms
        if input_transform is not None:
            input_transform = input_transform(d=self.D)
        if outcome_transform is not None:
            outcome_transform = outcome_transform(m=1)

        if outcome_transform is not None and isinstance(outcome_transform, botorch.models.transforms.outcome.Standardize):
            outcome_transform.means = torch.tensor([[0.0]], dtype=self.train_xs.dtype)
            outcome_transform.stdvs = torch.tensor([[1.0]], dtype=self.train_xs.dtype)
            outcome_transform._stdvs_sq = torch.tensor([[1.0]], dtype=self.train_xs.dtype)
            outcome_transform._is_trained = torch.tensor(True)

        if outcome_transform is not None and train_y is not None and train_y.nelement() > 0:
            transformed_train_y, _ = outcome_transform(train_y.unsqueeze(1))
            transformed_train_y = transformed_train_y.squeeze()
        else:
            transformed_train_y = train_y

        """Inits GP model with data and a Gaussian likelihood."""
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=noise_constraint, noise_prior=noise_hyperprior
        )

        if train_y is not None:
            train_y = train_y.squeeze(-1)
        super(ExactGPSEModel, self).__init__(train_x, transformed_train_y, likelihood)

        # Save transforms as attributes
        if input_transform is not None:
            self.input_transform = input_transform
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform

        # Update x data normalization parameters
        if hasattr(self, "input_transform") and self.input_transform is not None and self.train_xs.nelement() > 0:
            self.input_transform._update_coefficients(self.train_xs)

        self.mean_module = gpytorch.means.ConstantMean()
        if prior_mean is not None:
            self.mean_module.initialize(constant=prior_mean)
            self.mean_module.raw_constant.requires_grad = False

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=ard_num_dims,
                lengthscale_prior=lengthscale_hyperprior,
                lengthscale_constraint=lengthscale_constraint,
            ),
            outputscale_prior=outputscale_hyperprior,
            outputscale_constraint=outputscale_constraint,
        )
        # Initialize lengthscale and outputscale to mean of priors.
        if lengthscale_hyperprior is not None:
            self.covar_module.base_kernel.lengthscale = lengthscale_hyperprior.mean
        if outputscale_hyperprior is not None:
            self.covar_module.outputscale = outputscale_hyperprior.mean

        if is_double:
            likelihood.double()
            self.mean_module.double()
            self.covar_module.double()
            self.double()

        if self.N_max is not None and self.N_max > 0:
            # If N_max_nan is not set, is marked as unlimited (-1), or exceeds N_max,
            # reset it to half of N_max.
            if self.N_max_nan is None or self.N_max_nan == -1 or self.N_max_nan > self.N_max:
                warnings.warn(
                    "Initializing ExactGPSEModel: N_max_nan is not properly set (None, -1, or greater than N_max). "
                    "Setting N_max_nan to half of N_max."
                )
                self.N_max_nan = self.N_max // 2
            # Otherwise, if N_max_nan is higher than half of N_max, warn the user.
            elif self.N_max_nan > self.N_max / 2:
                warnings.warn(
                    "Initializing ExactGPSEModel: N_max_nan is greater than half of N_max. Consider lowering it."
                )

    def append_train_data(self, train_x, train_y, unlimited: bool=False, update_inout_transforms: bool=True):
        """Adaptively append training data. It appends the last N_max non-NaN values.

        Optionally translates train_x data for the state normalization of the
            MLP.

        Args:
            train_x: (1 x D) New training features.
            train_y: (1 x 1) New training target.
        """
        # Append new data
        self.train_xs = torch.cat([train_x.clone(), self.train_xs])
        self.train_ys = torch.cat([train_y.clone(), self.train_ys])

        if not unlimited:
            # Limit nan training data size.
            if (self.N_max_nan is not None) and (self.N_max_nan != -1):
                # Identify indices where targets are NaN or Inf.
                nan_mask = torch.isnan(self.train_ys) | torch.isinf(self.train_ys)
                nan_indices = torch.nonzero(nan_mask, as_tuple=True)[0]
                if nan_indices.numel() > self.N_max_nan:
                    # Since newest entries are at the beginning, keep the first N_max_nan of these.
                    nan_indices_to_keep = nan_indices[: self.N_max_nan]
                    # Create a mask that keeps:
                    #   - All valid (non-NaN/Inf) entries, and
                    #   - The allowed NaN/Inf entries.
                    keep_nan_mask = torch.zeros_like(self.train_ys, dtype=torch.bool)
                    keep_nan_mask[nan_indices_to_keep] = True
                    valid_mask = (~nan_mask) | keep_nan_mask
                    self.train_xs = self.train_xs[valid_mask]
                    self.train_ys = self.train_ys[valid_mask]

            # Limit overall training data size.
            if (self.N_max is not None) and (self.N_max != -1):
                self.train_xs = self.train_xs[: self.N_max]
                self.train_ys = self.train_ys[: self.N_max]

        mask = ~torch.isnan(self.train_ys) & ~torch.isinf(self.train_ys)

        # Make sure the model is in training mode since the input and output transforms depend on it
        current_mode = self.training
        self.train()

        # Standardize y data
        if update_inout_transforms and hasattr(self, "outcome_transform") and self.outcome_transform is not None and self.train_ys[mask].shape[0] > 1:
            std_train_ys, _ = self.outcome_transform(self.train_ys[mask].unsqueeze(1))
            std_train_ys = std_train_ys.squeeze()
        elif (not update_inout_transforms) and hasattr(self, "outcome_transform") and self.outcome_transform is not None and self.train_ys[mask].shape[0] > 1:
            training_state = self.outcome_transform.training
            self.outcome_transform.eval()
            std_train_ys, _ = self.outcome_transform(self.train_ys[mask].unsqueeze(1))
            self.outcome_transform.train(training_state)
            std_train_ys = std_train_ys.squeeze()
        else:
            std_train_ys = self.train_ys[mask]

        # Update x data normalization parameters
        if update_inout_transforms and hasattr(self, "input_transform") and self.input_transform is not None and self.train_xs[mask].shape[0] > 1:
            self.input_transform._update_coefficients(self.train_xs[mask])

        self.set_train_data(
            inputs=self.train_xs[mask],
            targets=std_train_ys,
            strict=False,
        )

        self.N = self.train_xs[mask].shape[0]

        # Restore the mode
        self.train(current_mode)

        return self.train_xs[mask].clone(), self.train_ys[mask].clone()

    def append_train_data_vdp(self, train_x, train_y, virtualpoint_function, unlimited: bool=False, update_inout_transforms: bool=True):
        """Adaptively append training data.

        Optionally translates train_x data for the state normalization of the
            MLP.

        Args:
            train_x: (1 x D) New training features.
            train_y: (1 x 1) New training target.
        """
        # Append new data
        self.train_xs = torch.cat([train_x.clone(), self.train_xs])
        self.train_ys = torch.cat([train_y.clone(), self.train_ys])

        if not unlimited:
            # Limit nan training data size.
            if (self.N_max_nan is not None) and (self.N_max_nan != -1):
                # Identify indices where targets are NaN or Inf.
                nan_mask = torch.isnan(self.train_ys) | torch.isinf(self.train_ys)
                nan_indices = torch.nonzero(nan_mask, as_tuple=True)[0]
                if nan_indices.numel() > self.N_max_nan:
                    # Since newest entries are at the beginning, keep the first N_max_nan of these.
                    nan_indices_to_keep = nan_indices[: self.N_max_nan]
                    # Create a mask that keeps:
                    #   - All valid (non-NaN/Inf) entries, and
                    #   - The allowed NaN/Inf entries.
                    keep_nan_mask = torch.zeros_like(self.train_ys, dtype=torch.bool)
                    keep_nan_mask[nan_indices_to_keep] = True
                    valid_mask = (~nan_mask) | keep_nan_mask
                    self.train_xs = self.train_xs[valid_mask]
                    self.train_ys = self.train_ys[valid_mask]
            
            # Limit overall training data size.
            if (self.N_max is not None) and (self.N_max != -1):
                self.train_xs = self.train_xs[: self.N_max]
                self.train_ys = self.train_ys[: self.N_max]

        # Generate virtual training targets: if a target is valid, use it;
        # otherwise, replace it with the output of virtualpoint_function.
        self.virtual_train_ys = torch.stack([
            self.train_ys[i].clone() 
            if not (torch.isnan(self.train_ys[i]) or torch.isinf(self.train_ys[i]))
            else virtualpoint_function(self.train_xs[i]).squeeze()
            for i in range(self.train_ys.shape[0])
        ])

        mask = ~torch.isnan(self.train_ys) & ~torch.isinf(self.train_ys)

        # Make sure the model is in training mode since the input and output transforms depend on it
        current_mode = self.training
        self.train()

        # Standardize y data
        if update_inout_transforms and hasattr(self, "outcome_transform") and self.outcome_transform is not None and self.train_ys[mask].shape[0] > 1:
            # Needed to get the standardization parameters without the virtual points
            # For consistency, the standardization parameters are updated using non-non points,
            # but the virtual points are standardized using the updated parameters.
            std_train_ys, _ = self.outcome_transform(self.train_ys[mask].unsqueeze(1))
            training_state = self.outcome_transform.training
            self.outcome_transform.eval() # Needed to stop the outcome_transform from updating its parameters
            std_train_ys, _ = self.outcome_transform(self.virtual_train_ys.unsqueeze(1))
            self.outcome_transform.train(training_state)
            std_train_ys = std_train_ys.squeeze()
        elif (not update_inout_transforms) and hasattr(self, "outcome_transform") and self.outcome_transform is not None and self.train_ys[mask].shape[0] > 1:
            training_state = self.outcome_transform.training
            self.outcome_transform.eval() # Needed to stop the outcome_transform from updating its parameters
            std_train_ys, _ = self.outcome_transform(self.virtual_train_ys.unsqueeze(1))
            self.outcome_transform.train(training_state)
            std_train_ys = std_train_ys.squeeze()
        else:
            std_train_ys = self.virtual_train_ys

        # Update x data normalization parameters
        if update_inout_transforms and hasattr(self, "input_transform") and self.input_transform is not None and self.train_xs[mask].shape[0] > 1:
            self.input_transform._update_coefficients(self.train_xs[mask])

        self.set_train_data(
            inputs=self.train_xs,
            targets=std_train_ys,
            strict=False,
        )

        self.N = self.train_xs.shape[0]

        # Restore the mode
        self.train(current_mode)

        return self.train_xs.clone(), self.virtual_train_ys.clone()
    
    def update_inout_transforms(self, input_dict=None, outcome_dict=None):

        mask = ~torch.isnan(self.train_ys) & ~torch.isinf(self.train_ys)
        
        if input_dict == None or outcome_dict == None:
            
            # Update input transform
            if hasattr(self, "input_transform") and self.input_transform is not None and self.train_xs[mask].shape[0] > 1:
                self.input_transform._update_coefficients(self.train_xs[mask])

            # Update outcome transform
            if hasattr(self, "outcome_transform") and self.outcome_transform is not None and self.train_ys[mask].shape[0] > 1:
                training_state = self.outcome_transform.training
                self.outcome_transform.train()
                _, _ = self.outcome_transform(self.train_ys[mask].unsqueeze(1))
                self.outcome_transform.train(training_state)

        else:
            
            if hasattr(self, "input_transform"):
                self.input_transform.load_state_dict(input_dict)
            if hasattr(self, "outcome_transform"):
                self.outcome_transform.load_state_dict(outcome_dict)

        if hasattr(self, "outcome_transform") and self.outcome_transform is not None and self.train_ys[mask].shape[0] > 1:
            training_state = self.outcome_transform.training
            self.outcome_transform.eval() # Necessary to stop parameter update of the outcome transform
            std_train_ys, _ = self.outcome_transform(self.train_ys[mask].unsqueeze(1))
            self.outcome_transform.train(training_state)
            std_train_ys = std_train_ys.squeeze()
        else:
            std_train_ys = self.train_ys[mask]

        # Make sure the model is in training mode since the input and output transforms depend on it
        current_mode = self.training
        self.train()

        self.set_train_data(
            inputs=self.train_xs[mask],
            targets=std_train_ys,
            strict=False,
        )

        # Restore the mode
        self.train(current_mode)

        return self.input_transform.state_dict() if hasattr(self, "input_transform") else {}, self.outcome_transform.state_dict() if hasattr(self, "outcome_transform") else {}
    
    def update_inout_transforms_vdp(self, input_dict=None, outcome_dict=None):

        mask = ~torch.isnan(self.train_ys) & ~torch.isinf(self.train_ys)
        
        if input_dict == None or outcome_dict == None:

            # Update input transform
            if hasattr(self, "input_transform") and self.input_transform is not None and self.train_xs[mask].shape[0] > 1:
                self.input_transform._update_coefficients(self.train_xs[mask])

            # Update outcome transform
            if hasattr(self, "outcome_transform") and self.outcome_transform is not None and self.train_ys[mask].shape[0] > 1:
                training_state = self.outcome_transform.training
                self.outcome_transform.train()
                _, _ = self.outcome_transform(self.train_ys[mask].unsqueeze(1))
                self.outcome_transform.train(training_state)

        else:
            
            if hasattr(self, "input_transform"):
                self.input_transform.load_state_dict(input_dict)
            if hasattr(self, "outcome_transform"):
                self.outcome_transform.load_state_dict(outcome_dict)

        if hasattr(self, "outcome_transform") and self.outcome_transform is not None and self.train_ys[mask].shape[0] > 1:
            training_state = self.outcome_transform.training
            self.outcome_transform.eval() # Necessary to stop parameter update of the outcome transform
            std_train_ys, _ = self.outcome_transform(self.virtual_train_ys.unsqueeze(1))
            self.outcome_transform.train(training_state)
            std_train_ys = std_train_ys.squeeze()
        else:
            std_train_ys = self.train_ys[mask]

        # Make sure the model is in training mode since the input and output transforms depend on it
        current_mode = self.training
        self.train()

        self.set_train_data(
            inputs=self.train_xs,
            targets=std_train_ys,
            strict=False,
        )

        # Restore the mode
        self.train(current_mode)

        return self.input_transform.state_dict() if hasattr(self, "input_transform") else {}, self.outcome_transform.state_dict() if hasattr(self, "outcome_transform") else {}


    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Compute the prior latent distribution on a given input.

        Typically, this will involve a mean and kernel function. The result must be a
        MultivariateNormal. Calling this model will return the posterior of the latent
        Gaussian process when conditioned on the training data. The output will be a
        MultivariateNormal.

        Args:
            x: (n x D) The test points.

        Returns:
            A MultivariateNormal.
        """
        # Apply input transform if in training mode
        if hasattr(self, "input_transform") and self.input_transform is not None and self.training:
            # Disable learning of coefficients to avoid changing the input transform.
            learn_coeffs = self.input_transform.learn_coefficients
            self.input_transform.learn_coefficients = False
            x = self.transform_inputs(x)
            self.input_transform.learn_coefficients = learn_coeffs

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DerivativeExactGPSEModel(ExactGPSEModel):
    """Derivative of the ExactGPSEModel w.r.t. the test points x.

    Since differentiation is a linear operator this is again a Gaussian process.

    Attributes:
        D: Dimension of train_x-/input-data.
        normalize: Optional normalization function for policy parameterization.
        unnormalize: Optional unnormalization function for policy
            parameterization.
        N_max: Maximum number of training samples (train_x, N) for model inference.
        lengthscale_constraint: Constraint for lengthscale of SE-kernel, gpytorch.constraints.
        lengthscale_hyperprior: Hyperprior for lengthscale of SE-kernel, gpytorch.priors.
        outputscale_constraint: Constraint for outputscale of SE-kernel, gpytorch.constraints.
        outputscale_hyperprior: Hyperprior for outputscale of SE-kernel, gpytorch.priors.
        noise_constraint: Constraint for noise, gpytorch.constraints.
        noise_hyperprior: Hyperprior for noise, gpytorch.priors.
        ard_num_dims: Set this if you want a separate lengthscale for each input dimension.
            Should be D if train_x is a N x D matrix.
        prior_mean: Value for constant mean.
    """

    def __init__(
        self,
        D: int,
        input_transform = None,
        outcome_transform = None,
        N_max: int = None,
        N_max_nan: int  = None,
        train_x: torch.Tensor = None,
        train_y: torch.Tensor = None,
        lengthscale_constraint = None,
        lengthscale_hyperprior = None,
        outputscale_constraint = None,
        outputscale_hyperprior = None,
        noise_constraint = None,
        noise_hyperprior = None,
        ard_num_dims = None,
        prior_mean = 0.0,
        is_double = False,
    ):
        """Inits GP model with data and a Gaussian likelihood."""
        
        super(DerivativeExactGPSEModel, self).__init__(
            D = D,
            input_transform = input_transform,
            outcome_transform = outcome_transform,
            N_max = N_max,
            N_max_nan = N_max_nan,
            train_x = train_x,
            train_y = train_y,
            lengthscale_constraint = lengthscale_constraint,
            lengthscale_hyperprior = lengthscale_hyperprior,
            outputscale_constraint = outputscale_constraint,
            outputscale_hyperprior = outputscale_hyperprior,
            noise_constraint = noise_constraint,
            noise_hyperprior = noise_hyperprior,
            ard_num_dims = ard_num_dims,
            prior_mean = prior_mean,
            is_double = is_double,
        )

    def get_L_lower(self):
        """Get Cholesky decomposition L, where L is a lower triangular matrix.

        Returns:
            Cholesky decomposition L.
        """
        return (
            self.prediction_strategy.lik_train_train_covar.root_decomposition()
            .root.evaluate()
            .detach()
        )

    def get_KXX(self):
        """Get matrix of K(X,X) + sigma_n^2*I.

        Returns:
            Matrix of K(X,X).
        """
        return self.prediction_strategy.lik_train_train_covar.to_dense().detach()

    def get_KXX_inv(self):
        """Get the inverse matrix of K(X,X).

        Returns:
            The inverse of K(X,X).
        """
        L_inv_upper = self.prediction_strategy.covar_cache.detach()
        return L_inv_upper @ L_inv_upper.transpose(0, 1)

    def get_KXX_inv_old(self):
        """Get the inverse matrix of K(X,X).

        Not as efficient as get_KXX_inv.

        Returns:
            The inverse of K(X,X).
        """
        X = self.train_inputs[0]
        if hasattr(self, "input_transform") and self.input_transform is not None and self.training:
            # Disable learning of coefficients to avoid changing the input transform.
            learn_coeffs = self.input_transform.learn_coefficients
            self.input_transform.learn_coefficients = False
            X = self.transform_inputs(X)
            self.input_transform.learn_coefficients = learn_coeffs

        sigma_n = self.likelihood.noise_covar.noise.detach()
        return torch.inverse(
            self.covar_module(X).evaluate() + torch.eye(X.shape[0]) * sigma_n
        )

    def _get_KxX_dx(self, x):
        """Computes the analytic derivative of the kernel K(x,X) w.r.t. x.

        Args:
            x: (n x D) Test points.

        Returns:
            (n x D x N) The derivative of K(x,X) w.r.t. x.
        """
        
        X = self.train_inputs[0]
        if hasattr(self, "input_transform") and self.input_transform is not None and self.training:
            # Disable learning of coefficients to avoid changing the input transform.
            learn_coeffs = self.input_transform.learn_coefficients
            self.input_transform.learn_coefficients = False
            X = self.transform_inputs(X)
            self.input_transform.learn_coefficients = learn_coeffs

        n = x.shape[0]
        K_xX = self.covar_module(x, X).evaluate()
        lengthscale = self.covar_module.base_kernel.lengthscale.detach()
        return (
            -torch.eye(self.D, device=x.device, dtype=X.dtype) / lengthscale ** 2 # DxD tensor containing the inverse of the lengthscale squared in the diagonal
            @ (
                (x.view(n, 1, self.D) - X.view(1, self.N, self.D)) # nxNxD tensor containing all the differences between x and X
                * K_xX.view(n, self.N, 1)                          # multiplication of the differences with the kernel values (every dimensions rescaled by the kernel value)
            ).transpose(1, 2) # nxDxN: for each of the n points in x, the inverse lengthscale squared is multiplied by the matrix between parentheses () relative to the n-th point
        ) # nxDxN matrix

    def _get_Kxx_dx2(self):
        """Computes the analytic second derivative of the kernel K(x,x) w.r.t. x.

        Args:
            x: (n x D) Test points.

        Returns:
            (n x D x D) The second derivative of K(x,x) w.r.t. x.
        """
        lengthscale = self.covar_module.base_kernel.lengthscale.detach()
        sigma_f = self.covar_module.outputscale.detach()
        return (
            torch.eye(self.D, device=lengthscale.device, dtype=self.train_inputs[0].dtype) / lengthscale ** 2
        ) * sigma_f

    def posterior_derivative(self, x):
        """Computes the posterior of the derivative of the GP w.r.t. the given test
        points x.

        Args:
            x: (n x D) Test points.

        Returns:
            A GPyTorchPosterior.
        """
        self.eval()

        # Normalize input point
        x = self.transform_inputs(x)

        if self.prediction_strategy is None:
            self.posterior(x)  # Call this to update prediction strategy of GPyTorch.
        
        K_xX_dx = self._get_KxX_dx(x)
        mean_d = K_xX_dx @ self.get_KXX_inv() @ self.train_targets
        variance_d = (
            self._get_Kxx_dx2() - K_xX_dx @ self.get_KXX_inv() @ K_xX_dx.transpose(1, 2)
        )
        variance_d = variance_d.clamp_min(1e-9)

        if hasattr(self, "input_transform") and self.input_transform is not None:
            mean_d = mean_d * ( 1 / self.input_transform.coefficient )
            variance_d = torch.diag((1/self.input_transform.coefficient).view(-1)) @ variance_d @ torch.diag((1/self.input_transform.coefficient).view(-1))
        
        if hasattr(self, "outcome_transform") and self.outcome_transform is not None:
            mean_d = mean_d * self.outcome_transform.stdvs.item()
            variance_d = variance_d * (self.outcome_transform.stdvs.item()**2)
        
        return mean_d, variance_d