from typing import Dict, Optional
import copy

import torch
import botorch
import gpytorch

from src.optimizers import (
    VDPBayesianGradientAscent,
)
from src.model import ExactGPSEModel, DerivativeExactGPSEModel
from src.acquisition_function import optimize_acqf_normalized_scipy, optimize_acqf_scipy, optimize_acqf_botorch

from src.linear_search_routine import gp_line_search

from botorch.models.transforms import Standardize, Normalize
from botorch.models.transforms.input import InputStandardize

from src.initial_point_generation import generate_init_lhs_points

# Dictionaries for translating config file.
# Note: can be extended
prior_dict = {
    "prior": {
        "normal": gpytorch.priors.NormalPrior,
        "gamma": gpytorch.priors.GammaPrior,
        "uniform": gpytorch.priors.UniformPrior,
    }
}
constraint_dict = {
    "constraint": {
        "greater_than": gpytorch.constraints.GreaterThan,
        "less_than": gpytorch.constraints.LessThan,
        "interval": gpytorch.constraints.Interval,
        "positive": gpytorch.constraints.Positive,
    }
}
acqf_dict = {
    "ei": botorch.acquisition.analytic.ExpectedImprovement,
    "log_ei": botorch.acquisition.analytic.LogExpectedImprovement,
    "ucb": botorch.acquisition.analytic.UpperConfidenceBound,
}
optimize_acqf_dict = {
    "norm-scipy": optimize_acqf_normalized_scipy,
    "scipy": optimize_acqf_scipy,
    "botorch-opt-acqf": optimize_acqf_botorch,
}
insertion_config = {
    "method": {
        "vdp-gibo": VDPBayesianGradientAscent,
    },
    "optimizer_config": {
        "OptimizerTorch": {"sgd": torch.optim.SGD, "adam": torch.optim.Adam},
        "optimizer_class": {"sgd": torch.optim.SGD, "adam": torch.optim.Adam},
        "Model": {
            "derivative_gp": DerivativeExactGPSEModel,
            "exact_gp": ExactGPSEModel,
        },
        "marginal_likelihood": {
            "mll":      gpytorch.mlls.ExactMarginalLogLikelihood,
            "loo-pll":  gpytorch.mlls.LeaveOneOutPseudoLikelihood,
        },
        "nan_it_routine": {
            "1DGP":           gp_line_search,
        },
        "nan_it_routine_config": {
            "acquisition_function": acqf_dict,
            "optimize_acqf": optimize_acqf_dict,
        },
        "lin_search_routine": {
            "1DGP":           gp_line_search,
        },
        "lin_search_routine_config": {
            "acquisition_function": acqf_dict,
            "optimize_acqf": optimize_acqf_dict,
        },
        "model_config": {
            "lengthscale_constraint": constraint_dict,
            "lengthscale_hyperprior": prior_dict,
            "outputscale_constraint": constraint_dict,
            "outputscale_hyperprior": prior_dict,
            "noise_constraint": constraint_dict,
            "noise_hyperprior": prior_dict,
            "input_transform": {
                "standardize": InputStandardize,
                "normalize": Normalize,
                },
            "outcome_transform": {
                "standardize": Standardize,
                },
        },
        "optimize_acqf": optimize_acqf_dict,
        "acquisition_function": acqf_dict,
        "generate_initial_data": {
                                 "lhs": generate_init_lhs_points,
                                 },
    },
}

def insert_(dict1: dict, dict2: dict):
    """Insert dict2 into dict1.

    Caution: dict1 is manipulated!

    Args:
        dict1: Dictionary which is manipulated.
        dict2: Dictionary from which insertion information is collected.
    """
    for key, value in dict1.items():
        if key in dict2.keys():
            if isinstance(value, dict):
                insert_(value, dict2[key])
            elif value is not None:
                dict1[key] = dict2[key][value]


def insert(dict1: dict, dict2: dict):
    """Insert dict2 into dict1.

    Args:
        dict1: Dictionary which is manipulated.
        dict2: Dictionary from which insertion information is collected.

    Return:
        Copied and manipulated version of dict1.
    """
    manipulated_dict1 = copy.deepcopy(dict1)
    insert_(manipulated_dict1, dict2)

    return manipulated_dict1

def evaluate_hyperpriors_(
    config: dict,
):
    """Evaluate functions entries in config related to hyperpriors.

    Caution: config is manipulated!

    Args:
        config: Dictionary entries to be evaluated.
    """
    search_constraints = [
        "lengthscale_constraint",
        "outputscale_constraint",
        "noise_constraint",
    ]
    search_priors = [
        "lengthscale_hyperprior",
        "outputscale_hyperprior",
        "noise_hyperprior",
    ]

    for key, value in config.items():
        if key in search_constraints and value is not None:
            if value["constraint"] is not None:
                config[key] = value["constraint"](**value["kwargs"])
            else:
                config[key] = None
        elif key in search_priors and value is not None:
            if value["prior"] is not None:
                config[key] = value["prior"](**value["kwargs"])
            else:
                config[key] = None
        elif isinstance(value, dict):
            evaluate_hyperpriors_(value)


def evaluate_hyperpriors(
    config: dict,
):
    """Evaluate functions entries in config related to hyperpriors.

    Args:
         config: dictionary to be evaluated.

     Return:
         Evaluated config.
    """
    copied_config = copy.deepcopy(config)
    evaluate_hyperpriors_(
        config=copied_config,
    )
    return copied_config