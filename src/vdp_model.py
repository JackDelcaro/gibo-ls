import torch
from src.model import DerivativeExactGPSEModel

def VDPLCB(x: torch.Tensor,
           x_it: torch.Tensor,
           beta: float,
           gp_model: DerivativeExactGPSEModel):
    
    with torch.no_grad():
        observation = gp_model.posterior(x.unsqueeze(0), observation_noise=True).distribution
        observation_it = gp_model.posterior(x_it.unsqueeze(0), observation_noise=True).distribution
    
    return (torch.max(observation.mean, observation_it.mean) + beta*observation.stddev).clone()

def VDPUCB(x: torch.Tensor,
           x_it: torch.Tensor,
           beta: float,
           gp_model: DerivativeExactGPSEModel):
    
    with torch.no_grad():
        observation = gp_model.posterior(x.unsqueeze(0), observation_noise=True).distribution
        observation_it = gp_model.posterior(x_it.unsqueeze(0), observation_noise=True).distribution
    
    return (torch.min(observation.mean, observation_it.mean) - beta*observation.stddev).clone()


def VDPNaN(x: torch.Tensor, y_max: torch.Tensor):
    return float('nan')