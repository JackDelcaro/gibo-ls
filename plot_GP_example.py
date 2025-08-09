import torch
import botorch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from src.model import DerivativeExactGPSEModel
import gpytorch
import copy
from src.utils import get_best_point
from src.vdp_model import VDPLCB

def compute_numerical_derivative(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the numerical derivative using central differences (mean of previous and next differences).
    For the first point, uses forward difference; for the last point, uses backward difference.

    Args:
        X: tensor of shape [n] or [..., 1]
        y: tensor of same shape as X

    Returns:
        derivative: 1D tensor of shape [n], containing the derivative at each X
    """
    # Flatten inputs to 1D
    X_flat = X.flatten()
    y_flat = y.flatten()
    n = X_flat.size(0)
    if n < 2:
        raise ValueError("Need at least two points to compute a derivative.")

    # Compute pairwise differences
    dy = y_flat[1:] - y_flat[:-1]    # shape [n-1]
    dx = X_flat[1:] - X_flat[:-1]    # shape [n-1]
    slopes = dy / dx                 # forward/backward slopes between points

    # First point: forward difference
    first = slopes[0:1]              # shape [1]

    # Interior points: average of backward and forward slopes
    # slopes[:-1] is backward slope at i (between i-1 and i)
    # slopes[1:]  is forward slope at i (between i and i+1)
    interior = (slopes[:-1] + slopes[1:]) * 0.5  # shape [n-2]

    # Last point: backward difference
    last = slopes[-1:].clone()       # shape [1]

    # Concatenate to get full derivative tensor
    derivative = torch.cat([first, interior, last], dim=0)  # shape [n]

    return derivative

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

torch.set_default_dtype(torch.float64)
torch.manual_seed(66) # 2 42 60 66 71

onlyfeaspoints_model = DerivativeExactGPSEModel(D=1)
onlyfeaspoints_model.likelihood.noise = 0.003
onlyfeaspoints_model.covar_module.base_kernel.lengthscale = 0.2
onlyfeaspoints_model.covar_module.outputscale = 1.0
onlyfeaspoints_model.mean_module.constant = torch.tensor(1.0)

virtualdatapoints_model = copy.deepcopy(onlyfeaspoints_model)

onlyfeaspoints_model.eval()
onlyfeaspoints_model.likelihood.eval()

constr_position = 0.8

with torch.no_grad(), gpytorch.settings.fast_pred_samples(), gpytorch.settings.prior_mode(True):

    X_plot = torch.linspace(0, 1, 200).unsqueeze(-1)
    X_samples = torch.rand(8).unsqueeze(-1) # Useless line, necessary to make a nice random posterior
    # X_samples = torch.tensor([[0.9810], [0.7798], [0.7798-0.075], [0.2337], [0.4789+0.1], [0.0314]])
    X_samples = torch.tensor([[0.87654+0.025], [0.7798], [0.7798-0.075], [0.2337], [0.4789+0.1], [-0.2]])

    prior_dist = onlyfeaspoints_model(torch.vstack([X_plot, X_samples]))
    f_tot = prior_dist.sample()
    y_tot = onlyfeaspoints_model.likelihood(f_tot).sample()
    f_plot = f_tot[:X_plot.size(0)]
    der_f_plot = compute_numerical_derivative(X_plot, f_plot)
    y_samples = y_tot[X_plot.size(0):]

    f_plot[X_plot.flatten() > constr_position] = torch.tensor(float('nan'))
    der_f_plot[X_plot.flatten() > constr_position] = torch.tensor(float('nan'))
    y_samples[X_samples.flatten() > constr_position] = torch.tensor(float('nan'))


# Append data to GP
onlyfeaspoints_model.append_train_data(X_samples, y_samples)
vdp_function = lambda x, x_curr: VDPLCB(x, x_curr, 2.0, onlyfeaspoints_model)
best_point, best_point_value = get_best_point(sample_points=X_samples, observations=-y_samples)
best_point_value = -best_point_value
gi_current_point_idx = 3
virtualdatapoints_model.append_train_data_vdp(X_samples, y_samples, lambda x: vdp_function(x, X_samples[gi_current_point_idx]))

onlyfeaspoints_model.posterior(X_samples)
virtualdatapoints_model.posterior(X_samples)

# Train mll
# mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(onlyfeaspoints_model.likelihood, onlyfeaspoints_model)
# mll = botorch.fit.fit_gpytorch_mll(mll)
# virtualdatapoints_model.load_state_dict(onlyfeaspoints_model.state_dict())
# onlyfeaspoints_model.posterior(X_samples)
# virtualdatapoints_model.posterior(X_samples)

# Compute posterior and derivatives
with torch.no_grad():

    virtualpointsline_plot = vdp_function(X_plot, X_samples[gi_current_point_idx]).squeeze()

    virt_gp_observed_pred = virtualdatapoints_model.likelihood(virtualdatapoints_model(X_plot))
    virt_gp_post_mean = virt_gp_observed_pred.mean
    virt_gp_post_std = virt_gp_observed_pred.stddev    
    virt_gp_post_lower, virt_gp_post_upper = virt_gp_post_mean - 2*virt_gp_post_std, virt_gp_post_mean + 2*virt_gp_post_std

    virt_gp_post_der_mean, virt_gp_post_der_var = virtualdatapoints_model.posterior_derivative(X_plot)
    virt_gp_post_der_lower, virt_gp_post_der_upper = virt_gp_post_der_mean.flatten() - 2 * torch.sqrt(virt_gp_post_der_var.flatten()), virt_gp_post_der_mean.flatten() + 2 * torch.sqrt(virt_gp_post_der_var.flatten())
    
    
    feas_gp_observed_pred = onlyfeaspoints_model.likelihood(onlyfeaspoints_model(X_plot))
    feas_gp_post_mean = feas_gp_observed_pred.mean
    feas_gp_post_std = feas_gp_observed_pred.stddev    
    feas_gp_post_lower, feas_gp_post_upper = feas_gp_post_mean - 2*feas_gp_post_std, feas_gp_post_mean + 2*feas_gp_post_std

    feas_gp_post_der_mean, feas_gp_post_der_var = onlyfeaspoints_model.posterior_derivative(X_plot)
    feas_gp_post_der_lower, feas_gp_post_der_upper = feas_gp_post_der_mean.flatten() - 2 * torch.sqrt(feas_gp_post_der_var.flatten()), feas_gp_post_der_mean.flatten() + 2 * torch.sqrt(feas_gp_post_der_var.flatten())

# Acquisition function (EI)
from botorch.acquisition import ExpectedImprovement
best_f = best_point_value.unsqueeze(-1)
ei = ExpectedImprovement(model=virtualdatapoints_model, best_f=best_f, maximize=False)
# compute EI on your plot grid
with torch.no_grad():
    ei_acq_values = ei(X_plot.unsqueeze(1).unsqueeze(1)).flatten()

# Acquisition function (GI)
from src.acquisition_function import GradientInformation
gi = GradientInformation(model=virtualdatapoints_model)

gi.update_theta_i(X_samples[gi_current_point_idx])
# compute GI on your plot grid
with torch.no_grad():
    gi_acq_values = gi(X_plot.unsqueeze(1)).flatten()


## PLOT

legend_fontsize = 12
axes_fontsize = 14
labels_fontsize = 16

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 7), sharex=True, gridspec_kw={'hspace': 0.05})
colors = ["#1461f0", "#ed6a5a"]
acqf_colors = ["#248277", "#9d4edd"] # 75b68d
color_axes = "#262626"
color_grey = "#bcbcbc"

# --- plot objective ---
# ax1.plot(X_plot.flatten().numpy(), virtualpointsline_plot.numpy(), linestyle=':', color=colors[0], label="Virt points line")

mask = y_samples.isfinite()
feas_gp_allpoints_x = X_samples[mask]
feas_gp_allpoints_y = y_samples[mask]

ax1.fill_between([constr_position, constr_position + 1], [-3 -3], [+3 +3], alpha=0.2, color=color_grey, label=r'Unfeasible area', zorder=-1)
ax1.plot(X_plot.flatten().numpy(), f_plot.numpy(), 'k--', label='True fcn', zorder=3)
pts = ax1.scatter(X_samples[gi_current_point_idx].flatten().numpy(), y_samples[gi_current_point_idx].flatten().numpy(),
                 s=60, facecolors='white', edgecolors='none',marker='o', linewidths=0, label="Current it point", zorder=7)
pts.set_path_effects([pe.Stroke(linewidth=6, foreground=colors[1]), pe.Stroke(linewidth=2, foreground=colors[0])])

ax1.plot(X_plot.flatten().numpy(), virt_gp_post_mean.numpy(), color=colors[0], label='Virt GP mean', zorder=3)
ax1.fill_between(X_plot.flatten().numpy(), virt_gp_post_lower.numpy(), virt_gp_post_upper.numpy(), alpha=0.2, color=colors[0], label=r'Virt GP $2\cdot$std', zorder=2)
ax1.scatter(X_samples[~mask].flatten().numpy(), virtualdatapoints_model.train_targets[~mask].flatten().numpy(), color=colors[0], s=30, marker='x', label="Virtual pts", zorder=8)

ax1.plot(X_plot.flatten().numpy(), feas_gp_post_mean.numpy(), color=colors[1], label='Feas GP mean', zorder=1)
ax1.fill_between(X_plot.flatten().numpy(), feas_gp_post_lower.numpy(), feas_gp_post_upper.numpy(), alpha=0.2, color=colors[1], label=r'Feas GP $2\cdot$std', zorder=0)
ax1.scatter(feas_gp_allpoints_x[feas_gp_allpoints_x != X_samples[gi_current_point_idx]].flatten().numpy(), feas_gp_allpoints_y[(feas_gp_allpoints_x != X_samples[gi_current_point_idx]).flatten()].flatten().numpy(), color=colors[0], s=60, edgecolors=colors[1], linewidth=2, marker='o', label="Sampled pts", zorder=5)

l = ax1.vlines(X_samples[gi_current_point_idx, 0].item(), -3, 3, linewidth=0.5, alpha=0.208, color=color_axes, label='_nolegend_', zorder=-2)
l.set_linestyle((0, (6, 6)))

ax1.tick_params(axis='both', which='major', labelsize=axes_fontsize)
ax1.set_ylabel(r'$J(\theta)$', fontsize=labels_fontsize, color=color_axes)
ax1.legend(ncol=3, loc='upper center', fontsize=legend_fontsize, bbox_to_anchor=(0.5, 1.52), handlelength=1.0, handletextpad=0.5, columnspacing=3.0)
ax1.set_ylim(((virt_gp_post_lower.min()-0.05*(virt_gp_post_upper.max()-virt_gp_post_lower.min())).item(), (virt_gp_post_upper.max()+0.20*(virt_gp_post_upper.max()-virt_gp_post_lower.min())).item()))
ax1.grid(True, linestyle='-', linewidth=0.5, alpha=0.208, color=color_axes)
ax1.tick_params(axis='both', which='both', direction='in', top=True, right=True, colors=color_axes)
ax1.xaxis.label.set_color(color_axes)
ax1.yaxis.label.set_color(color_axes)
for spine in ax1.spines.values():
    spine.set_color(color_axes)
ax1.set_ylim((-2, +2))

# --- plot derivative ---
ax2.fill_between([constr_position, constr_position + 1], [-20 -20], [+25 +25], alpha=0.2, color=color_grey, label='_nolegend_', zorder=-1)
ax2.fill_between(X_plot.flatten().numpy(), virt_gp_post_der_lower.numpy(), virt_gp_post_der_upper.numpy(), alpha=0.3, color=colors[0], label=r"$\mathrm{Virt}\;\nabla\;GP\;2\cdot\mathrm{std}$", zorder=1)
ax2.plot(X_plot.flatten().numpy(), virt_gp_post_der_mean.numpy(), color=colors[0], label=r"$\mathrm{Virt}\;\nabla\;GP\;\mathrm{mean}$", zorder=2)
ax2.plot(X_plot.flatten().numpy(), der_f_plot.numpy(), 'k--', label='True fcn der', zorder=3)

l = ax2.vlines(X_samples[gi_current_point_idx, 0].item(), -20, 30, linewidth=0.5, alpha=0.208, color=color_axes, label='_nolegend_', zorder=-2)
l.set_linestyle((0, (6, 6)))

ax2.tick_params(axis='both', which='major', labelsize=axes_fontsize)
ax2.set_ylabel(r'$\nabla J(\theta)$', fontsize=labels_fontsize, color=color_axes)
ax2.legend(ncol=3, loc='upper right', fontsize=legend_fontsize, handlelength=1.0, handletextpad=0.5, columnspacing=1.0)
ax2.set_ylim((((virt_gp_post_der_mean - 2 * virt_gp_post_der_var.sqrt()).min()-0.05*((virt_gp_post_der_mean + 2 * virt_gp_post_der_var.sqrt()).max()-(virt_gp_post_der_mean - 2 * virt_gp_post_der_var.sqrt()).min())).item(), ((virt_gp_post_der_mean + 2 * virt_gp_post_der_var.sqrt()).max()+0.20*((virt_gp_post_der_mean + 2 * virt_gp_post_der_var.sqrt()).max()-(virt_gp_post_der_mean - 2 * virt_gp_post_der_var.sqrt()).min())).item()))
ax2.grid(True, linestyle='-', linewidth=0.5, alpha=0.208, color=color_axes)
ax2.tick_params(axis='both', which='both', direction='in', top=True, right=True, colors=color_axes)
ax2.xaxis.label.set_color(color_axes)
ax2.yaxis.label.set_color(color_axes)
for spine in ax2.spines.values():
    spine.set_color(color_axes)
ax2.set_ylim((-15, +22))

# --- plot acquisition ---
ei_acq_values_scaled = (ei_acq_values-ei_acq_values.min())/(ei_acq_values.max()-ei_acq_values.min())
gi_acq_values_scaled = (gi_acq_values-gi_acq_values.min())/(gi_acq_values.max()-gi_acq_values.min())
ei_best_idx = ei_acq_values_scaled.argmax().item()
gi_best_idx = gi_acq_values_scaled.argmax().item()
ax3.fill_between(X_plot.flatten().numpy(), torch.zeros_like(X_plot).numpy().flatten(), ei_acq_values_scaled.numpy(), alpha=0.2, color=acqf_colors[0], zorder=0)
ax3.fill_between(X_plot.flatten().numpy(), torch.zeros_like(X_plot).numpy().flatten(), gi_acq_values_scaled.numpy(), alpha=0.2, color=acqf_colors[1], zorder=2)
ax3.plot(X_plot.flatten().numpy(), ei_acq_values_scaled.numpy(), '-', color=acqf_colors[0], label='Expected Improvement', zorder=1)
ax3.plot(X_plot.flatten().numpy(), gi_acq_values_scaled.numpy(), '-', color=acqf_colors[1], label='Gradient Information', zorder=3)
ax3.scatter(X_plot[ei_best_idx].flatten().numpy(), ei_acq_values_scaled[ei_best_idx].numpy(), marker='*', s=80, facecolors='white', edgecolors=acqf_colors[0], linewidths=1.5, label='_nolegend_', zorder=3)
ax3.scatter(X_plot[gi_best_idx].flatten().numpy(), gi_acq_values_scaled[gi_best_idx].numpy(), marker='*', s=80, facecolors='white', edgecolors=acqf_colors[1], linewidths=1.5, label='_nolegend_', zorder=3)

l = ax3.vlines(X_samples[gi_current_point_idx, 0].item(), -1, 2, linestyle='--', linewidth=0.5, alpha=0.208, color=color_axes, label='_nolegend_', zorder=-2)
l.set_linestyle((0, (6, 6)))

ax3.tick_params(axis='both', which='major', labelsize=axes_fontsize)
ax3.set_ylabel('(scaled) acq fcn', fontsize=labels_fontsize, color=color_axes)
ax3.set_xlabel(r'$\theta$', fontsize=labels_fontsize, color=color_axes)
ax3.legend(ncol=1, loc='upper right', fontsize=legend_fontsize, handlelength=1.0, handletextpad=0.5, columnspacing=1.0)
ax3.set_ylim((0, 1.2))
ax3.grid(True, linestyle='-', linewidth=0.5, alpha=0.208, color=color_axes)
ax3.tick_params(axis='both', which='both', direction='in', top=True, right=True, colors=color_axes)
ax3.xaxis.label.set_color(color_axes)
ax3.yaxis.label.set_color(color_axes)
for spine in ax3.spines.values():
    spine.set_color(color_axes)

ax1.set_xlim((0, 1))
plt.tight_layout(rect=[0,0,1,0.95])


# fig.savefig(
#     "data/GIBO_example.png",
#     dpi=600,               # high resolution
#     bbox_inches="tight",   # crop any extra whitespace
#     pad_inches=0.1           # remove any padding around the axes
# )

plt.show()