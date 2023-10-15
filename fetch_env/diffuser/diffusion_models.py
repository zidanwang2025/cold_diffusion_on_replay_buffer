import numpy as np
import torch
import gymnasium as gym
from torch import nn
from experiment_config import Args
import utils1 as utils
from utils1.helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)
from utils1.progress import Silent
import pickle
import sys
import random

class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, action_weight=1.0, loss_discount=1.0, loss_weights=None):
        super().__init__()
        self.action_weight = None
        if Args.include_goal_in_state:
            observation_dim=observation_dim+Args.repeat_len
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        print("trans dim:", self.transition_dim)

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        # set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        # decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        # manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights
    #------------------------------------------ sampling ------------------------------------------#


    @torch.no_grad()
    def backward_sample(self, x, cond, t):
        x_recon = self.model(x, cond, t)

        if t.all() == 0:
            return x_recon
        else:
            return self.forward_sample(x_recon, t-1, noise=None)


    @torch.no_grad()
    def backward_sample_cold(self, x, cond, t):
        x_recon = self.model(x, cond, t)
        x_cold = x - self.forward_sample(x_recon, t, noise=None) + self.forward_sample(x_recon, t-1, noise=None)
        return x_cold


    @torch.no_grad()
    def backward_sample_loop(self, shape, cond, verbose=False, cold=False):

        batch_size = shape[0]
        x = torch.randn(shape).to(torch.device(Args.device))
        x = apply_conditioning(x, cond, self.action_dim)


        recon_list = []
        progress = utils.Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, dtype=torch.long).to(torch.device(Args.device))
            if cold:
                x = self.backward_sample_cold(x, cond, timesteps)
            else:
                x = self.backward_sample(x, cond, timesteps)
            x = apply_conditioning(x, cond, self.action_dim)
            recon_list.append(x)

            progress.update({'t': i})


        progress.close()

        return recon_list

    #------------------------------------------ training ------------------------------------------#

    def forward_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample


    def backward_losses(self, x_start, cond, t):
        noise = None
        x_noisy = self.forward_sample(x_start, t)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)
        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, cond):
        batch_size = len(x)

        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()

        return self.backward_losses(x, cond, t)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)
