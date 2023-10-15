import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# https://github.com/arogozhnikov/einops
import einops
from einops.layers.torch import Rearrange
import pdb

DTYPE = torch.float
DEVICE = 'cpu'


def get_preprocess_fn(fn_names, env):
    fns = [eval(name)(env) for name in fn_names]
    return compose(*fns)


def compose(*fns):

    def _fn(x):
        for fn in fns:
            x = fn(x)
        return x

    return _fn


def to_np(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x


def to_torch(x, dtype=None, device=None):
    dtype = dtype or DTYPE
    device = device or DEVICE
    if type(x) is dict:
        return {k: to_torch(v, dtype, device) for k, v in x.items()}
    elif torch.is_tensor(x):
        return x.to(device).type(dtype)
    # import pdb; pdb.set_trace()
    return torch.tensor(x, dtype=dtype, device=device)


def to_device(x, device=DEVICE):
    if torch.is_tensor(x):
        return x.to(device)
    elif type(x) is dict:
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        print(f'Unrecognized type in `to_device`: {type(x)}')
        pdb.set_trace()


# return [x.to(device) for x in xs]


# -----------------------------------------------------------------------------#
# ---------------------------------- modules ----------------------------------#
# -----------------------------------------------------------------------------#

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


# -----------------------------------------------------------------------------#
# ---------------------------------- sampling ---------------------------------#
# -----------------------------------------------------------------------------#

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def apply_conditioning(x, conditions, action_dim):
    if conditions.get('dataset', None) is None:
        for t, val in conditions.items():
            x[:, t, action_dim:] = val.clone()
    else:
        cond_start = conditions.get('cond_start', None)
        cond_end = conditions.get('cond_end', None)
        goal_index = conditions.get('goal_index', -1)
        for t, val in conditions.items():
            if isinstance(t, int):
                val_copy = val.clone()
                if cond_start is not None and cond_end is not None and t != goal_index:
                    x[:, t, action_dim+cond_start:action_dim+cond_end] = val_copy[cond_start:cond_end]
                elif t == goal_index:
                    if "Reach" not in conditions['dataset']:
                        x[:, t, action_dim:action_dim+6] = val_copy
                    else:
                        x[:, t, action_dim:action_dim+3] = val_copy

    return x

# -----------------------------------------------------------------------------#
# ---------------------------------- losses -----------------------------------#
# -----------------------------------------------------------------------------#

class WeightedLoss(nn.Module):

    def __init__(self, weights, action_dim):
        super().__init__()
        self.register_buffer('weights', weights)
        self.action_dim = action_dim

    def forward(self, pred, targ, online_weight=1):
        """
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        """
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights * online_weight).mean()
        a0_loss = (loss[:, 0, :self.action_dim] / self.weights[0, :self.action_dim]).mean()
        return weighted_loss, {'a0_loss': a0_loss}


class ValueLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        pass

    def forward(self, pred, targ):
        loss = self._loss(pred, targ).mean()

        if len(pred) > 1:
            corr = np.corrcoef(
                to_np(pred).squeeze(),
                to_np(targ).squeeze()
            )[0, 1]
        else:
            corr = np.NaN

        info = {
            'mean_pred': pred.mean(), 'mean_targ': targ.mean(),
            'min_pred': pred.min(), 'min_targ': targ.min(),
            'max_pred': pred.max(), 'max_targ': targ.max(),
            'corr': corr,
        }

        return loss, info


class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')


class ValueL1(ValueLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class ValueL2(ValueLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')


Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
    'value_l1': ValueL1,
    'value_l2': ValueL2,
}
