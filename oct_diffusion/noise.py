import torch
import torch.nn as nn
import cornucopia as cc

from oct_diffusion.schedulers import BetaSchedule, AlphaSchedule


class RandomMarkovChainSampler(nn.Module):
    """
    Sample random timepoint in gaussian Markov chain.

    Parameters
    ----------
    n_timesteps : int
        Number of timesteps between max and min beta values. Default is 200.
    """

    def __init__(self, n_timesteps: int = 200):
        """
        Sample random timepoint in gaussian Markov chain.

        Parameters
        ----------
        n_timesteps : int
            Number of timesteps between max and min beta values. Default is
            200.
        """
        super(RandomMarkovChainSampler, self).__init__()
        self.n_timesteps = n_timesteps
        self.timestep_sampler = cc.RandInt(self.n_timesteps - 1)
        self.beta_schedule = BetaSchedule(self.n_timesteps)
        self.alpha_schedule = AlphaSchedule(self.beta_schedule)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of sampler.

        Parameters
        ----------
        x0 : torch.Tensor
            Clearn tensor to add noise to.

        Returns
        -------
        zt : torch.Tensor
            Noisy output with same shape as x0.
        """
        t = self.timestep_sampler()
        alpha = self.alpha_schedule(t)
        epsilon = torch.randn_like(x0)
        zt = torch.sqrt(alpha) * x0 + torch.sqrt(1 - alpha) * epsilon
        return zt
