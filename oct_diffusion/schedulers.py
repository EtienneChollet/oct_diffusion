import torch
from torch import nn
import cornucopia as cc


class BetaSchedule(nn.Module):
    """
    Linear beta schedule used for diffusion process.

    Parameters
    ----------
    n_timesteps : int
        Number of diffusion steps.
    """
    def __init__(self, n_timesteps: int):
        super(BetaSchedule, self).__init__()
        self.n_timesteps = n_timesteps
        self.beta_schedule = torch.linspace(1e-4, 0.02, n_timesteps)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get the noise level for a given time step.

        Parameters
        ----------
        t : torch.Tensor
            Tensor of time steps of shape (batch_size,).

        Returns
        -------
        torch.Tensor
            Noise level corresponding to the time steps.
        """
        return self.beta_schedule[t]

# Example
# beta_schedule = BetaSchedule(100)
# beta_schedule(3)


class AlphaSchedule(nn.Module):
    """
    Alpha and cumulative alpha schedule derived from beta schedule.

    Parameters
    ----------
    beta_schedule : BetaSchedule
        Beta schedule from which to compute alpha values.
    """
    def __init__(self, beta_schedule: BetaSchedule):
        super(AlphaSchedule, self).__init__()
        self.beta_schedule = beta_schedule.beta_schedule

        # Compute alpha_t = 1 - beta_t
        self.alpha_schedule = 1.0 - self.beta_schedule

        # Compute cumulative product \bar{alpha_t} = prod(alpha_1 to alpha_t)
        self.alpha_cumprod = torch.cumprod(self.alpha_schedule, dim=0)

    def forward(self, t: torch.Tensor, cumulative: bool = True
                ) -> torch.Tensor:
        """
        Get the alpha value (or cumulative alpha) for a given time step.

        Parameters
        ----------
        t : torch.Tensor
            Tensor of time steps of shape (batch_size,).
        cumulative : bool
            Whether to return the cumulative product \bar{alpha_t} or alpha_t.

        Returns
        -------
        torch.Tensor
            Alpha value (or cumulative alpha) for the given time steps.
        """
        if cumulative:
            return self.alpha_cumprod[t]
        else:
            return self.alpha_schedule[t]

# Example
# alpha_schedule = AlphaSchedule(beta_schedule)
# alpha_schedule(3)


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
