"""
Bandwidth-constrained Variational Message Encoding (BVME) Module.

Implements the BVME framework from:
    "Bandwidth-constrained Variational Message Encoding for Cooperative MARL"
    Duan et al., AAMAS 2026. (arXiv:2512.11179)

This module treats inter-agent messages as samples from learned Gaussian
posteriors regularized via KL divergence to an uninformative prior,
providing principled control over communication bandwidth.

Key design decisions (from the paper):
    1. ON-PATH coupling: sampled messages z (not mean mu) feed into Q-network
       during training, ensuring KL regularization constrains the actual
       representations used for decision-making.
    2. Deterministic at eval: z = mu (no sampling) for stability.
    3. Closed-form KL divergence for diagonal Gaussian posterior vs isotropic prior.
"""

import torch
import torch.nn as nn


class BVMEModule(nn.Module):
    """
    BVME variational message encoder.

    Takes a message vector and produces a compressed stochastic representation
    by parameterizing a diagonal Gaussian posterior, sampling via the
    reparameterization trick, and computing KL divergence for regularization.

    Args:
        input_dim (int): Dimension of incoming message (e.g., hidden_dim=64).
        compressed_dim (int): Dimension of compressed message output.
            If None, defaults to input_dim (information-theoretic compression
            only, without dimensional reduction).
        sigma_0 (float): Prior standard deviation. Controls per-dimension
            information capacity. Smaller = tighter bandwidth constraint.
            Typical range: 0.005 to 0.02.
        log_var_min (float): Minimum clamped log-variance. Default -5.0
            (corresponds to sigma_min ~ 0.08).
        log_var_max (float): Maximum clamped log-variance. Default 3.0
            (corresponds to sigma_max ~ 4.48).
    """

    def __init__(
        self,
        input_dim: int,
        compressed_dim: int = None,
        sigma_0: float = 0.01,
        log_var_min: float = -5.0,
        log_var_max: float = 3.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.compressed_dim = compressed_dim if compressed_dim is not None else input_dim
        self.sigma_0 = sigma_0
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

        # Variational encoders: single-layer MLPs (Appendix B.1 of BVME paper)
        self.enc_mu = nn.Linear(input_dim, self.compressed_dim)
        self.enc_log_var = nn.Linear(input_dim, self.compressed_dim)

    def forward(self, msg: torch.Tensor, test_mode: bool = False):
        """
        Forward pass: encode message into variational representation.

        Args:
            msg: Message tensor of shape (batch, input_dim).
            test_mode: If True, use deterministic mean (no sampling).

        Returns:
            z: Compressed message, shape (batch, compressed_dim).
                During training: sampled via reparameterization.
                During eval: deterministic mean mu.
            kl_loss: Scalar KL divergence (averaged over batch and dimensions).
                Returns 0.0 during eval.
        """
        # Parameterize posterior: q(z|msg) = N(mu, diag(sigma^2))
        mu = self.enc_mu(msg)
        log_var = self.enc_log_var(msg)

        # Clamp log-variance for numerical stability (Section 4.2 of paper)
        log_var = torch.clamp(log_var, min=self.log_var_min, max=self.log_var_max)

        if test_mode:
            # Deterministic at evaluation (Section 4.4 of paper)
            return mu, torch.tensor(0.0, device=msg.device)

        # Reparameterization trick: z = mu + sigma * epsilon (Eq. 9)
        sigma = torch.exp(0.5 * log_var)  # log_var = log(sigma^2), so sigma = exp(0.5 * log_var)
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon

        # Compute closed-form KL divergence (Eq. 13, Appendix A)
        kl_loss = self._compute_kl(mu, log_var)

        return z, kl_loss

    def _compute_kl(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence: KL(q(z|msg) || p(z))

        Where:
            q(z|msg) = N(mu, diag(sigma^2))    -- learned posterior
            p(z) = N(0, sigma_0^2 * I)          -- uninformative prior

        Closed-form for diagonal Gaussian posterior vs isotropic prior
        (Appendix A of BVME paper):

            KL = (1/2) * sum_d [ (sigma_d^2 + mu_d^2) / sigma_0^2
                                  - log(sigma_d^2 / sigma_0^2) - 1 ]

        Normalized by compressed_dim for consistent regularization across
        different compression ratios (Appendix A, Normalization section).

        Args:
            mu: Posterior means, shape (batch, compressed_dim).
            log_var: Posterior log-variances, shape (batch, compressed_dim).

        Returns:
            Scalar KL divergence averaged over batch and dimensions.
        """
        sigma_0_sq = self.sigma_0 ** 2
        var = torch.exp(log_var)  # sigma_d^2

        # Per-dimension KL: (sigma_d^2 + mu_d^2) / sigma_0^2 - log(sigma_d^2 / sigma_0^2) - 1
        kl_per_dim = (var + mu ** 2) / sigma_0_sq - log_var + torch.log(
            torch.tensor(sigma_0_sq, device=mu.device)
        ) - 1.0

        # Sum over dimensions, mean over batch, normalize by dimension count
        # (Appendix A, Normalization section: divide by d_msg for consistent strength)
        kl = 0.5 * kl_per_dim.sum(dim=-1).mean() / self.compressed_dim

        return kl
