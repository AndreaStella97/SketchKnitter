import numpy as np
import torch as th


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = np.exp(lv1)
    v2 = np.exp(lv2)
    kl = 0.5 * (lv2 - lv1 + (v1 + (mu1 - mu2) ** 2) / v2 - 1)
    return kl


def discretized_gaussian_log_likelihood(x, means, log_scales):
    """
    Calcola il log-likelihood di una distribuzione discreta di Gaussiana.

    Args:
        x (torch.Tensor): Tensor di input.
        means (torch.Tensor): Tensor delle medie.
        log_scales (torch.Tensor): Tensor dei logaritmi delle deviazioni standard.

    Returns:
        torch.Tensor: Il log-likelihood di una distribuzione discreta di Gaussiana.
    """
    n_bins = x.shape[-1]
    x = x.view(-1, 1, n_bins)
    means = means.view(-1, 1, n_bins)
    log_scales = log_scales.view(-1, 1, n_bins)

    log_probs = th.distributions.Normal(loc=means, scale=log_scales.exp()).log_prob(x)
    log_probs = log_probs.sum(dim=-1)  # somma le probabilità sui bins

    return log_probs
