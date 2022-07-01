import torch
import numpy as np
import scipy as scp
import itertools


"""
utils to check sigma etc.
"""


def check_p0(m, return_dets=False):
    """
    checks if a (n,n) tensor is a P0 Matrix
    using naive way of calculating the determinants of all submatrices
    """
    dim = m.shape[0]

    if dim > 1:
        n = 0
        for i in range(dim):
            n += scp.special.binom(dim, i)
        n = int(n)
        dets = torch.zeros(n) * torch.nan

        dets[0] = torch.det(m)
        count = 1
        for i in range(1, dim):
            for comb in itertools.combinations(np.arange(dim), i):
                dets[count] = torch.det(m[np.array(comb)][:, np.array(comb)])
                count += 1

    else:
        dets = m

    if return_dets:
        return torch.all(dets >= 0), dets
    else:
        return torch.all(dets >= 0)


def check_valid_sigma(sigma, return_dets=False):
    """
    checks if (sigma**-1 - I) is a P0 matrix
    """
    test = check_p0(
        torch.inverse(sigma) - torch.eye(sigma.shape[0]), return_dets=return_dets
    )
    return test


# define sigma from Takashi, for testing purposes
_sigma = torch.tensor(
    [
        [0.85, -0.34, -0.07, 0.16, -0.06],
        [-0.11, 0.46, 0.06, -0.09, -0.05],
        [-0.16, -0.42, 0.74, 0.66, -0.28],
        [0.01, -0.08, -0.13, 0.70, -0.30],
        [0.02, 0.15, -0.04, 0.23, 0.80],
    ]
)
