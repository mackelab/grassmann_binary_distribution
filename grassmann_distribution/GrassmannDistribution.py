import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Adam

import warnings
import scipy as scp
import scipy.optimize
import numpy as np
import itertools

from grassmann_distribution.utils import check_valid_sigma


class GrassmannBinary:
    """
    Implementation of a multivariate binary probability distribution in the Grassmann formalism
    (Arai, Takashi. Multivariate binary probability distribution in the Grassmann formalism. PhysRevE. 2021.)
    """

    def __init__(
        self, sigma: Tensor, lambd=None,
    ):
        assert len(sigma.shape) == 2
        assert check_valid_sigma(sigma)
        self.dim = sigma.shape[0]
        self.sigma = sigma
        if lambd == None:
            self.lambd = torch.inverse(sigma)  # this is not used at the moment!

    def prob(self, x):
        # Todo: make more efficient by only computing once per oberved states?
        return self.prob_grassmann(x, self.sigma)

    @staticmethod
    def prob_grassmann(x: Tensor, sigma: Tensor,) -> Tensor:  # n x d  # (d x d)
        """
        Return the probability of `x` under a GrassmannBinary with specified parameters.
        As standalone method.
        Args:
            x: Location at which to evaluate the Grassmann, aka binary vector.
            sigma: 
        Returns:
            Log-probabilities of each input.
        """
        assert len(x.shape) == 2  # check dim: batch, d

        batch_size = x.shape[0]
        dim = sigma.shape[0]

        m = torch.zeros((batch_size, dim, dim))

        # vectorized version
        m = sigma.repeat(batch_size, 1, 1) * ((-1) ** (1 - x)).repeat(1, dim).view(
            batch_size, dim, dim
        )
        m = m * (
            1 - torch.eye(dim, dim).repeat(batch_size, 1, 1)
        )  # replace diag with 0
        m = m + (
            torch.eye(dim).repeat(batch_size, 1, 1)
            * (torch.diag(sigma).repeat(batch_size, 1) ** x)
            .repeat(1, dim)
            .view(batch_size, dim, dim)
            * (torch.diag(1 - sigma).repeat(batch_size, 1) ** (1 - x))
            .repeat(1, dim)
            .view(batch_size, dim, dim)
        )

        # looped version
        # for i in range(batch_size):
        #    m[i] = sigma* (-1)**(1-x[i])
        #    m[i].fill_diagonal_(0)
        #    m[i] = m[i] + torch.eye(dim) * torch.diag(sigma)**x[i] * torch.diag(1-sigma)**(1-x[i])

        p = torch.det(m)

        return p

    def mean(self):
        """
        returns the expected value based on self.sigma
        """
        return torch.diagonal(self.sigma, dim1=0, dim2=1)

    @staticmethod
    def cov_grassmann(sigma):
        """
        calculates the cov a a grassmann distribution
        Args:
            sigma (Tensor): parameter of gr

        Returns:
            Tensor: cov
        """
        cov = torch.zeros(sigma.shape)
        dim = sigma.shape[-1]
        for i in range(dim):
            for j in range(i + 1):
                if i == j:
                    cov[i, i] = sigma[i, i] * (1 - sigma[i, i])
                else:
                    cov[i, j] = -sigma[i, j] * sigma[j, i]
                    cov[j, i] = cov[i, j]

        return cov

    def cov(self):
        """
        returns covariance matrix based on self.sigma
        """
        cov = GrassmannBinary.cov_grassmann(self.sigma)

        return cov

    def corr(self):
        """
        returns corr matrix based on self.sigma
        """
        cov = self.cov()
        std = torch.sqrt(torch.diag(cov))
        std_mat = torch.outer(std, std)

        return cov / (std_mat + 1e-8)

    def conditional_sigma(self, xc: Tensor) -> Tensor:
        """
        returns the conditional grassmann matrix for the remaining dimensions, given xc
            xc: Tensor of full dim, with "nan" in remaining positions. (batch_size x d)
        """
        batch_size = xc.shape[0]

        # number of remaining dimensions should be all the same for one batch
        dim_r = (torch.isnan(xc)).sum(1)  # nan if unconditioned
        assert torch.all(torch.torch.eq(dim_r, dim_r[0]))
        dim_r = dim_r[0]
        dim_c = self.dim - dim_r

        sigma_r = torch.zeros((batch_size, dim_r, dim_r))

        # todo: make more efficient by dealing with 0 and 1 differently? split up sigma CC?
        # see paper for details.

        for i in range(batch_size):
            mask = ~torch.isnan(xc[i])  # True if conditioned

            sigma_r[i] = (
                self.sigma[~mask][:, ~mask]  # sigma RR
                - self.sigma[~mask][:, mask]  # sigma RC
                @ torch.inverse(
                    self.sigma[mask][:, mask]  # sigma CC
                    - (torch.eye(dim_c) * (1 - xc[i][mask]))
                )
                @ self.sigma[mask][:, ~mask]  # sigma CR
            )

        return sigma_r

    def sample(self, num_samples: int,) -> Tensor:
        """
        Return samples of a GrassmannBinary with specified parameters.
        Args:
            num_samples: Number of samples to generate.
        Returns:
            Tensor: Samples from the GrassmannBinary.
        """

        samples = torch.zeros((num_samples, self.dim)) * torch.nan

        # sample first dim. simple bernoulli from sigma_00
        samples[:, 0] = torch.bernoulli(self.sigma[0, 0].repeat(num_samples))

        # test code to store conditional probabilities
        # ps = torch.zeros((num_samples, self.dim)) * torch.nan
        # ps[:,0] = self.sigma[0,0].repeat(num_samples)

        for i in range(1, self.dim):
            sigma_c = self.conditional_sigma(samples)
            samples[:, i] = torch.bernoulli(sigma_c[:, 0, 0])

            """
            ### code for testing for invalid ps
            ps[:,i] = sigma_c[:,0,0]
            try:
                samples[:,i] = torch.bernoulli(sigma_c[:,0,0])
            except:
                for j in range(num_samples):
                    try:
                        samples[j,i] = torch.bernoulli(sigma_c[j,0,0])
                    except:
                        samples[j,i] = 1.
             ###
             """

        return samples  # ,ps


"""
Mixture of Grassmann 
"""


class MoGrassmannBinary:
    """
    Mixture of GrassmannBinary 
    """

    def __init__(self, sigma: Tensor, mixing_p: Tensor):
        """
        Args:
            sigma (Tensor): (nc,dim,dim) parameters for mogr
            mixing_p (Tensor): mixing coefficients for mogr, should sum up to one
        """
        assert len(sigma.shape) == 3
        for i in range(sigma.shape[0]):
            assert check_valid_sigma(sigma[i])
        assert sigma.shape[0] == mixing_p.shape[0]
        self.dim = sigma.shape[1]
        self.nc = sigma.shape[0]  # number of components
        self.sigma = sigma
        self.mixing_p = mixing_p

    def prob(self, x):
        """
        evaluates the probability of the mogr, based on self.sigma and self.mixing_p
        Args:
            x (Tensor): samples (batch, dim)

        Returns:
            Tensor: prbabilities (batch)
        """
        # Todo: make more efficient by only computing once per oberved states?
        return self.prob_mograssmann(x, self.mixing_p, self.sigma)

    @staticmethod
    def prob_mograssmann(inputs: Tensor, mixing_p: Tensor, sigmas: Tensor,) -> Tensor:
        """
        Return the probability of `inputs` under a MoGrassmann with specified parameters.
        Unlike the `prob()` method, this method is fully detached from the neural
        network and can be used independent of the neural net in case the MoGrassmann
        parameters are already known.
        Args:
            inputs: 01-tensors at which to evaluate the MoGrassmann. (batch_size, parameter_dim)
            mixing_p: weights of each component of the MoGrassmann. Shape: (num_components).
            sigmas: Parameters of each MoGrassmann, shape (num_components, parameter_dim, parameter_dim).
        Returns:
            Log-probabilities of each input.
        """
        assert len(inputs.shape) == 2  # check dim: batch, dim
        assert sigmas.shape[0] == mixing_p.shape[0]  # check: n_components

        batch_size = inputs.shape[0]
        dim = inputs.shape[-1]
        num_components = mixing_p.shape[0]

        diag_mask = torch.eye(dim).repeat(batch_size, num_components, 1, 1)

        m = sigmas * ((-1) ** (1 - inputs)).repeat(1, num_components * dim).view(
            batch_size, num_components, dim, dim
        )
        m = m * (1 - diag_mask)  # replace diag with 0
        m = m + (
            (diag_mask * sigmas)
            ** inputs.repeat(1, num_components * dim).view(
                batch_size, num_components, dim, dim
            )
            * (diag_mask * (1 - sigmas))
            ** (1 - inputs)
            .repeat(1, num_components * dim)
            .view(batch_size, num_components, dim, dim)
        )

        p = (mixing_p * torch.det(m)).sum(-1)

        return p

    def mean(self):
        """
        returns the expected value based on self.sigma
        """
        return torch.sum(
            torch.diagonal(self.sigma, dim1=-1, dim2=-2) * self.mixing_p.unsqueeze(-1),
            -2,
        )

    @staticmethod
    def cov_mograssmann(mixing_p, sigma) -> Tensor:
        """
        computes the cov for the given mixing coefficients and sigma
        as standalone
        returns:
            mixing_p (n_components)
            cov (dim,dim)
        """
        # get dims
        dim = sigma.shape[-1]
        n_comp = mixing_p.shape[-1]

        assert sigma.shape[0] == n_comp
        # check if mixing coefficients sum up to 1
        assert torch.isclose(torch.sum(mixing_p), torch.ones(1), atol=1e-4)

        # compute cov per component
        # compute diag as p*(1-p)
        means = torch.diagonal(sigma, dim1=-1, dim2=-2)
        cov_diag = torch.diag_embed(means * (1 - means))
        # compute offdiag as -sigma_ij*sigma_ji
        cov_offdiag = -sigma * torch.transpose(sigma, -1, -2)
        # add these up with diag_mask
        diag_mask = torch.eye(dim, dtype=bool).repeat(n_comp, 1, 1)
        cov_per_comp = cov_diag * diag_mask + cov_offdiag * (~diag_mask)

        # compute additional cov from different means
        mean_of_means = torch.sum(
            (torch.diagonal(sigma, dim1=-1, dim2=-2).T * mixing_p).T, -2
        )
        mui_mu = mean_of_means - means
        cov_of_means = torch.einsum(
            "ni,jn->nij", mui_mu, mui_mu.transpose(-1, -2)
        )  # batchwise outer product
        # final weighted sum
        cov = torch.sum(
            cov_per_comp * mixing_p.unsqueeze(-1).unsqueeze(-1), 0
        ) + torch.sum(cov_of_means * mixing_p.unsqueeze(-1).unsqueeze(-1), 0)

        return cov

    def cov(self):
        """
        returns covariance matrix based on self.sigma
        """
        cov = MoGrassmannBinary.cov_mograssmann(self.mixing_p, self.sigma)

        return cov

    def corr(self):
        """
        returns corr matrix based on self.sigma
        """
        cov = self.cov()
        std = torch.sqrt(torch.diag(cov))
        std_mat = torch.outer(std, std)

        return cov / (std_mat + 1e-8)

    @staticmethod
    def corr_mograssmann(mixing_p, sigma) -> Tensor:
        """
        computes the corr
        inputs:
            mixing_p (batch,n_components)
            sigma (num_components, dim, dim)
        returns:
            cov (batch,dim,dim)
        """
        # compute cov, including all components
        cov = MoGrassmannBinary.cov_mograssmann(mixing_p, sigma)
        std = torch.sqrt(torch.diagonal(cov, dim1=-1, dim2=-2))
        std_mat = torch.outer(std, std)
        return cov / (std_mat + 1e-8)

    def conditional_sigma(self, sigma: Tensor, xc: Tensor) -> Tensor:
        """
        returns the conditional grassmann matrix for the remaining dimensions, given xc
            xc: Tensor of full dim, with "nan" in remaining positions. (batch_size x d)
        """
        batch_size = xc.shape[0]

        # number of remaining dimensions should be all the same for one batch
        dim_r = (torch.isnan(xc)).sum(1)  # nan if unconditioned
        assert torch.all(torch.torch.eq(dim_r, dim_r[0]))
        dim_r = dim_r[0]
        dim_c = self.dim - dim_r

        sigma_r = torch.zeros((batch_size, dim_r, dim_r))

        # todo: make more efficient by dealing with 0 and 1 differently? split up sigma CC?
        # see paper for details.

        for i in range(batch_size):
            mask = ~torch.isnan(xc[i])  # True if conditioned

            sigma_r[i] = (
                sigma[~mask][:, ~mask]  # sigma RR
                - sigma[~mask][:, mask]  # sigma RC
                @ torch.inverse(
                    sigma[mask][:, mask]  # sigma CC
                    - (torch.eye(dim_c) * (1 - xc[i][mask]))
                )
                @ sigma[mask][:, ~mask]  # sigma CR
            )

        return sigma_r

    def sample(self, num_samples: int,) -> Tensor:
        """
        Return samples of a moGrassmannBinary with specified parameters.
        Args:
            num_samples: Number of samples to generate.
        Returns:
            Tensor: Samples from the GrassmannBinary.
        """

        # sample how many samples from each component
        ns = torch.tensor(np.random.multinomial(num_samples, self.mixing_p))

        samples = torch.zeros((num_samples, self.dim)) * torch.nan

        count = 0
        for j, n in enumerate(ns):
            if n > 0:
                # sample first dim. simple bernoulli from sigma_00
                samples[count : count + n, 0] = torch.bernoulli(
                    self.sigma[j][0, 0].repeat(n)
                )

                # test code to store conditional probabilities
                # ps = torch.zeros((num_samples, self.dim)) * torch.nan
                # ps[:,0] = self.sigma[0,0].repeat(num_samples)

                for i in range(1, self.dim):
                    sigma_c = self.conditional_sigma(
                        self.sigma[j], samples[count : count + n]
                    )
                    samples[count : count + n, i] = torch.bernoulli(sigma_c[:, 0, 0])

                    """
                    ### code for testing for invalid ps
                    ps[:,i] = sigma_c[:,0,0]
                    try:
                        samples[:,i] = torch.bernoulli(sigma_c[:,0,0])
                    except:
                        for j in range(num_samples):
                            try:
                                samples[j,i] = torch.bernoulli(sigma_c[j,0,0])
                            except:
                                samples[j,i] = 1.
                     ###
                     """
            count += n

        return samples[torch.randperm(num_samples)]  # ,ps


if __name__ == "__main__":
    """
    testing
    """
    # define sigma
    _sigma = torch.tensor(
        [
            [0.85, -0.34, -0.07, 0.16, -0.06],
            [-0.11, 0.46, 0.06, -0.09, -0.05],
            [-0.16, -0.42, 0.74, 0.66, -0.28],
            [0.01, -0.08, -0.13, 0.70, -0.30],
            [0.02, 0.15, -0.04, 0.23, 0.80],
        ]
    )

    # this is what lamda should look like
    lambd_check = torch.tensor(
        [
            [1.2977, 0.9094, 0.0147, -0.2190, 0.0772],
            [0.2638, 2.2488, -0.1012, 0.2521, 0.2195],
            [0.3605, 1.0785, 1.1217, -1.0340, 0.0993],
            [0.0456, 0.2466, 0.2035, 1.0937, 0.5002],
            [-0.0770, -0.4614, 0.0162, -0.4079, 1.0681],
        ]
    )

    # define three example events
    x = torch.zeros((3, 5))
    x[0, 0] = 1

    x[1, 1] = 1

    x[2, 0] = 0
    x[2, 1] = 1
    x[2, 2] = 0
    x[2, 3] = 0
    x[2, 4] = 1

    # x should have these probs
    prob_x_check = torch.tensor([0.0232, 0.0018, 0.0101])

    gr = GrassmannBinary(_sigma)

    assert torch.allclose(gr.lambd, lambd_check, atol=1e-4)
    assert torch.allclose(gr.prob(x), prob_x_check, atol=1e-4)
    assert check_valid_sigma(_sigma)
