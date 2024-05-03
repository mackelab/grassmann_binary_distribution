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

"""
Estimating classes
1. Moment matching
2. based on samples
3. MoGrassmann on samples
"""


class EstimateGrassmannMomentMatching:
    """
    get naive sigma estimate based on moment matching and quasi symmetric sigma
    input: mean and cov of target distribution (for ex. sampling mean and cov)
    """

    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.dim = mean.shape[-1]
        assert cov.shape[-1] == self.dim

    def construct_sigma(self, verbose=False):
        """
        get naive sigma estimate based on moment matching and quasi symmetric sigma
        """
        sigma = torch.diag(self.mean)

        for i in range(self.dim):
            for j in range(i):
                sigma[i, j] = torch.abs(self.cov[i, j]) ** 0.5
                sigma[j, i] = -sigma[i, j] * torch.sign(self.cov)[i, j]

        if not (check_valid_sigma(sigma)):  # checks if the cov gives a valid sigma
            if verbose:
                print(
                    "Sampling covariance returns no valid sigma initialization. Trying downscaling."
                )
            warnings.warn(
                "Sampling covariance returns no valid sigma . Trying downscaling. "
            )
            mask = torch.ones((self.dim, self.dim)) - torch.eye((self.dim))
            for scale in np.arange(0, 1, 0.02):
                sigma = (
                    sigma - sigma * mask * scale
                )  # *torch.rand((self.dim, self.dim))
                if verbose:
                    print("determinants: ", check_valid_sigma(sigma, return_dets=True))

                if check_valid_sigma(sigma):
                    if verbose:
                        print(f"downscaling cov with {scale} helps.")
                    break

            if not (
                check_valid_sigma(sigma)
            ):  # checks if the sampling cov gives a valid sigma
                warnings.warn(
                    "No valid sigma found. Check if you really want to use the Grassmann framework!"
                )
                return None

        self.sigma = sigma
        return sigma


class EstimateGrassmann(nn.Module):
    """
    class for fitting a GrassmannBinary via gradient descent.
    parametrized by B, C st. sigma = inv(B C^-1 + I )
    """

    def __init__(
        self,
        dim: int,
        B_init=None,
        C_init=None,
        init_on_samples=False,
        samples_init=None,
        verbose=False,
    ):
        """multi-layer NN
        Args:
            dim: Dimensionality of the input: 
            num_hiddens: Number of hidden units in fully connected layer
        """
        super().__init__()
        self.dim = dim
        self.verbose = verbose

        if init_on_samples and not (samples_init == None):
            B_init, C_init = self.get_initial_BC(samples_init)

        # initialize parameters for grassmann
        if B_init == None:
            B_init = torch.randn((self.dim, self.dim))
        if C_init == None:
            C_init = torch.randn((self.dim, self.dim))

        self.B = torch.nn.Parameter(data=B_init, requires_grad=True)
        self.C = torch.nn.Parameter(data=C_init, requires_grad=True)

        self.sigma = self.compute_sigma(self.B, self.C)

    def get_initial_BC(self, samples, return_sigma_init=False):
        """
        based on moment matching
        optimizing via scipy
        """

        if self.verbose:
            print("performing moment matching initialization...")
        # get naive sigma estimate based on moment matching and quasi symmetric sigma

        sample_cov = torch.cov(samples.T)
        sample_mean = samples.mean(0)
        sigma_init = torch.eye(self.dim) * sample_mean
        for i in range(self.dim):
            for j in range(i):
                sigma_init[i, j] = torch.abs(sample_cov[i, j]) ** 0.5
                sigma_init[j, i] = -sigma_init[i, j] * torch.sign(sample_cov)[i, j]

        if not (
            check_valid_sigma(sigma_init)
        ):  # checks if the sampling cov gives a valid sigma
            if self.verbose:
                print(
                    "Sampling covariance returns no valid sigma initialization. Trying downscaling and adding some noise."
                )
            warnings.warn(
                "Sampling covariance returns no valid sigma initialization. Trying downscaling and adding some noise. "
            )
            mask = torch.ones((self.dim, self.dim)) - torch.eye((self.dim))
            for scale in [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.4, 0.3, 0.2, 0.1]:
                sigma_init = (
                    sigma_init * torch.eye((self.dim))
                    + sigma_init * mask * torch.rand((self.dim, self.dim)) * scale
                )
                if check_valid_sigma(sigma_init):
                    if self.verbose:
                        print(f"downscaling cov with {scale} helps.")
                    break

            if not (
                check_valid_sigma(sigma_init)
            ):  # checks if the sampling cov gives a valid sigma
                warnings.warn(
                    "B,C got randomly initialized. Check if you really want to use the Grassmann framework!"
                )
                return None, None

        BC_init = np.array(torch.rand((1, 2 * self.dim * self.dim)))
        res = scp.optimize.minimize(self.loss_BC_init, BC_init, args=(sigma_init))
        BC = torch.tensor(res["x"])
        B = BC.view(2, self.dim, self.dim)[0]
        C = BC.view(2, self.dim, self.dim)[1]

        if return_sigma_init:
            return B, C, sigma_init
        else:
            return B, C

    def loss_BC_init(self, BC, sigma):
        """
        computes L2 loss on compute_sigma(BC) and sigma
        """
        BC = torch.tensor(BC)
        B = BC.view(2, self.dim, self.dim)[0]
        C = BC.view(2, self.dim, self.dim)[1]
        s_est = self.compute_sigma(B, C)
        l = torch.dist(sigma, s_est)
        return np.array(l)

    def compute_sigma(self, B, C):
        """
        Relu on diag such that b_ii and c_ii > 0
        calculates b_new_ii = b_ii + sum_{i \neq j} b_ij, same for C
        """
        # apply relu to diagonal elements of B and C
        mask = torch.ones((self.dim, self.dim)) - torch.eye((self.dim))
        B_ = B * mask + torch.eye(self.dim) * F.relu(torch.diag(B))  # torch.exp
        C_ = C * mask + torch.eye(self.dim) * F.relu(torch.diag(C))

        # make it row diagonal dominant
        B_ = B_ + torch.eye(self.dim) * (torch.sum(torch.abs(B_), 1) - torch.diag(B_))
        C_ = C_ + torch.eye(self.dim) * (torch.sum(torch.abs(C_), 1) - torch.diag(C_))

        lambd = B_ @ torch.inverse(C_) + torch.eye(self.dim)  # BC**-1 + I (80)

        sigma = torch.inverse(lambd)

        # apply clip for stable training
        # sigma = torch.clip(sigma,min=-0.99, max=0.99)*mask + torch.eye(self.dim)*torch.clip(torch.diag(sigma),min=0,max=0.9)

        return sigma

    def prob_grassmann(self, x: Tensor, sigma: Tensor,) -> Tensor:  # n x d  # (d x d)
        """
        Return the probability of `x` under a GrassmannBinary with specified parameters.
        Args:
            x: Location at which to evaluate the Grassmann, aka binary vector.
            sigma: (d x d)
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

        p = torch.det(m)

        return p

    def forward(self, x):
        """Network forward pass.
        Args:
            x: Input tensor (batch_size, dim)
        Returns:
            Network output (batch_size, 1).
            logprob
        """
        assert self.dim == x.shape[1]
        self.sigma = self.compute_sigma(self.B, self.C)

        p = self.prob_grassmann(x, self.sigma)

        logprob = torch.mean(torch.log(p))

        return logprob


class EstimateMoGrassmann(nn.Module):
    """
    class for fitting a MoGrassmannBinary via gradient descent.
    parametrized by B, C st. sigma = inv(B C^-1 + I )
    """

    def __init__(
        self,
        dim: int,
        nc: int,
        B_init=None,
        C_init=None,
        init_on_samples=False,
        samples_init=None,
        verbose=False,
    ):
        """multi-layer NN
        Args:
            dim: Dimensionality of the input: 
            nc: number of components
            num_hiddens: Number of hidden units in fully connected layer
        """
        super().__init__()
        self.dim = dim
        self.nc = nc
        self.verbose = verbose

        # initialize parameters for grassmann
        B_init = torch.randn((self.nc, self.dim, self.dim))
        C_init = torch.randn((self.nc, self.dim, self.dim))
        p_mixing_init = torch.rand(self.nc)

        if init_on_samples and not (samples_init == None):
            B_init1, C_init1 = self.get_initial_BC(samples_init)
            for i in range(nc):
                B_init[i] = B_init1 + torch.randn((self.dim, self.dim)) * 1e-10
                C_init[i] = C_init1 + torch.randn((self.dim, self.dim)) * 1e-10

        self.B = torch.nn.Parameter(data=B_init, requires_grad=True)
        self.C = torch.nn.Parameter(data=C_init, requires_grad=True)
        self.p_mixing_un = torch.nn.Parameter(data=p_mixing_init, requires_grad=True)
        self.p_mixing = F.softmax(self.p_mixing_un, dim=0)

        self.sigma = torch.zeros((self.nc, self.dim, self.dim))
        for i in range(nc):
            self.sigma[i] = self.compute_sigma1(self.B[i], self.C[i])

    def get_initial_BC(self, samples, return_sigma_init=False):
        """
        based on moment matching
        optimizing via scipy
        """

        if self.verbose:
            print("performing moment matching initialization...")
        # get naive sigma estimate based on moment matching and quasi symmetric sigma

        sample_cov = torch.cov(samples.T)
        sample_mean = samples.mean(0)
        sigma_init = torch.eye(self.dim) * sample_mean
        for i in range(self.dim):
            for j in range(i):
                sigma_init[i, j] = torch.abs(sample_cov[i, j]) ** 0.5
                sigma_init[j, i] = -sigma_init[i, j] * torch.sign(sample_cov)[i, j]

        if not (
            check_valid_sigma(sigma_init)
        ):  # checks if the sampling cov gives a valid sigma
            if self.verbose:
                print(
                    "Sampling covariance returns no valid sigma initialization. Trying downscaling and adding some noise."
                )
            warnings.warn(
                "Sampling covariance returns no valid sigma initialization. Trying downscaling and adding some noise. "
            )
            mask = torch.ones((self.dim, self.dim)) - torch.eye((self.dim))
            for scale in [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.4, 0.3, 0.2, 0.1]:
                sigma_init = (
                    sigma_init * torch.eye((self.dim))
                    + sigma_init * mask * torch.rand((self.dim, self.dim)) * scale
                )
                if check_valid_sigma(sigma_init):
                    if self.verbose:
                        print(f"downscaling cov with {scale} helps.")
                    break

            if not (
                check_valid_sigma(sigma_init)
            ):  # checks if the sampling cov gives a valid sigma
                warnings.warn(
                    "B,C got randomly initialized. Check if you really want to use the Grassmann framework!"
                )
                return None, None

        BC_init = np.array(torch.rand((1, 2 * self.dim * self.dim)))
        res = scp.optimize.minimize(self.loss_BC_init, BC_init, args=(sigma_init))
        BC = torch.tensor(res["x"])
        B = BC.view(2, self.dim, self.dim)[0]
        C = BC.view(2, self.dim, self.dim)[1]

        if return_sigma_init:
            return B, C, sigma_init
        else:
            return B, C

    def loss_BC_init(self, BC, sigma):
        """
        computes L2 loss on compute_sigma(BC) and sigma
        """
        BC = torch.tensor(BC)
        B = BC.view(2, self.dim, self.dim)[0]
        C = BC.view(2, self.dim, self.dim)[1]
        s_est = self.compute_sigma1(B, C)
        l = torch.dist(sigma, s_est)
        return np.array(l)

    def compute_sigma1(self, B, C):
        """
        Relu on diag such that b_ii and c_ii > 0
        calculates b_new_ii = b_ii + sum_{i \neq j} b_ij, same for C
        """
        # apply relu to diagonal elements of B and C
        mask = torch.ones((self.dim, self.dim)) - torch.eye((self.dim))
        B_ = B * mask + torch.eye(self.dim) * F.relu(torch.diag(B))  # torch.exp
        C_ = C * mask + torch.eye(self.dim) * F.relu(torch.diag(C))

        # make it row diagonal dominant
        B_ = B_ + torch.eye(self.dim) * (torch.sum(torch.abs(B_), 1) - torch.diag(B_))
        C_ = C_ + torch.eye(self.dim) * (torch.sum(torch.abs(C_), 1) - torch.diag(C_))

        lambd = B_ @ torch.inverse(C_) + torch.eye(self.dim)  # BC**-1 + I (80)

        sigma = torch.inverse(lambd)

        return sigma

    def compute_sigma(self, B, C):
        """
        Relu on diag such that b_ii and c_ii > 0
        calculates b_new_ii = b_ii + sum_{i \neq j} b_ij, same for C
        """
        sigma = torch.zeros(B.shape)
        for i in range(self.nc):
            # apply relu to diagonal elements of B and C
            mask = torch.ones((self.dim, self.dim)) - torch.eye((self.dim))
            B_ = B[i] * mask + torch.eye(self.dim) * F.relu(
                torch.diag(B[i])
            )  # torch.exp
            C_ = C[i] * mask + torch.eye(self.dim) * F.relu(torch.diag(C[i]))

            # make it row diagonal dominant
            B_ = B_ + torch.eye(self.dim) * (
                torch.sum(torch.abs(B_), 1) - torch.diag(B_)
            )
            C_ = C_ + torch.eye(self.dim) * (
                torch.sum(torch.abs(C_), 1) - torch.diag(C_)
            )

            lambd = B_ @ torch.inverse(C_) + torch.eye(self.dim)  # BC**-1 + I (80)

            sigma[i] = torch.inverse(lambd)

        return sigma

    def prob_mograssmann(
        self, x: Tensor, sigma: Tensor, p_mixing: Tensor  # n x d  # (nc, d x d)  # (nc)
    ) -> Tensor:
        """
        Return the probability of `x` under a GrassmannBinary with specified parameters.
        Args:
            x: Location at which to evaluate the Grassmann, aka binary vector.
            sigma: (d x d)
        Returns:
            Log-probabilities of each input.
        """

        assert len(x.shape) == 2  # check dim: batch, dim

        batch_size = x.shape[0]
        dim = x.shape[-1]
        num_components = p_mixing.shape[0]

        diag_mask = torch.eye(dim).repeat(batch_size, num_components, 1, 1)

        m = sigma * ((-1) ** (1 - x)).repeat(1, num_components * dim).view(
            batch_size, num_components, dim, dim
        )
        m = m * (1 - diag_mask)  # replace diag with 0
        m = m + (
            (diag_mask * sigma)
            ** x.repeat(1, num_components * dim).view(
                batch_size, num_components, dim, dim
            )
            * (diag_mask * (1 - sigma))
            ** (1 - x)
            .repeat(1, num_components * dim)
            .view(batch_size, num_components, dim, dim)
        )

        p = (p_mixing * torch.det(m)).sum(-1)

        return p

    def forward(self, x):
        """Network forward pass.
        Args:
            x: Input tensor (batch_size, dim)
        Returns:
            Network output (batch_size, 1).
            logprob
        """
        assert self.dim == x.shape[1]

        self.sigma = self.compute_sigma(self.B, self.C)

        self.p_mixing = F.softmax(self.p_mixing_un, dim=0)
        p = self.prob_mograssmann(x, self.sigma, self.p_mixing)

        logprob = torch.mean(torch.log(p))

        return logprob


"""
training routine
"""


def train_EstimateGrassmann(
    model,
    samples,
    steps=2_000,
    clip_gradient=True,
    verbose=True,
    batch_size=1_000,
    early_stop=False,
):
    """
    simple gradient descent on the model, picking randomly batch_size samples w/o replacement
    returns: loss
    """
    optimizer = Adam(model.parameters(), lr=0.001)

    loss_stored = torch.zeros(steps)
    id_list = np.arange(samples.shape[0])

    running_loss = 0.0
    if verbose:
        print("Started training...")
    for step in range(steps):

        # get the inputs; data is a list of [inputs, labels]
        # _x = gr.sample_grassmann(200)
        # _x = samples[i*200:(i+1)*200]
        ids = np.random.choice(id_list, size=batch_size, replace=False)
        _x = samples[ids]

        if step == 0 and verbose:
            print(f"data shape of one batch: {_x.shape}")

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        logprob = model(_x)

        loss = -logprob
        # loss.requires_grad = True

        loss.backward()

        if clip_gradient:
            # clip gradient of params
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 0.1, error_if_nonfinite=True
            )

        optimizer.step()

        # print statistics
        running_loss += loss.item()
        loss_stored[step] = loss.item()

        if verbose:
            if step % 100 == 1:
                print(
                    f"""step [{step}] loss: {torch.mean(loss_stored[(step-99):step]):.3f} """
                )

        if early_stop and step > 200:
            loss_new = torch.mean(loss_stored[(step - 99) : step])
            loss_old = torch.mean(loss_stored[(step - 200) : step - 100])

            if loss_new > loss_old * 0.999:
                if verbose:
                    print("early stopping with loss(new,old): ", loss_new, loss_old)
                break
    if verbose:
        print("Finished Training.")
    return loss_stored
