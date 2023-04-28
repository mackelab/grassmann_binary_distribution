from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

"""
conditional version of a MoGr distribtuion 
using https://github.com/mackelab/pyknos/blob/main/pyknos/mdn/mdn.py as a template
"""


class GrassmannConditional(nn.Module):
    """
    Conditional density for Grassmann distribution
    """

    def __init__(
        self,
        features: int,
        hidden_features: int,
        hidden_net: nn.Module,
        num_components=1,
        custom_initialization=False,
        embedding_net=None,
    ):
        """Conditional Grassmann with possibly multiple components
        Args:
            features: Dimension of output density.
            hidden_features: Dimension of final layer of `hidden_net`.
            hidden_net: A Module which outputs final hidden representation before
                paramterization layers (i.e sigma, mixing coefficient).
            num_components: Number of Grassmann components.
            custom_initialization: XXX not yet implemented
            embedding_net: not yet implemented
        """

        super().__init__()

        self._features = features
        self._hidden_features = hidden_features
        self._num_components = num_components

        # Modules
        self._hidden_net = hidden_net

        self._logits_layer = nn.Linear(
            hidden_features, num_components
        )  # unnormalized mixing coefficients

        self._BC_layer = nn.Linear(
            hidden_features, num_components * 2 * features**2
        )  # parameterization layer for sigma

        # XXX docstring text
        # embedding_net: NOT IMPLEMENTED
        #         A `nn.Module` which has trainable parameters to encode the
        #         context (conditioning). It is trained jointly with the Grassmann.
        if embedding_net is not None:
            raise NotImplementedError

        # Constant for numerical stability.
        self._epsilon = 1e-4  # 1e-2

        # Initialize mixture coefficients and precision factors sensibly.
        if custom_initialization:
            self._initialize()

    def get_grassmann_params(
        self, context: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return logits, and sigma
        Args:
            context: Input to the MDN, leading dimension is batch dimension.
        Returns:
            A tuple with mixing_p (num_components),
            sigma (num_components, features, features) All batched.
        """

        h = self._hidden_net(context)

        # Logits and B,C are unconstrained and are obtained directly from the
        # output of a linear layer.
        logits = self._logits_layer(h)
        # apply softmax to get normalized mixing coeffiecients
        mixing_p = torch.softmax(logits, 1)

        BC = self._BC_layer(h).view(-1, self._num_components, 2, self._features, self._features)

        sigma = self.compute_sigma(BC[:, :, 0, :, :], BC[:, :, 1, :, :])

        return mixing_p, sigma

    def get_mean(self, context: Tensor) -> Tensor:
        """
        computes the means for the given context
        """
        mixing_p, sigma = self.get_grassmann_params(context)

        return torch.sum(torch.diagonal(sigma, dim1=-1, dim2=-2) * mixing_p.unsqueeze(-1), -2)

    def cov(self, context: Tensor) -> Tensor:
        """
        computes the cov for the given context
        returns:
            cov (batch,dim,dim)
        """
        # get sigmas
        mixing_p, sigma = self.get_grassmann_params(context)

        return self.cov_mograssmann(mixing_p, sigma)

    @staticmethod
    def cov_mograssmann(mixing_p, sigma) -> Tensor:
        """
        computes the cov for the given mixing coefficients and sigma
        as standalone
        returns:
            mixing_p (batch,n_components)
            cov (batch,n_components,dim,dim)
        """
        # get dims
        dim = sigma.shape[-1]
        n_comp = mixing_p.shape[-1]
        batch_size = mixing_p.shape[0]

        assert sigma.shape[0] == batch_size
        assert sigma.shape[1] == n_comp
        # check if mixing coefficients sum up to 1
        assert torch.all(torch.isclose(torch.sum(mixing_p, 1), torch.ones(1), atol=1e-4))

        # compute cov per component
        # compute diag as p*(1-p)
        means = torch.diagonal(sigma, dim1=-1, dim2=-2)
        cov_diag = torch.diag_embed(means * (1 - means))
        # compute offdiag as -sigma_ij*sigma_ji
        cov_offdiag = -sigma * torch.transpose(sigma, -1, -2)
        # add these up with diag_mask
        diag_mask = torch.eye(dim, dtype=bool).repeat(batch_size, n_comp, 1, 1)
        cov_per_comp = cov_diag * diag_mask + cov_offdiag * (~diag_mask)

        # compute additional cov from different means
        mean_of_means = torch.sum(
            torch.diagonal(sigma, dim1=-1, dim2=-2) * mixing_p.unsqueeze(-1), -2
        )
        mui_mu = mean_of_means.unsqueeze(1) - means
        cov_of_means = torch.einsum(
            "bni,bjn->bnij", mui_mu, mui_mu.transpose(-1, -2)
        )  # batchwise outer product

        # final weighted sum
        cov = torch.sum(cov_per_comp * mixing_p.unsqueeze(-1).unsqueeze(-1), 1) + torch.sum(
            cov_of_means * mixing_p.unsqueeze(-1).unsqueeze(-1), 1
        )

        return cov

    def corr(self, context: Tensor) -> Tensor:
        """
        computes the corr for the given context
        returns:
            corr (batch,dim,dim)
        """

        mixing_p, sigma = self.get_grassmann_params(context)

        corr = self.corr_mograssmann(mixing_p, sigma)

        return corr

    @staticmethod
    def corr_mograssmann(mixing_p, sigma) -> Tensor:
        """
        computes the corr
        inputs:
            mixing_p (batch,n_components)
            sigma (batch,num_components, dim, dim)
        returns:
            cov (batch,dim,dim)
        """
        # compute cov, including all components
        cov = GrassmannConditional.cov_mograssmann(mixing_p, sigma)
        std = torch.sqrt(torch.diagonal(cov, dim1=-1, dim2=-2))
        std_mat = torch.einsum("bi,bj->bij", (std, std))  # batchwise outer product
        return cov / (std_mat + 1e-8)

    def compute_sigma(self, B, C):
        """
        computes sigma from unconstrained B and C tensors.
        Applying additional transformations to fulfill constraints:
        Relu on diag such that b_ii and c_ii > 0
        calculates b_new_ii = b_ii + sum_{i \neq j} b_ij, same for C
        Args:
            B,C: Tensors with shape (batch_size, num_components,dim,dim)

        """

        dim = self._features
        batch_size = B.shape[0]
        num_components = B.shape[1]

        # apply relu to diagonal elements of B and C
        mask = torch.ones((dim, dim)) - torch.eye((dim)).repeat(batch_size, num_components, 1, 1)
        diag_mask = torch.eye(dim).repeat(batch_size, num_components, 1, 1)

        B_ = B * mask + diag_mask * F.relu(
            B * torch.eye(dim).repeat(batch_size, num_components, 1, 1)
        )
        C_ = C * mask + diag_mask * F.relu(
            C * torch.eye(dim).repeat(batch_size, num_components, 1, 1)
        )

        # make it row diagonal dominant
        B_ = B_ + torch.diag_embed(torch.sum(torch.abs(B_), -1) + self._epsilon) - B_ * diag_mask
        C_ = C_ + torch.diag_embed(torch.sum(torch.abs(C_), -1) + self._epsilon) - C_ * diag_mask

        lambd = B_ @ torch.inverse(C_) + torch.eye(dim).repeat(
            batch_size, num_components, 1, 1
        )  # BC**-1 + I (80)

        sigma = torch.inverse(lambd)

        return sigma

    def prob(self, inputs: Tensor, context: Tensor) -> Tensor:
        """Return MoGrass(inputs|context) where MoG is a mixture of Grassmann density.
        The MoGrass's parameters (mixture coefficients, Sigma) are the
        outputs of a neural network.
        Args:
            inputs: Input variable, leading dim interpreted as batch dimension.
            context: Conditioning variable, leading dim interpreted as batch dimension.
        Returns:
            probability of inputs given context under a MoG model. (NOT in log space)
        """
        logits, sigmas = self.get_grassmann_params(context)
        return self.prob_mograssmann(inputs, logits, sigmas)

    def forward(self, inputs: Tensor, context: Tensor) -> Tensor:
        """alias for self.prob
        ---
        Return MoGrass(inputs|context) where MoG is a mixture of Grassmann density.
        The MoGrass's parameters (mixture coefficients, Sigma) are the
        outputs of a neural network.
        Args:
            inputs: Input variable, leading dim interpreted as batch dimension.
            context: Conditioning variable, leading dim interpreted as batch dimension.
        Returns:
            probability of inputs given context under a MoG model. (NOT in log space)
        """
        return self.prob(inputs, context)

    @staticmethod
    def prob_mograssmann(
        inputs: Tensor,
        mixing_p: Tensor,
        sigmas: Tensor,
    ) -> Tensor:
        """
        Return the probability of `inputs` under a MoGrassmann with specified parameters.
        Unlike the `prob()` method, this method is fully detached from the neural
        network and can be used independent of the neural net in case the MoGrassmann
        parameters are already known.
        Args:
            inputs: 01-tensors at which to evaluate the MoGrassmann. (batch_size, parameter_dim)
            mixing_p: weights of each component of the MoGrassmann. Shape: (batch_size,
                num_components).
            sigmas: Parameters of each MoGrassmann, shape (batch_size, num_components, parameter_dim, parameter_dim).
        Returns:
            probabilities of each input.
        """
        assert len(inputs.shape) == 2  # check dim: batch, dim
        assert inputs.shape[0] == mixing_p.shape[0]  # check: batch
        assert sigmas.shape[1] == mixing_p.shape[1]  # check: n_components

        batch_size = inputs.shape[0]
        dim = inputs.shape[-1]
        num_components = mixing_p.shape[1]

        diag_mask = torch.eye(dim).repeat(batch_size, num_components, 1, 1)

        m = sigmas * ((-1) ** (1 - inputs)).repeat(1, num_components * dim).view(
            batch_size, num_components, dim, dim
        )
        m = m * (1 - diag_mask)  # replace diag with 0
        m = m + (
            (diag_mask * sigmas)
            ** inputs.repeat(1, num_components * dim).view(batch_size, num_components, dim, dim)
            * (diag_mask * (1 - sigmas))
            ** (1 - inputs)
            .repeat(1, num_components * dim)
            .view(batch_size, num_components, dim, dim)
        )

        p = (mixing_p * torch.det(m)).sum(-1)

        return p

    def sample(self, num_samples: int, context: Tensor) -> Tensor:
        """
        Return num_samples independent samples from MoGrass( . | context).
        Generates num_samples samples for EACH item in context batch i.e. returns
        (num_samples * batch_size) samples in total.
        Args:
            num_samples: Number of samples to generate.
            context: Conditioning variable, leading dimension is batch dimension.
                only for batch_dim = 1 implemented.
        Returns:
            Generated samples: (num_samples, output_dim) with leading batch dimension.
        """

        # only one context at a time is implemente!
        assert context.shape[0] == 1

        # Get necessary quantities.
        mixing_p, sigmas = self.get_grassmann_params(context)
        return self.sample_mograssmann(num_samples, mixing_p.squeeze(0), sigmas.squeeze(0))

    @staticmethod
    def conditional_sigma(sigma: Tensor, xc: Tensor) -> Tensor:
        """
        returns the conditional grassmann matrix for the remaining dimensions, given xc
            xc: Tensor of full dim, with "nan" in remaining positions. (batch_size x d)
        """
        batch_size = xc.shape[0]
        dim = sigma.shape[-1]

        # number of remaining dimensions should be all the same for one batch
        dim_r = (torch.isnan(xc)).sum(1)  # nan if unconditioned
        assert torch.all(torch.torch.eq(dim_r, dim_r[0]))
        dim_r = dim_r[0]
        dim_c = dim - dim_r

        sigma_r = torch.zeros((batch_size, dim_r, dim_r))

        # todo: make more efficient by dealing with 0 and 1 differently? split up sigma CC?
        # see paper for details.

        for i in range(batch_size):
            mask = ~torch.isnan(xc[i])  # True if conditioned

            sigma_r[i] = (
                sigma[~mask][:, ~mask]  # sigma RR
                - sigma[~mask][:, mask]  # sigma RC
                @ torch.inverse(
                    sigma[mask][:, mask] - (torch.eye(dim_c) * (1 - xc[i][mask]))  # sigma CC
                )
                @ sigma[mask][:, ~mask]  # sigma CR
            )

        return sigma_r

    @staticmethod
    def sample_mograssmann(num_samples: int, mixing_p: Tensor, sigma: Tensor) -> Tensor:
        """
        Return samples of a MoGrass with specified parameters.
        Unlike the `sample()` method, this method is fully detached from the neural
        network and can be used independent of the neural net in case the MoGrass
        parameters are already known.
        Args:
            num_samples: Number of samples to generate.
            mixing_p: weights of each component of the MoGrass. Shape: (num_components).
            sigma: parameter for mograss Shape: (num_components, parameter_dim, parameter_dim).
        Returns:
            Tensor: Samples from the MoGrass.
        """

        dim = sigma.shape[-1]
        nc = mixing_p.shape[-1]

        if nc > 1:
            # sample how many samples from each component
            ns = torch.tensor(np.random.multinomial(num_samples, mixing_p.detach().numpy()))
        else:
            ns = [num_samples]

        samples = torch.zeros((num_samples, dim)) * torch.nan

        count = 0
        for j, n in enumerate(ns):
            if n > 0:
                # sample first dim. simple bernoulli from sigma_00
                samples[count : count + n, 0] = torch.bernoulli(sigma[j][0, 0].repeat(n))

                # test code to store conditional probabilities
                # ps = torch.zeros((num_samples, self.dim)) * torch.nan
                # ps[:,0] = self.sigma[0,0].repeat(num_samples)

                for i in range(1, dim):
                    sigma_c = GrassmannConditional.conditional_sigma(
                        sigma[j], samples[count : count + n]
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

    def _initialize(self) -> None:
        """
        Initialize MDN so that mixture coefficients are approximately uniform,
        and covariances are approximately the identity.
        """

        raise NotImplementedError

        # Initialize mixture coefficients to near uniform.
        self._logits_layer.weight.data = self._epsilon * torch.randn(
            self._num_components, self._hidden_features
        )
        self._logits_layer.bias.data = self._epsilon * torch.randn(self._num_components)

        # Initialize diagonal of precision factors to inverse of softplus at 1.
        self._unconstrained_diagonal_layer.weight.data = self._epsilon * torch.randn(
            self._num_components * self._features, self._hidden_features
        )
        self._unconstrained_diagonal_layer.bias.data = torch.log(
            torch.exp(torch.tensor([1 - self._epsilon])) - 1
        ) * torch.ones(self._num_components * self._features) + self._epsilon * torch.randn(
            self._num_components * self._features
        )

        # Initialize off-diagonal of precision factors to zero.
        self._upper_layer.weight.data = self._epsilon * torch.randn(
            self._num_components * self._num_upper_params, self._hidden_features
        )
        self._upper_layer.bias.data = self._epsilon * torch.randn(
            self._num_components * self._num_upper_params
        )


class hidden_fc_net(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_fc_layers: int = 3,
        num_hiddens: int = 128,
    ):
        """multi-layer NN
        Args:
            input_dim: Dimensionality of input
            num_layers: Number of layers of the network.
            output_dim: output dim, should correspond to hidden_features
        """
        super().__init__()
        self.output_dim = output_dim
        self.num_hiddens = num_hiddens

        # construct fully connected layers
        fc_layers = [nn.Linear(input_dim, num_hiddens), nn.ReLU()]
        for _ in range(num_fc_layers - 1):
            fc_layers.append(nn.Linear(num_hiddens, num_hiddens))
            fc_layers.append(nn.ReLU())

        self.fc_subnet = nn.Sequential(*fc_layers)

        self.final_layer = nn.Linear(num_hiddens, output_dim)

    def forward(self, x):
        """Network forward pass.
        Args:
            x: Input tensor (batch_size, input_dim)
        Returns:
            Network output (batch_size, output_dim).
        """

        embedding = self.fc_subnet(x)

        out = self.final_layer(embedding)

        return out


"""
testing
"""
if __name__ == "__main__":
    # define three example events
    x = torch.zeros((4, 5))
    x[0, 0] = 1

    x[1, 1] = 1

    x[2, 0] = 0
    x[2, 1] = 1
    x[2, 2] = 0
    x[2, 3] = 0
    x[2, 4] = 1

    x[-1] = 0
    # x should have these probs
    prob_x_check = torch.tensor([0.0232, 0.0018, 0.0101, 0.0014])

    test = _sigma.repeat(4, 3, 1, 1)
    logits = torch.ones(4, 3) / 3

    B = torch.tensor(
        [
            [-742.1493, -708.8158, -188.6618, -13.4060, 77.3592],
            [-1638.5694, -241.5870, -93.3107, -273.7858, 557.2550],
            [-45.4633, -764.7070, -464.6229, 996.7072, -193.7594],
            [-85.0106, -38.7596, -1180.5608, -1357.5016, 739.2887],
            [32.9458, -257.1105, -198.2849, -793.8524, -625.4517],
        ],
        dtype=torch.float64,
    )
    C = torch.tensor(
        [
            [465.4584, 1168.5946, 1099.2468, 865.9058, 154.5655],
            [80.9033, -874.9588, -1319.2415, 368.8852, -485.5291],
            [545.7150, 367.9474, -160.2272, -1724.7222, 1418.3287],
            [147.9080, 48.9294, 1807.2269, -171.5253, -1192.0453],
            [112.6032, 571.6511, 1312.5974, 2075.5068, -237.1650],
        ],
        dtype=torch.float64,
    )

    sigma_init_true = torch.tensor(
        [
            [0.8383, 0.1722, 0.0960, 0.0459, -0.0278],
            [0.1788, 0.4447, -0.1311, 0.0325, -0.0716],
            [0.0824, 0.1216, 0.6990, -0.2666, 0.0773],
            [0.0145, 0.0449, 0.2760, 0.7009, -0.2277],
            [0.0141, 0.0728, 0.0692, 0.2159, 0.7614],
        ],
        dtype=torch.float64,
    )

    mograss = GrassmannConditional(
        features=5,
        hidden_features=30,
        hidden_net=None,
        num_components=3,
    )

    p_test = mograss.prob_mograssmann(x, logits, test)
    assert torch.all(torch.isclose(p_test, prob_x_check, atol=1e-4))

    sigma_init = mograss.compute_sigma(B.view(1, 1, 5, 5), C.view(1, 1, 5, 5))
    assert torch.all(torch.isclose(sigma_init, sigma_init_true, atol=1e-4))
