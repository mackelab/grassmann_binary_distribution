import torch

from grassmann_distribution.GrassmannDistribution import GrassmannBinary
from grassmann_distribution.utils import check_valid_sigma

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


def test():
    gr = GrassmannBinary(_sigma)

    assert torch.allclose(gr.lambd, lambd_check, atol=1e-4)
    assert torch.allclose(gr.prob(x), prob_x_check, atol=1e-4)
    assert check_valid_sigma(_sigma)


# assert torch.allclose(gr.lambd, lambd_check, atol=1e-4)
# assert torch.allclose(gr.prob(x), prob_x_check, atol=1e-4)
# assert check_valid_sigma(_sigma)
