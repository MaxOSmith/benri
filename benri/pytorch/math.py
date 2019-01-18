""" Simple math utility operations. """
import torch


def stable_softmax(x, dim=1):
    """ Numerically stable softmax.

    :param x: Logits.
    :return: Softmaxed logits.
    """
    z = x - x.max(dim=dim, keepdim=True)[0]
    numerator = torch.exp(z)
    denominator = torch.sum(numerator, dim=dim, keepdim=True)
    softmax = numerator / denominator
    return softmax
