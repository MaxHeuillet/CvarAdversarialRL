from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

from tianshou.utils.net.discrete import ImplicitQuantileNetwork


class ImplicitQuantileNetworkCVaR(ImplicitQuantileNetwork):
    """Implicit Quantile Network with CVaR.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param int action_dim: the dimension of action space.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int num_cosines: the number of cosines to use for cosine embedding.
        Default to 64.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.
    :param float alpha: the target quantile in the CVaR_\alpha objective.

    .. note::

        Although this class inherits Critic, it is actually a quantile Q-Network
        with output shape (batch_size, action_dim, sample_size).

        The second item of the first return value is tau vector.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int] = (),
        num_cosines: int = 64,
        preprocess_net_output_dim: Optional[int] = None,
        device: Union[str, int, torch.device] = "cpu",
        alpha: float = 1.0,  ) -> None:
        assert 0.0 < alpha <= 1.0, "alpha should be in (0, 1]"
        self.alpha = alpha
        super().__init__(preprocess_net, action_shape, hidden_sizes, num_cosines, preprocess_net_output_dim, device)

    def forward(  # type: ignore
        self, s: Union[np.ndarray, torch.Tensor], sample_size: int, **kwargs: Any ) -> Tuple[Any, torch.Tensor]:
        r"""Mapping: s -> Q(s, \*)."""
        logits, h = self.preprocess(s, state=kwargs.get("state", None))
        # Sample fractions.
        batch_size = logits.size(0)
        taus = torch.rand(batch_size, sample_size, dtype=logits.dtype, device=logits.device)

        # CVaR addition
        taus = taus * self.alpha

        embedding = (logits.unsqueeze(1) * self.embed_model(taus)).view(batch_size * sample_size, -1)
        out = self.last(embedding).view(batch_size, sample_size, -1).transpose(1, 2)
        return (out, taus), h


class IdentityNet(nn.Module):
    def __init__( self, input_dim: int, device=None, ) -> None:
        super().__init__()
        self.input_dim = np.prod(input_dim)
        self.output_dim = self.input_dim
        self.device = device

    def forward(self, s: Union[np.ndarray, torch.Tensor], state) -> torch.Tensor:
        return torch.as_tensor(s, device=self.device, dtype=torch.float32), state
