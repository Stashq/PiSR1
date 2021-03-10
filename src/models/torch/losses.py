import torch
from torch.nn.modules.loss import _Loss


class L1Loss(_Loss):

    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction='mean',
        rate: float = 0.01
    ):
        super(L1Loss, self).__init__(size_average, reduce, reduction)
        self.RATE = rate

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor = None
    ) -> torch.Tensor:

        loss = input.abs()
        loss = loss.sum()
        loss *= self.RATE

        return loss


class L2Loss(_Loss):

    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction='mean',
        rate: float = 0.01
    ):
        super(L2Loss, self).__init__(size_average, reduce, reduction)
        self.RATE = rate

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor = None
    ) -> torch.Tensor:

        loss = input ** 2
        loss = loss.sum()
        loss *= self.RATE

        return loss
