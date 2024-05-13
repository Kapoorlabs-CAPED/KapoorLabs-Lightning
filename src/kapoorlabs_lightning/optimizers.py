from typing import Optional

from torch import optim
from torch.nn.modules.module import Module

__all__ = ["Adam", "SGD", "Rprop", "RMSprop"]


class _Optimizer(Module):
    def __init__(
        self,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        momentum=0,
        dampening=0,
        nesterov=False,
        weight_decay=0,
        amsgrad=False,
        alpha=0.99,
        centered=False,
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
        etas=(0.5, 1.2),
        step_sizes=(1e-6, 50)
    ):
        super().__init__()

        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.amsgrad = amsgrad
        self.alpha = alpha
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.foreach = foreach
        self.maximize = maximize
        self.capturable = capturable
        self.differentiable = differentiable
        self.fused = fused
        self.dampening = dampening
        self.nesterov = nesterov
        self.etas = etas
        self.step_sizes = step_sizes
        self.centered = centered


class Adam(_Optimizer):
    def __init__(
        self,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None
    ):
        super().__init__(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )

    def forward(self, params):
        return optim.Adam(
            params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )


class SGD(_Optimizer):
    def __init__(
        self,
        lr=1e-3,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False
    ):
        super().__init__(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
        )

    def forward(self, params):
        return optim.SGD(
            params,
            lr=self.lr,
            momentum=self.momentum,
            dampening=self.dampening,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov,
        )


class RMSprop(_Optimizer):
    def __init__(
        self,
        lr=1e-2,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0,
        momentum=0,
        centered=False,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        differentiable: bool = False,
    ):

        super().__init__(
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
            foreach=foreach,
            maximize=maximize,
            differentiable=differentiable,
        )

    def forward(self, params):
        return optim.RMSprop(
            params,
            lr=self.lr,
            alpha=self.alpha,
            eps=self.eps,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            centered=self.centered,
            foreach=self.foreach,
            maximize=self.maximize,
            differentiable=self.differentiable,
        )


class Rprop(_Optimizer):
    def __init__(
        self,
        lr=1e-2,
        etas=(0.5, 1.2),
        step_sizes=(1e-6, 50),
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        differentiable: bool = False
    ):
        super().__init__(lr=lr)
        self.etas = etas
        self.step_sizes = step_sizes
        self.foreach = foreach
        self.maximize = maximize
        self.differentiable = differentiable

    def forward(self, params):
        return optim.Rprop(
            params, lr=self.lr, etas=self.etas, step_sizes=self.step_sizes
        )
