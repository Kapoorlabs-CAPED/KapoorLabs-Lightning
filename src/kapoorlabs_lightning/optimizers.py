from typing import Optional
import torch
from torch import optim
from torch.nn.modules.module import Module
from lightly.utils.lars import LARS as LightlyLARS

__all__ = ["Adam", "SGD", "Rprop", "RMSprop", "AdamW", "AdamWClipStyle", "LARS"]


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
            amsgrad=self.amsgrad
           
        )
class AdamW(_Optimizer):
    def __init__(
        self,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
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
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )

    def forward(self, params):
        return optim.AdamW(
            params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            foreach=self.foreach,
            maximize=self.maximize,
            capturable=self.capturable,
            differentiable=self.differentiable,
            fused=self.fused,
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
            nesterov=self.nesterov
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
         
         super().__init__(lr=lr,
                          alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
            foreach=foreach,
            maximize=maximize,
            differentiable=differentiable)
    
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
            differentiable=self.differentiable
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
            params,
            lr=self.lr,
            etas=self.etas,
            step_sizes=self.step_sizes
            
        )
        
class AdamWClipStyle(_Optimizer):
    def __init__(
        self,
        lr=5e-4,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.2,
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        super().__init__(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )

    def forward(self, params):
        # If user passed a model instead of a param list
        if isinstance(params, torch.nn.Module):
            decay_params = []
            no_decay_params = []

            norm_layers = (
                torch.nn.LayerNorm,
                torch.nn.BatchNorm1d,
                torch.nn.BatchNorm2d,
                torch.nn.BatchNorm3d,
                torch.nn.GroupNorm,
                torch.nn.LocalResponseNorm,
            )

            for module in params.modules():
                for name, param in module.named_parameters(recurse=False):
                    if not param.requires_grad:
                        continue
                    if name == "bias" or isinstance(module, norm_layers):
                        no_decay_params.append(param)
                    else:
                        decay_params.append(param)

            param_groups = [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ]
        else:
            param_groups = params  # treat as pre-defined param group(s)

        return optim.AdamW(
            param_groups,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=0.0,  # Handled via param_groups
            foreach=self.foreach,
            maximize=self.maximize,
            capturable=self.capturable,
            differentiable=self.differentiable,
            fused=self.fused,
        )


class LARS(_Optimizer):
    """
    LARS optimizer wrapper following the module pattern.

    Layer-wise Adaptive Rate Scaling for large batch training.
    Recommended for SimCLR with batch sizes >= 256.
    Uses lightly.utils.lars.LARS under the hood.

    Args:
        lr: Base learning rate (default: 0.1)
        momentum: Momentum factor (default: 0.9)
        weight_decay: Weight decay (default: 1e-6)
        eta: LARS coefficient (trust coefficient) (default: 0.001)
        eps: Small constant for numerical stability (default: 1e-8)
    """
    def __init__(
        self,
        lr=0.1,
        momentum=0.9,
        dampening: float = 0,
        weight_decay=1e-6,
        eps=1e-8,
        nesterov=False,
        trust_coefficient: float = 0.001,
    ):
        super().__init__(lr=lr, momentum=momentum, weight_decay=weight_decay, eps=eps, nesterov = nesterov, dampening = dampening)
        self.trust_coefficient = trust_coefficient
    def forward(self, params):
        if isinstance(params, torch.nn.Module):
            params = params.parameters()

        return LightlyLARS(
            params,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov,
            trust_coefficient=self.trust_coefficient,
            eps = self.eps
        )