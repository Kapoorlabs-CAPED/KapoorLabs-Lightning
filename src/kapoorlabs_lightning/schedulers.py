from torch import optim
from torch.nn.modules.module import Module
from torch.optim.lr_scheduler import _LRScheduler

__all__ = [
    "ReduceLROnPlateau",
    "ConstantLR",
    "LinearLR",
    "MultiStepLR",
    "WarmCosineAnnealingLR",
    "CosineAnnealingScheduler",
    "CosineScheduler",
    "ExponentialLR",
    "SameLR",
    "NoScheduler",
]


class _Schedulers(Module):
    def __init__(
        self,
        mode="min",
        factor=0.1,
        patience=10,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-8,
        verbose=False,
        start_factor=1.0 / 3,
        end_factor=1.0,
        total_iters=5,
        last_epoch=-1,
        milestones=[15, 30, 45, 60, 75, 90, 105, 120],
        gamma=0.1,
        restart_epochs=10,
        eta_min=0,
        t_max=10,
        t_warmup=0,
        total_steps=1,
    ):
        super().__init__()

        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        self.verbose = verbose
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        self.last_epoch = last_epoch
        self.milestones = milestones
        self.gamma = gamma
        self.restart_epochs = restart_epochs
        self.eta_min = eta_min
        self.t_max = t_max
        self.t_warmup = t_warmup
        self.total_steps = total_steps


class CosineScheduler(_Schedulers):
    def __init__(self, restart_epochs=10):
        super().__init__(restart_epochs=restart_epochs)

    def forward(self, optimizer):
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, self.restart_epochs, verbose=True
        )


class NoScheduler(_LRScheduler):
    def __init__(self, optimizer):
        super().__init__(optimizer)

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self, epoch=None):
        pass


class ExponentialLR(_Schedulers):
    def __init__(self, gamma=0.1):
        super().__init__(gamma=gamma)

    def forward(self, optimizer):
        return optim.lr_scheduler.ExponentialLR(optimizer, self.gamma, verbose=True)


class SameLR(_Schedulers):
    def __init__(self):
        super().__init__()

    def forward(self, optimizer):
        return NoScheduler(optimizer)


class WarmCosineAnnealingLR(_Schedulers):
    def __init__(self, t_max, t_warmup=0, factor=1.0, eta_min=1.0e-10):
        super().__init__(t_max=t_max, t_warmup=t_warmup, factor=factor, eta_min=eta_min)

    def forward(self, optimizer):
        milestones = [self.t_warmup]
        schedulers = [
            optim.lr_scheduler.LinearLR(
                optimizer, start_factor=self.factor, total_iters=self.t_warmup
            ),
            optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.t_max, eta_min=self.eta_min, verbose=True
            ),
        ]
        return optim.lr_scheduler.SequentialLR(
            optimizer, schedulers, milestones, verbose=True
        )


class CosineAnnealingScheduler(_Schedulers):
    def __init__(self, t_max, t_warmup=0, eta_min=1.0e-10):
        super().__init__(t_max=t_max, t_warmup=t_warmup, eta_min=eta_min)

    def forward(self, optimizer):
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.t_max, eta_min=self.eta_min, verbose=True
        )


class MultiStepLR(_Schedulers):
    def __init__(self, milestones, gamma=0.1, last_epoch=-1, verbose=True):

        super().__init__(
            milestones=milestones, gamma=gamma, last_epoch=last_epoch, verbose=verbose
        )

    def forward(self, optimizer):
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.milestones,
            gamma=self.gamma,
            last_epoch=self.last_epoch,
            verbose=self.verbose,
        )


class ReduceLROnPlateau(_Schedulers):
    def __init__(
        self,
        mode="min",
        factor=0.1,
        patience=10,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-8,
        verbose=False,
    ):
        super().__init__(
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
            verbose=verbose,
        )

    def forward(self, optimizer):
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.mode,
            factor=self.factor,
            patience=self.patience,
            verbose=self.verbose,
            threshold=self.threshold,
            threshold_mode=self.threshold_mode,
            cooldown=self.cooldown,
            min_lr=self.min_lr,
            eps=self.eps,
        )


class ConstantLR(_Schedulers):
    def __init__(self, factor=1.0 / 3, total_iters=5, last_epoch=-1, verbose=False):
        super().__init__(
            factor=factor,
            total_iters=total_iters,
            last_epoch=last_epoch,
            verbose=verbose,
        )

    def forward(self, optimizer):
        return optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=self.factor,
            total_iters=self.total_iters,
            last_epoch=self.last_epoch,
        )


class LinearLR(_Schedulers):
    def __init__(
        self,
        start_factor=1.0 / 3,
        end_factor=1.0,
        total_iters=5,
        last_epoch=-1,
        verbose=False,
    ):
        super().__init__(
            self,
            start_factor=start_factor,
            end_factor=end_factor,
            total_iters=total_iters,
            last_epoch=last_epoch,
            verbose=verbose,
        )

    def forward(self, optimizer):
        return optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=self.start_factor,
            end_factor=self.end_factor,
            total_iters=self.total_iters,
            last_epoch=self.last_epoch,
        )
