from torch import optim
from torch.nn.modules.module import Module


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
    def __init__(
        self, factor=1.0 / 3, total_iters=5, last_epoch=-1, verbose=False
    ):
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
