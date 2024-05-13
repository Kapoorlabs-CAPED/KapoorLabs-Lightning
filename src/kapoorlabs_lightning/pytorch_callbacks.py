from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint, EarlyStopping

from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities import LightningEnum
from typing import Any, Dict, Optional
import psutil
import pprint
from lightning.pytorch.trainer.states import RunningStage
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateFinder
from lightning.pytorch.callbacks import DeviceStatsMonitor
from pathlib import Path
import os
from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional, Union, cast
from rich.style import Style


class Interval(LightningEnum):
    step = "step"
    epoch = "epoch"


class CustomDeviceStatsMonitor(DeviceStatsMonitor):
    def __init__(self, cpu_stats: Optional[bool] = None) -> None:
        super().__init__(cpu_stats=cpu_stats)
        self.device_monitor_callback = self


class ExponentialDecayCallback(Callback):
    def __init__(self, multiplier, n_epochs):
        self.multiplier = multiplier
        self.n_epochs = n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if current_epoch < self.n_epochs - 1:
            optimizer = trainer.optimizers[0]
            for param_group in optimizer.param_groups:
                if "weight_decay" in param_group:
                    param_group["weight_decay"] *= 1 / self.multiplier


class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)
            print(self.lr_finder.results)
            fig = self.lr_finder.plot(suggest=True)
            fig.show()


class SaveFilesCallback(Callback):
    def __init__(self, save_interval_epochs, root_dir):
        super().__init__()
        self.save_interval_epochs = save_interval_epochs
        self.root_dir = root_dir

    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Backing up metrics at Current epoch: {trainer.current_epoch}")
        epoch = trainer.current_epoch
        backup_dir = os.path.join(self.root_dir, "backup")
        Path(backup_dir).mkdir(parents=True, exist_ok=True)
        if epoch % self.save_interval_epochs == 0:
            latest_file = None
            latest_timestamp = 0

            for file in os.listdir(self.root_dir):
                if file.endswith(".npz"):
                    file_path = os.path.join(self.root_dir, file)
                    file_timestamp = os.path.getctime(file_path)

        if file_timestamp > latest_timestamp:
            latest_file = file
            latest_timestamp = file_timestamp
            if latest_file:
                latest_file_path = os.path.join(self.root_dir, latest_file)
                backup_file_path = os.path.join(backup_dir, latest_file)

                try:
                    os.replace(latest_file_path, backup_file_path)
                    print(f"Latest file {latest_file} backed up successfully!")
                except OSError as e:
                    if e.errno == 16:  # Device or resource busy
                        print(f"Latest file {latest_file} is busy. Cannot backup.")
                    else:
                        raise  # Re-raise any other OSError
            else:
                print("No files with the specified extension found.")


class EarlyStoppingCall(EarlyStopping):
    def __init__(
        self,
        monitor="val_accuracy",
        min_delta: float = 0.0,
        patience: int = 20,
        verbose: bool = True,
        mode: str = "max",
        strict: bool = False,
        check_finite: bool = True,
        stopping_threshold: Optional[float] = None,
        divergence_threshold: Optional[float] = None,
        check_on_train_epoch_end: Optional[bool] = None,
        log_rank_zero_only: bool = False,
    ):

        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            mode=mode,
            patience=patience,
            verbose=verbose,
            strict=strict,
            check_finite=check_finite,
            stopping_threshold=stopping_threshold,
            divergence_threshold=divergence_threshold,
            check_on_train_epoch_end=check_on_train_epoch_end,
            log_rank_zero_only=log_rank_zero_only,
        )
        self.early_stopping_callback = self



@dataclass
class RichProgressBarTheme:
    """Styles to associate to different base components.

    Args:
        description: Style for the progress bar description. For eg., Epoch x, Testing, etc.
        progress_bar: Style for the bar in progress.
        progress_bar_finished: Style for the finished progress bar.
        progress_bar_pulse: Style for the progress bar when `IterableDataset` is being processed.
        batch_progress: Style for the progress tracker (i.e 10/50 batches completed).
        time: Style for the processed time and estimate time remaining.
        processing_speed: Style for the speed of the batches being processed.
        metrics: Style for the metrics

    https://rich.readthedocs.io/en/stable/style.html

    """

    description: Union[str, Style] = "white"
    progress_bar: Union[str, Style] = "#6206E0"
    progress_bar_finished: Union[str, Style] = "#6206E0"
    progress_bar_pulse: Union[str, Style] = "#6206E0"
    batch_progress: Union[str, Style] = "white"
    time: Union[str, Style] = "grey54"
    processing_speed: Union[str, Style] = "grey70"
    metrics: Union[str, Style] = "white"
    metrics_text_delimiter: str = " "
    metrics_format: str = ".9f"

class CustomProgressBar(RichProgressBar):
    def __init__(
        self,
        description_color: str = "green_yellow",
        progress_bar_color: str = "green1",
        metrics_precision: int = 9,
        
    ):
        custom_theme = RichProgressBarTheme(
            description=description_color,
            progress_bar=progress_bar_color,
            metrics_format=f".{metrics_precision}f",
        )
        super().__init__(theme=custom_theme)
        self.progress_bar = RichProgressBar(theme=custom_theme)


class CheckpointModel(ModelCheckpoint):
    def __init__(
        self,
        save_dir,
        monitor=None,
        verbose=True,
        save_last=None,
        save_top_k=-1,
        save_weights_only=False,
        mode="min",
        auto_insert_metric_name=True,
        every_n_train_steps=None,
        train_time_interval=None,
        every_n_epochs=None,
        save_on_train_epoch_end=None,
    ):

        self._dirpath = save_dir

        super().__init__(
            dirpath=self._dirpath,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end,
        )
        self.checkpoint_callback = self

    @property
    def dirpath(self):
        return self._dirpath

    @dirpath.setter
    def dirpath(self, value):
        self._dirpath = value
        Path(self._dirpath).mkdir(parents=True, exist_ok=True)
        print(f"dirpath set to {self._dirpath}")


class CustomVirtualMemory(Callback):
    def __init__(
        self,
        interval: str = Interval.step,
        verbose: bool = True,
    ) -> None:
        super().__init__()

        self._vmem = psutil.virtual_memory()[3] / 1000000000
        self._interval = interval
        self._verbose = verbose
        self._start_vmem: Dict[RunningStage, Optional[float]] = {
            stage: None for stage in RunningStage
        }
        self._end_vmem: Dict[RunningStage, Optional[float]] = {
            stage: None for stage in RunningStage
        }
        self._offset = 0

    def start_vmem(self, stage: str = RunningStage.TRAINING) -> Optional[float]:
        stage = RunningStage(stage)
        return self._start_vmem[stage]

    def end_vmem(self, stage: str = RunningStage.TRAINING) -> Optional[float]:
        stage = RunningStage(stage)
        return self._end_vmem[stage]

    def vmem_elapsed(self, stage: str = RunningStage.TRAINING) -> float:
        start = self.start_vmem(stage)
        end = self.end_vmem(stage)
        offset = self._offset if stage == RunningStage.TRAINING else 0
        if start is None:
            return offset
        if end is None:
            return psutil.virtual_memory()[3] / 1000000000 - start + offset
        return end - start + offset

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self._start_vmem[RunningStage.TRAINING] = (
            psutil.virtual_memory()[3] / 1000000000
        )
        self._start_vmem[RunningStage.TRAINING] = self._start_vmem[
            RunningStage.TRAINING
        ]
        pprint.pprint(
            f"RAM in use on train start {self._start_vmem[RunningStage.TRAINING]} GB"
        )

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self._end_vmem[RunningStage.TRAINING] = psutil.virtual_memory()[3] / 1000000000
        self._end_vmem[RunningStage.TRAINING] = self._end_vmem[RunningStage.TRAINING]
        pprint.pprint(
            f"RAM in use on train end {self._end_vmem[RunningStage.TRAINING]} GB"
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "vmem_in_use": {stage.value: self.end_vmem(stage) for stage in RunningStage}
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        vmem_in_use = state_dict.get("vmem_in_use", {})
        self._offset = vmem_in_use.get(RunningStage.TRAINING.value, 0)
