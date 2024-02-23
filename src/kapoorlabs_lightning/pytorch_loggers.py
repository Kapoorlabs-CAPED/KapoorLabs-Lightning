import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torch import Tensor
from torch.nn import Module


class CustomNPZLogger(Logger):
    def __init__(self, save_dir: str, experiment_name: str = "BLT", version: str = "1"):
        super().__init__()
        self._experiment = None
        self._save_dir = save_dir
        self._experiment_name = experiment_name
        self._version = version
        self.finalized = False
        self.hparams_logged = None
        self.after_save_checkpoint_called = False
        self.metrics_logged = {}
        # make sure savedir exists
        if not os.path.exists(self._save_dir):
            print(f"Creating save directory {self._save_dir} for custom npz logger.")
            os.makedirs(self._save_dir)

    @property
    def name(self):
        return "NPZLogger"

    @rank_zero_experiment
    def experiment(self):
        if self._experiment is not None:
            return self._experiment
        if self._experiment_name:
            self._experiment.set_name(self._experiment_name)

    @rank_zero_only
    def log_hyperparams(self, params):
        self.hparams_logged = params
        

    @rank_zero_only
    def log_metrics(self, metrics, step):
        if not hasattr(self, "metrics_logged"):
            self.metrics_logged = {}

        for key, value in metrics.items():
            if key not in self.metrics_logged:
                self.metrics_logged[key] = {"steps": [], "values": []}
            self.metrics_logged[key]["steps"].append(step)
            self.metrics_logged[key]["values"].append(value)

    @rank_zero_only
    def finalize(self, status):
        self.finalized_status = status

    @rank_zero_only
    def save(self):
        save_experiment = self._save_dir
        save_experiment = os.fspath(save_experiment)
        save_experiment_name = str(self._experiment_name) + ".npz"
        save_path = Path(save_experiment) / save_experiment_name
        np.savez(save_path, **self.metrics_logged)

    @property
    def version(self):
        return self._version

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_experiment_key"] = (
            self._experiment.id if self._experiment is not None else None
        )
        state["_experiment"] = None
        return state

    def log_graph(self, model: Module, input_array: Optional[Tensor] = None) -> None:
        if self._experiment is not None:
            self._experiment.set_model_graph(model)
