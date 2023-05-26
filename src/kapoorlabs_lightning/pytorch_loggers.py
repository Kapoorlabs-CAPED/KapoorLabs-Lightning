import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torch import Tensor
from torch.nn import Module


class CustomNPZLogger(Logger):

    def __init__(self, save_dir:str, experiment_name: str = 'Autoencoder', name: str = 'Autoencoder_net', version: str = "1"):

         super().__init__() 
         self._experiment = None
         self._save_dir: Optional[str]
         self.rest_api_key: Optional[str]
         self.train_loss_epoch = []
         self.val_loss_epoch = []
         self.train_loss_step = []
         self.val_loss_step = [] 
         self.val_accuracy_field = []
         
         self._experiment_name: Optional[str] = experiment_name 
         self._save_dir = save_dir
         self._name = name 
         self._version = version
         self.hparams_logged = None
         self.metrics_logged = {}
         self.finalized = False
         self.after_save_checkpoint_called = False 


    @property
    @rank_zero_experiment
    def experiment(self):
        if self._experiment is not None:
            return self._experiment
        if self._experiment_name:
            self._experiment.set_name(self._experiment_name)

    @rank_zero_only
    def log_hyperparams(self, params):
        self.hparams_logged = params
        self.experiment.log_hparams(params)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        self.metrics_logged = metrics

        if 'train_loss_step' in self.metrics_logged:
          self.train_loss_step.append([step, metrics['train_loss_step']])
        if 'train_loss_epoch' in self.metrics_logged:
          self.train_loss_epoch.append([step, metrics['train_loss_epoch']])
        if 'val_loss_step' in self.metrics_logged:
          self.val_loss_step.append([step, metrics['val_loss_step']])
        if 'val_loss_epoch' in self.metrics_logged:
          self.val_loss_epoch.append([step, metrics['val_loss_epoch']])
        if 'val_accuracy' in self.metrics_logged:  
             self.val_accuracy_field.append([step, metrics['val_accuracy']])


    @rank_zero_only
    def finalize(self, status):
        self.finalized_status = status

    @rank_zero_only
    def save(self):
        save_experiment = os.path.join(self._save_dir, self._experiment_name)
        Path(os.path.join(save_experiment))
        np.savez(save_experiment+self._name+'.npz', train_loss_step=self.train_loss_step , train_loss_epoch=self.train_loss_epoch, val_loss_step=self.val_loss_step,  val_loss_epoch = self.val_loss_epoch )

    @property
    def save_dir(self) -> Optional[str]:
        """Return the root directory where experiment logs get saved, or `None` if the logger does not save data
        locally."""
        return self._save_dir

  
    
    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return self._version

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()

        # Save the experiment id in case an experiment object already exists,
        # this way we could create an ExistingExperiment pointing to the same
        # experiment
        state["_experiment_key"] = self._experiment.id if self._experiment is not None else None

        # Remove the experiment object as it contains hard to pickle objects
        # (like network connections), the experiment object will be recreated if
        # needed later
        state["_experiment"] = None
        return state

    def log_graph(self, model: Module, input_array: Optional[Tensor] = None) -> None:
        if self._experiment is not None:
            self._experiment.set_model_graph(model)
