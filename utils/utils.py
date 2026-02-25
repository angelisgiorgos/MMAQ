import os
import sys
from datetime import datetime
from dataset.transforms import Normalize
import numpy as np
import torch

def create_logdir(name: str, wandb_logger):
  basepath = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),'runs', name)
  run_name = wandb_logger.experiment.name
  if callable(run_name):
      run_name = run_name()
  
  if run_name is None:
      run_name = "run"
      
  logdir = os.path.join(basepath, run_name + "_" + str(datetime.now().strftime("%Y%m%d_%H%M%S")))
  if os.path.exists(logdir):
    raise Exception(f'Run {run_name} already exists. Please delete the folder {logdir} or choose a different run name.')
  os.makedirs(logdir,exist_ok=True)
  return logdir



def undo_normalization(pred, target, datastats):
  pred = Normalize.undo_no2_standardization(datastats, pred)
  target = Normalize.undo_no2_standardization(datastats, target)
  return pred, target

def none_or_true(x):
    if x == 'None':
        return None
    return bool(x)
