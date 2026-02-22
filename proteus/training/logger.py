# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		io_utils.py
description:	utility classes for input/output operations during training 
'''
# ----------------------------------------------------------------------------------------------------------------------

from pathlib import Path
import textwrap
import logging
import torch
import math
import sys
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
import warnings
import shutil 
from dataclasses import dataclass
from hydra.core.hydra_config import HydraConfig
import mlflow
import os

# ----------------------------------------------------------------------------------------------------------------------

@dataclass
class LoggerCfg:
	run_name: str = ""
	experiment_name: str = "debug"
	overwrite: bool = False
	log_system_metrics: bool = True
	system_metrics_sample_interval: int = 10 # seconds between each system metric collection
	system_metrics_log_interval: int = 1 # number of samples to collect before aggregating
	log_interval: int = 10 # number of steps between each training metric log, metrics are accumulated over this period

class Logger():

	def __init__(self, cfg: LoggerCfg):

		self.out_path = Path(HydraConfig.get().runtime.output_dir)
		self.log = logging.getLogger(__name__)
		self.log_interval = cfg.log_interval
		
		host = os.environ.get("MLFLOW_HOST", "localhost")
		port = os.environ.get("MLFLOW_PORT", "5000")
		mlflow.set_tracking_uri(f"http://{host}:{port}")
		mlflow.set_experiment(cfg.experiment_name)
		if cfg.log_system_metrics:
			mlflow.enable_system_metrics_logging()
			mlflow.set_system_metrics_sampling_interval(cfg.system_metrics_sample_interval)
			mlflow.set_system_metrics_samples_before_logging(cfg.system_metrics_log_interval)
		else:
			mlflow.disable_system_metrics_logging()

		self.run_name = cfg.run_name if cfg.run_name else None

	def log_step(self, step_metrics, step):
		mlflow.log_metrics(metrics=step_metrics, step=step)

	def log_param(self, key, val):
		mlflow.log_param(key, val)
	
# ----------------------------------------------------------------------------------------------------------------------