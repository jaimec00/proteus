# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		training_run.py
description:	runs training
'''
# ----------------------------------------------------------------------------------------------------------------------

import gc
import os
import time
import hydra
import mlflow
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig, MISSING

import torch

from proteus.data.data_loader import DataHolder, DataHolderCfg
from proteus.data.construct_registry import ConstructRegistry
from proteus.data.data_utils import DataBatch
from proteus.training.logger import Logger, LoggerCfg
from proteus.losses.training_loss import TrainingRunLosses, LossHolder, LossFnCfg
from proteus.training.optim import OptimCfg, setup_optim
from proteus.training.scheduler import SchedulerCfg, setup_scheduler
from proteus.utils.profiling import ProfilerCfg, Profiler
from proteus.utils.checkpoint_utils import load_model_cls, save_cfg, load_cfg
from proteus.types import Any, Dict, Iterator, Tuple

# ----------------------------------------------------------------------------------------------------------------------

# detect anomolies in training and dont tf32 matmuls 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

@dataclass
class TrainingParamsCfg:
	max_steps: int = 100_000        # Stop after this many steps
	val_interval: int = 1_000       # Run validation every N steps
	accumulation_steps: int = 1
	grad_clip_norm: float = 0.0
	compile_model: bool = False
	load_from_checkpoint: str = ""
	checkpoint_interval: int = 1_000
	gc_interval: int = 500

@dataclass
class TrainingRunCfg:
	model: Any = MISSING
	construct_function: str = MISSING
	data: DataHolderCfg = MISSING
	logger: LoggerCfg = MISSING
	losses: LossFnCfg = MISSING
	optim: OptimCfg = MISSING
	scheduler: SchedulerCfg = MISSING
	profiler: ProfilerCfg = MISSING
	training_params: TrainingParamsCfg = MISSING

class TrainingRun:

	def __init__(self, cfg: TrainingRunCfg) -> None:

		OmegaConf.resolve(cfg)

		self.gpu = torch.device('cuda' if torch.cuda.is_available() else "cpu")
		self.cpu = torch.device("cpu")

		# tells the data holder what data to create
		ConstructRegistry.set_construct_function(cfg.construct_function)

		self.data = DataHolder(cfg.data)
		self.losses = TrainingRunLosses(cfg.losses)
		self.logger = Logger(cfg.logger)
		self.profiler = Profiler(cfg.profiler, self.logger.out_path)

		self.max_steps = cfg.training_params.max_steps
		self.val_interval = cfg.training_params.val_interval
		self.accumulation_steps = cfg.training_params.accumulation_steps
		self.grad_clip_norm = cfg.training_params.grad_clip_norm
		self.batch_counter = 0
		self.epoch = 0
		self.gc_interval = cfg.training_params.gc_interval
		
		self.model, self.optim, self.scheduler, cfg = self.maybe_load_checkpoint(cfg)
		self.checkpoint_interval = cfg.training_params.checkpoint_interval

		if cfg.training_params.compile_model:
			self.log("compiling model...")
			self.model = torch.compile(self.model, dynamic=True)

		with mlflow.start_run():
			self.last_ts = time.perf_counter()
			self.last_logged_step = 0
			self.log_params(cfg)
			self.train()
			self.test()
			self.log("fin", fancy=True)

	def maybe_load_checkpoint(self, cfg: TrainingRunCfg) -> Tuple[Any, Any, Any, Any]:

		MODEL_YAML = "model_cfg.yaml"
		OPTIM_YAML = "optim_cfg.yaml"
		SCHEDULER_YAML = "scheduler_cfg.yaml"

		if cfg.training_params.load_from_checkpoint:

			weights_path = Path(cfg.training_params.load_from_checkpoint)
			checkpoint_path = weights_path.parent
			
			cfg.model = load_cfg(checkpoint_path / MODEL_YAML)
			if not cfg.training_params.reset_state:
				cfg.optim = load_cfg(checkpoint_path / OPTIM_YAML)
				cfg.scheduler = load_cfg(checkpoint_path / SCHEDULER_YAML)
		
			weights = torch.load(str(weights_path), map_location=self.cpu, weights_only=True)
			model_cls = load_model_cls(cfg.model)
			model = model_cls(cfg.model)
			model.load_state_dict(weights["model"], strict=False)

			optim = setup_optim(cfg.optim, model)
			if not cfg.training_params.reset_state:
				optim.load_state_dict(weights["optim"])
	
			scheduler = setup_scheduler(cfg.scheduler, optim)
			if not cfg.training_params.reset_state:
				scheduler.load_state_dict(weights["scheduler"])

			# TODO: handle dataloader being in sync with this

		else:
			model_cls = load_model_cls(cfg.model)
			model = model_cls(cfg.model)
			optim = setup_optim(cfg.optim, model)
			scheduler = setup_scheduler(cfg.scheduler, optim)

		# move to gpu
		model = model.to(self.gpu)
		for state in optim.state.values():
			for k, v in state.items():
				if torch.is_tensor(v):
					state[k] = v.to(self.gpu)

		# save the fully built cfgs 
		save_cfg(cfg.model, self.logger.out_path / MODEL_YAML)
		save_cfg(cfg.optim, self.logger.out_path / OPTIM_YAML)
		save_cfg(cfg.scheduler, self.logger.out_path / SCHEDULER_YAML)
		
		return model, optim, scheduler, cfg

	def maybe_save_checkpoint(self) -> None:
		if (
			not self.learn_step
			or self.last_step % self.checkpoint_interval != 0 
			or self.last_step < self.checkpoint_interval
		):
			return
		
		weights = {
			"model": self.model.state_dict(),
			"optim": self.optim.state_dict(),
			"scheduler": self.scheduler.state_dict(),
		}

		checkpoint_path = str(self.logger.out_path / f"checkpoint_step-{self.last_step:,}.pt")
		self.log(f"saved checkpoint to {checkpoint_path}")
		torch.save(weights, checkpoint_path)

	def set_training(self) -> None:
		self.model.train()

	def run_val(self) -> bool:
		return (
			self.last_step % self.val_interval == 0 
			and self.last_step > self.val_interval
			and self.learn_step
		)

	def get_batch(self, train_iter: Iterator) -> DataBatch:
		try:
			return next(train_iter), train_iter
		except StopIteration:
			self.epoch += 1
			train_iter = iter(self.data.train)
			return next(train_iter), train_iter

	def garbage_collect(self) -> None:
		if (
			self.gc_interval 
			and self.last_step >= self.gc_interval 
			and self.last_step % self.gc_interval == 0
		):
			torch.cuda.empty_cache()
			gc.collect()

	def train(self) -> None:

		self.log(f"initializing training for {self.max_steps} steps...")

		self.set_training()
		train_iter = iter(self.data.train)

		with self.profiler as profiler:
			with self.create_pbar(self.max_steps, "training progress") as pbar:
				while self.last_step < self.max_steps:

					# next batch
					data_batch, train_iter = self.get_batch(train_iter)

					# learn
					self.batch_learn(data_batch)

					# update progress bar
					if self.learn_step:
						self.update_pbar(pbar, data_batch)
						self.maybe_log_step()

					# profiler step
					profiler.step()

					# save the checkpoint
					self.maybe_save_checkpoint()

					# garbage collection
					self.garbage_collect()

					# validation at intervals
					if self.run_val():
						self.flush_train_metrics()
						self.validation()
						self.set_training()

		self.flush_train_metrics()
		self.log(f"training finished after {self.last_step} steps", fancy=True)

	@torch.no_grad()
	def validation(self) -> None:

		self.model.eval()
		self.losses.loss_holder.clear()
		val_start = time.perf_counter()

		with self.create_pbar(len(self.data.val), "validation progress") as pbar:
			for data_batch in self.data.val:
				self.batch_forward(data_batch)
				self.update_pbar(pbar, data_batch, step=data_batch.samples)

		# log accumulated val metrics to mlflow
		step_dict, _ = self._build_step_dict(self.losses.loss_holder, "val", start_ts=val_start)
		self.logger.log_step(step_dict, self.last_step)
		self.losses.loss_holder.clear()

	@torch.no_grad()
	def test(self) -> None:

		self.model.eval()
		self.log("starting testing", fancy=True)
		self.losses.loss_holder.clear()
		test_start = time.perf_counter()

		with self.create_pbar(len(self.data.test), "test progress") as pbar:
			for data_batch in self.data.test:
				self.batch_forward(data_batch)
				self.update_pbar(pbar, data_batch, step=data_batch.samples)

		# log accumulated test metrics to mlflow
		step_dict, _ = self._build_step_dict(self.losses.loss_holder, "test", start_ts=test_start)
		self.logger.log_step(step_dict, self.last_step)
		self.losses.loss_holder.clear()

	def batch_learn(self, data_batch: DataBatch) -> None:
		self.batch_forward(data_batch)
		self.batch_backward(data_batch)
		self.batch_counter += 1

	def batch_forward(self, data_batch: DataBatch) -> None:

		data_batch.move_to(self.gpu)
		outputs = self.model(data_batch)
		loss_output = self.losses.loss_fn(outputs)
		self.losses.loss_holder.add(loss_output)
		self.losses.loss_holder.add_data(data_batch)

	def batch_backward(self, data_batch: DataBatch) -> None:

		loss = self.losses.loss_holder.get_last_loss()
		loss.backward()

		if self.learn_step:

			# grad clip
			if self.grad_clip_norm:
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)

			# step
			self.optim.step()
			self.scheduler.step()
			self.optim.zero_grad()

	def _build_step_dict(self, tmp: LossHolder, prefix: str, start_ts: float | None = None) -> Tuple[Dict[str, float], float]:
		"""build a metrics dict from accumulated loss holder state, returns (metrics, timestamp)"""
		cur_ts = time.perf_counter()
		ref_ts = start_ts if start_ts is not None else self.last_ts
		delta_ts = cur_ts - ref_ts

		losses_dict = tmp.get_metrics()
		total_tokens = tmp.total_tokens
		total_loss_tokens = tmp.total_loss_tokens.item() if tmp.total_loss_tokens is not None else 0
		total_samples = tmp.total_samples
		num_batches = tmp.num_batches

		num_accumulated_batches = num_batches / max(1, self.accumulation_steps)
		data_dict = {
			"toks_per_batch": total_tokens / max(1, num_accumulated_batches),
			"loss_toks_per_batch": total_loss_tokens / max(1, num_accumulated_batches),
			"loss_toks_per_sample": total_loss_tokens / max(1, total_samples),
			"loss_toks_pct": total_loss_tokens / max(1, total_tokens),
			"samples_per_batch": total_samples / max(1, num_accumulated_batches),
			"toks_per_sample": total_tokens / max(1, total_samples),
		}
		throughput_dict = {
			"toks_per_sec": total_tokens / max(1e-6, delta_ts),
			"fwd_bwd_per_sec": num_batches / max(1e-6, delta_ts),
			"loss_toks_per_sec": total_loss_tokens / max(1e-6, delta_ts),
		}
		if prefix == "train":
			steps_in_period = self.last_step - self.last_logged_step
			throughput_dict["updates_per_sec"] = max(1, steps_in_period) / max(1e-6, delta_ts)

		losses_dict = {f"{prefix}/loss/{k}": v for k, v in losses_dict.items()}
		data_dict = {f"{prefix}/data/{k}": v for k, v in data_dict.items()}
		throughput_dict = {f"{prefix}/throughput/{k}": v for k, v in throughput_dict.items()}

		step_dict = losses_dict | data_dict | throughput_dict
		if prefix == "train":
			step_dict["train/lr"] = self.last_lr
		return step_dict, cur_ts

	def maybe_log_step(self) -> None:
		"""log training metrics if at a log interval boundary"""
		if self.last_step % self.logger.log_interval != 0:
			return
		self.flush_train_metrics()

	def flush_train_metrics(self) -> None:
		"""log accumulated training metrics and reset"""
		if self.losses.loss_holder.num_batches == 0:
			return
		step_dict, ts = self._build_step_dict(self.losses.loss_holder, "train")
		self.logger.log_step(step_dict, self.last_step)
		self.losses.loss_holder.clear()
		self.last_ts = ts
		self.last_logged_step = self.last_step

	def log_params(self, cfg: TrainingRunCfg) -> None:
		self.logger.log_param("configuration", OmegaConf.to_yaml(cfg))
		num_params = sum(p.numel() for p in self.model.parameters())
		self.logger.log_param("parameters", num_params)
		self.logger.log_param("run_dir", self.logger.out_path)


	def log(self, message: str, fancy=False) -> None:
		if fancy:
			message = f"\n\n{'-'*80}\n{message}\n{'-'*80}\n"
		self.logger.log.info(message)

	def create_pbar(self, total: int, desc: str) -> tqdm:
		"""Create a progress bar for training/validation/test loops."""
		return tqdm(total=total, desc=desc, unit="steps")

	def update_pbar(self, pbar: tqdm, data_batch, step=1) -> None:
		"""Update progress bar with loss and advance by 1 step."""
		if self.last_step % 10 == 0:
			pbar.set_postfix(loss=self.losses.loss_holder.get_last_loss())
		pbar.update(step)

	@property
	def last_lr(self) -> int:
		return self.scheduler.get_last_lr()[0]

	@property
	def last_step(self) -> int:
		'''just for clearer naming'''
		return int(self.scheduler.last_epoch)

	@property
	def learn_step(self) -> bool:
		return (self.batch_counter + 1) % self.accumulation_steps == 0
