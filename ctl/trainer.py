from math import sqrt
import copy
from random import choice
from pathlib import Path
from shutil import rmtree
from PIL import Image
import os
from beartype.typing import Union, List, Optional, Tuple
from typing_extensions import Annotated

from beartype import beartype
from beartype.door import is_bearable
from beartype.vale import Is

import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from einops import rearrange

from ema_pytorch import EMA

from PIL import Image
from torchvision import transforms
from core.ofa import OFATokenizer
from core.ofa.modeling_ofa import OFAModelForVQA
from core.optimizer import get_optimizer
from core.ofa.generate import sequence_generator
# from core.datasets.file_dataset import FileDataset

from core.datasets.vqa_gen_dataset import VqaGenDataset , VQACollator, VqaDataset
from transformers import (
    TrainingArguments,
    set_seed,
    Trainer,
)
from transformers import AdamW, get_scheduler
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from accelerate import DistributedType
import wandb

import transformers
from core.ofa.label_smoothed_cross_entropy import AdjustLabelSmoothedCrossEntropyCriterion


# helpers

def exists(val):
	return val is not None

def noop(*args, **kwargs):
	pass

def cycle(dl):
	while True:
		for data in dl:
			yield data

def cast_tuple(t):
	return t if isinstance(t, (tuple, list)) else (t,)

def yes_or_no(question):
	answer = input(f'{question} (y/n) ')
	return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
	for key, new_value in new_logs.items():
		old_value = log.get(key, 0.)
		log[key] = old_value + new_value
	return log

# auto data to module keyword argument routing functions

def has_duplicates(tup):
	counts = dict()
	for el in tup:
		if el not in counts:
			counts[el] = 0
		counts[el] += 1
	return any(filter(lambda count: count > 1, counts.values()))

def determine_types(data, config):
	output = []
	for el in data:
		for name, data_type in config.items():
			if is_bearable(el, data_type):
				output.append(name)
				break
		else:
			raise TypeError(f'unable to determine type of {data}')

	return tuple(output)

# main trainer class

class VQATrainer(nn.Module):
	def __init__(
		self,
		vqa_model: OFAModelForVQA,
		tokenizer:OFATokenizer,
		args,
		training_args,
		wandb_every = 100,
		apply_grad_penalty_every = 4,
		valid_frac = 0.01,
		max_grad_norm = 0.5,
		accelerate_kwargs: dict = dict(),
	):
		super().__init__()

		kwargs = DistributedDataParallelKwargs(find_unused_parameters = True)
		self.accelerator = Accelerator()

		transformers.set_seed(42)


		if self.is_main:
			wandb.login()
			wandb.init(project="ofa_vqa")

		self.results_folder = Path(training_args.output_dir)
		self.results_folder.mkdir(parents = True, exist_ok = True)

		self.register_buffer('steps', torch.Tensor([0]))
		self.vqa_model = vqa_model

		# tsv_dataset = FileDataset(args.data_folder, [0,5,2,3,4])
		# self.ds = VqaGenDataset(
		# 	split = "train",
		# 	dataset = tsv_dataset,
		# 	max_src_length=128,
		# 	max_object_length=30,
		# 	max_tgt_length=30,
		# 	patch_image_size=224,
		# 	add_object=False,
		# 	imagenet_default_mean_and_std=False,
		# 	prompt_type="none"
		# )

		
		self.ann_file = '/srv/scratch/sanisetty3/DLM/AliceMind/mPLUG/data/json/vqa_ocr_object/vqa_train_ocr.json'
		self.vqa_root = '/srv/datasets/coco/'
		self.ds = VqaDataset(
			ann_file=self.ann_file,
			vqa_root=self.vqa_root,
		)

		self.num_train_steps = training_args.num_train_epochs * len(self.ds)
		self.batch_size = training_args.per_device_train_batch_size
		self.grad_accum_every = training_args.gradient_accumulation_steps

		self.loss_fnc = AdjustLabelSmoothedCrossEntropyCriterion()
		self.optim = get_optimizer(self.vqa_model.parameters(), lr = training_args.learning_rate, wd = training_args.weight_decay)
		self.lr_scheduler = get_scheduler(
			name = training_args.lr_scheduler_type,
			optimizer=self.optim,
			num_warmup_steps=training_args.warmup_steps,
			num_training_steps=self.num_train_steps,
		)

		self.max_grad_norm = max_grad_norm


		
		data_collator = VQACollator(tokenizer=tokenizer, max_seq_length=args.max_seq_length)

		# split for validation

		if valid_frac > 0:
			train_size = int((1 - valid_frac) * len(self.ds))
			valid_size = len(self.ds) - train_size
			self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(42))
			self.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
		else:
			self.valid_ds = self.ds
			self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

		# dataloader

		data_collator = VQACollator(tokenizer=tokenizer, max_seq_length=self.max_seq_length)
		

		self.dl = DataLoader(self.ds, batch_size = self.batch_size, collate_fn=data_collator ,num_workers = training_args.dataloader_num_workers, shuffle = True)

		self.valid_dl = DataLoader(self.valid_ds, batch_size = self.batch_size, collate_fn=data_collator,num_workers = training_args.dataloader_num_workers, shuffle = False)

		# prepare with accelerator

		(
			self.vqa_model,
			self.optim,
			self.dl,
			self.valid_dl
		) = self.accelerator.prepare(
			self.vqa_model,
			self.optim,
			self.dl,
			self.valid_dl
		)

		self.dl_iter = cycle(self.dl)
		self.valid_dl_iter = cycle(self.valid_dl)

		self.save_model_every = training_args.save_steps
		# self.save_results_every = training_args.save_steps
		self.log_losses_every = training_args.logging_steps
		self.wandb_every = wandb_every

		self.apply_grad_penalty_every = apply_grad_penalty_every

		hps = {"num_train_steps": self.num_train_steps, "max_seq_length": args.max_seq_length, "learning_rate": training_args.learning_rate}
		self.accelerator.init_trackers("ofa_vqa", config=hps)        


		# trainer = Trainer(
		# 	model=vqa_model,
		# 	args=training_args,
		# 	train_dataset=self.ds,
		# 	eval_dataset = self.valid_ds,
		# 	data_collator=data_collator,
		# 	tokenizer=tokenizer
		# )

		# train_result = trainer.train()
		# metrics = train_result.metrics
		# trainer.log_metrics("train", metrics)
		# trainer.save_metrics("train", metrics)
		# trainer.save_state()
		# trainer.save_model(os.path.join(training_args.output_dir, 'checkpoint-final'))
			
	@property
	def device(self):
		return self.accelerator.device

	@property
	def is_distributed(self):
		return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

	@property
	def is_main(self):
		return self.accelerator.is_main_process

	@property
	def is_local_main(self):
		return self.accelerator.is_local_main_process

	def save(self, path):
		pkg = dict(
			model = self.accelerator.get_state_dict(self.vqa_model),
			optim = self.optim.state_dict(),
			steps = self.steps
		)

		if self.use_ema:
			pkg['ema_model'] = self.ema_vqa_model.state_dict()


		torch.save(pkg, path)

	@property
	def unwrapped_vqa_model(self):
		return self.accelerator.unwrap_model(self.vqa_model)

	def load(self, path):
		path = Path(path)
		assert path.exists()
		pkg = torch.load(str(path), map_location = 'cpu')

		self.unwrapped_vqa_model.load_state_dict(pkg['model'])

		if self.use_ema:
			assert 'ema_model' in pkg
			self.ema_vqa_model.load_state_dict(pkg['ema_model'])

		self.optim.load_state_dict(pkg['optim'])
		self.steps = pkg["steps"]
		print("starting at step: ", self.steps)



	def train_step(self):

		steps = int(self.steps.item())
		apply_grad_penalty = self.apply_grad_penalty_every > 0 and not (steps % self.apply_grad_penalty_every)
		log_losses = self.log_losses_every > 0 and not (steps % self.log_losses_every)

		self.vqa_model.train()

		# logs

		logs = {}

		# update vae (generator)

		for _ in range(self.grad_accum_every):
			batch = next(self.dl_iter)

			loss, sample_size, logging_output = self.loss_fnc(self.vqa_model,batch,steps)
			
			self.accelerator.backward(loss / self.grad_accum_every)

			accum_log(logs, dict(loss = loss.item() / self.grad_accum_every,))

			if log_losses:
				accum_log(logs, logging_output)

		if exists(self.max_grad_norm):
			self.accelerator.clip_grad_norm_(self.vqa_model.parameters(), self.max_grad_norm)

		self.optim.step()
		self.lr_scheduler.step()
		self.optim.zero_grad()

		# for _ in range(self.grad_accum_every):
		# 	wave, = next(self.dl_iter)

		# 	# print(wave.shape)
		# 	# wave = wave.to(device)

		# 	discr_losses = self.soundstream(
		# 		wave,
		# 		apply_grad_penalty = apply_grad_penalty,
		# 		return_discr_loss = True,
		# 		return_discr_losses_separately = True
		# 	)

		# 	for name, discr_loss in discr_losses:
		# 		self.accelerator.backward(discr_loss / self.grad_accum_every, retain_graph = True)
		# 		accum_log(logs, {name: discr_loss.item() / self.grad_accum_every})

		# build pretty printed losses

		losses_str = f"{steps}: ofa model total loss: {logs['loss']:.3f}"
		if log_losses:
			self.accelerator.log({
				"total_loss": logs['loss'],
				"nll_loss": logs['nll_loss'],
			}, step=steps)

		# log
		if self.is_main and (steps%self.wandb_every == 0):
			for key , value in logs.items():
				wandb.log({f'train_loss/{key}': value})           

		self.print(losses_str)

		# if self.is_main and (steps % self.save_results_every == 0):
		# 	models = [(self.unwrapped_soundstream, str(steps))]
		# 	if self.use_ema:
		# 		models.append((self.ema_soundstream.ema_model if self.use_ema else self.unwrapped_soundstream, f'{steps}.ema'))

		# 	wave, = next(self.valid_dl_iter)
		# 	wave = wave

		# 	for model, label in models:
		# 		model.eval()

		# 		with torch.no_grad():
		# 			recons = model(wave, return_recons_only = True)

		# 		for ind, recon in enumerate(recons.unbind(dim = 0)):
		# 			os.makedirs(os.path.join(self.results_folder , "samples" ) , exist_ok=True)
		# 			filename = (os.path.join(self.results_folder , "samples" , f'sample_{label}.flac'))
		# 			torchaudio.save(filename, recon.cpu().detach(), self.unwrapped_soundstream.target_sample_hz)

		# 	self.print(f'{steps}: saving sample to {str(os.path.join(self.results_folder , "samples" ))}')

		# save model every so often
		
		if self.is_main and not (steps % self.save_model_every):
			os.makedirs(os.path.join(self.results_folder , "results" ) , exist_ok=True)
			model_path = os.path.join(self.results_folder , "results" ,  f'ofa_vqa.{steps}.pt')
			self.save(model_path)

			self.print(f'{steps}: saving model to {str(os.path.join(self.results_folder , "results" ) )}')


		self.steps += 1
		return logs

	def train(self, resume = False, log_fn = noop):


		if resume:
			save_path = os.path.join(self.results_folder , "results")
			chk = sorted(os.listdir(save_path) , key = lambda x: int(x.split('.')[1]))[-1]
			print("resuming from ", os.path.join(save_path , chk))
			self.load(os.path.join(save_path , chk))

		while self.steps < self.num_train_steps:
			logs = self.train_step()
			log_fn(logs)

		self.print('training complete')
