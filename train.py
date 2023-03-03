
import argparse
import os
import numpy as np
import random
import time
import datetime
from pathlib import Path
import torch

from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
import json

from core.ofa.modeling_ofa import OFAModelForVQA
from core.ofa import OFATokenizer
from ctl.trainer import VQATrainer
from ctl.vqa_arguments import VQAArguments

def main(args_base ,args , training_args):


    if len(os.listdir(training_args.output_dir)) == 0:
        tokenizer = OFATokenizer.from_pretrained(args.pretrained)
        model = OFAModelForVQA.from_pretrained(args.pretrained, use_cache=False)

    else:
        tokenizer = OFATokenizer.from_pretrained(args.output)
        model = OFAModelForVQA.from_pretrained(args.output, use_cache=False)


    trainer = VQATrainer(
        model,
        tokenizer,
        args,
        training_args,
        wandb_every = args_base.wandb_every,
        
        

    

    ).cuda()
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', default="/srv/share4/sanisetty3/FMA/fma_large/" , help="folder with train and test data")
    parser.add_argument('--pretrained', default='/srv/scratch/sanisetty3/DLM/OFA_VQA/OFA-tiny')
    parser.add_argument('--resume', default=True, type = bool)
    parser.add_argument('--output_dir', default="/srv/scratch/sanisetty3/DLM/OFA_VQA/checkpoints")
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--fp16', default=True, type=bool)

    parser.add_argument('--per_device_train_batch_size', default=12, type=int,)
    parser.add_argument('--per_device_eval_batch_size', default=12, type=int,)
    parser.add_argument('--gradient_accumulation_steps', default=12, type=int,)

    parser.add_argument("--num_train_epochs",  default=100,type=int)
    parser.add_argument("--save_steps",  default=500,type=int)
    parser.add_argument("--logging_steps",  default=10,type=int)
    parser.add_argument("--wandb_every",  default=50,type=int)
    parser.add_argument("--train_args_file", type=str, default='train_args/train_ofa.json', help="")
    
    parser.add_argument('--freeze_encoder', default=True, type = bool)
    parser.add_argument('--max_seq_length', default=128, type=int,)
    parser.add_argument('--max_object_length', default=30, type=int,)
    parser.add_argument('--max_tgt_length', default=30, type=int,)
    parser.add_argument('--patch_image_size', default=224, type=int,)

    args_base = parser.parse_args()
    parser = HfArgumentParser((VQAArguments, TrainingArguments))
    args, training_args = parser.parse_json_file(json_file=args_base.train_args_file)


    for atr in [a for a in dir(args_base) if not a.startswith('__')]:

        try:
            setattr(training_args ,atr ,getattr(args_base , atr))
        except:
            continue

        try:
            setattr(args ,atr ,getattr(args_base , atr))
        except:
            continue


    main(args_base , args , training_args)









# accelerate configuration saved at /nethome/sanisetty3/.cache/huggingface/accelerate/default_config.yaml   