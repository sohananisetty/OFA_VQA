
import argparse
import os
import numpy as np
import random
import time
import datetime
from pathlib import Path
import torch
from glob import glob
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

    tokenizer = OFATokenizer.from_pretrained(args.pretrained)

    if len(glob(("{training_args.output_dir}/**/*.pt") , recursive  =True)) == 0 and args_base.resume == False:
        model = OFAModelForVQA.from_pretrained(args.pretrained, use_cache=False)
        

        print("using pretrained checkpoint")

    # else:
    #     model = OFAModelForVQA.from_pretrained(args.output, use_cache=False)


    trainer = VQATrainer(
        vqa_model = model,
        tokenizer = tokenizer,
        args = args,
        training_args = training_args,
        wandb_every = args_base.wandb_every,
        data_folder=args_base.data_folder

    ).cuda()


    trainer.train(args_base.resume)
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', default='/srv/scratch/sanisetty3/DLM/AliceMind/mPLUG/data/json/vqa_ocr_object/', help="folder with train and test data")
    parser.add_argument('--pretrained', default='/srv/scratch/sanisetty3/DLM/OFA-tiny')
    parser.add_argument('--resume', default=True, type = bool)
    parser.add_argument('--output_dir', default="/srv/scratch/sanisetty3/DLM/OFA_VQA/checkpoints")
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--fp16', default=True, type=bool)

    parser.add_argument('--per_device_train_batch_size', default=24, type=int,)
    parser.add_argument('--per_device_eval_batch_size', default=12, type=int,)
    parser.add_argument('--gradient_accumulation_steps', default=8, type=int,)
    parser.add_argument('--label_smoothing', default=0.1, type=float,)


    parser.add_argument("--num_train_epochs",  default=100,type=int)
    parser.add_argument("--save_steps",  default=500,type=int)
    parser.add_argument("--logging_steps",  default=10,type=int)
    parser.add_argument("--wandb_every",  default=50,type=int)
    parser.add_argument("--train_args_file", type=str, default='/srv/scratch/sanisetty3/DLM/OFA_VQA/ctl/train_ofa.json', help="")
    
    parser.add_argument('--freeze_encoder', default=True, type = bool)
    parser.add_argument('--max_seq_length', default=80, type=int,)
    parser.add_argument('--max_object_length', default=30, type=int,)
    parser.add_argument('--max_tgt_length', default=30, type=int,)
    parser.add_argument('--patch_image_size', default=480, type=int,)

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

    # print("args" , args)
    # print("args_base" , args_base)



    main(args_base , args , training_args)









# accelerate configuration saved at /nethome/sanisetty3/.cache/huggingface/accelerate/default_config.yaml   