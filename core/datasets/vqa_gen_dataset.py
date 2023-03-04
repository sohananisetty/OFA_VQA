# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from io import BytesIO

import logging
import warnings

import numpy as np
import torch
import base64
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import os
import json
from PIL import Image, ImageFile
import re
from io import BytesIO


ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class VQACollator():
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.bos = torch.LongTensor([self.tokenizer.bos_token_id])
        self.eos = torch.LongTensor([self.tokenizer.eos_token_id])
        self.pad = torch.LongTensor([self.tokenizer.pad_token_id])

    def __call__(self, samples):

        src = [samples["source"] for s in samples]
        src_tokens = self.tokenizer( src, return_tensors="pt", max_length=self.max_seq_length, truncation=True, padding=True).input_ids
        src_lengths = torch.LongTensor([s.ne(self.pad).long().sum() for s in src])
        
        patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
        patch_masks = torch.cat([sample['patch_mask'] for sample in samples])

        tgt = [samples["target"] for s in samples]
        tgt_tokens = self.tokenizer( tgt, return_tensors="pt", max_length=self.max_seq_length, truncation=True, padding=True)
        attention_mask = tgt_tokens.attention_mask
        tgt_tokens = tgt_tokens.input_ids
        tgt_lengths = torch.LongTensor([s.ne(self.pad).long().sum() for s in tgt])
        # ntokens = tgt_lengths.sum().item()

        if samples[0]["prompt_type"] == 'none':
            prev_output_tokens = tgt_tokens[:,:-1]
            target_item = tgt_tokens[:,1:]
            decoder_input_ids = torch.repeat_interleave(self.bos , target_item.shape[0]).reshape(1,-1)



        # prefix_tokens = None
        # if samples[0].get("decoder_prompt", None) is not None:
        #     prefix_tokens = decoder_input_ids[:, 1:]

        batch = {
            "input_ids": src_tokens,
            "src_lengths": src_lengths,
            "patch_images": patch_images,
            "patch_masks": patch_masks,
            # "prev_output_tokens": prev_output_tokens,
            "decoder_input_ids": decoder_input_ids,
            "target": tgt_tokens,
            # "prefix_tokens": prefix_tokens,
            'return_loss': False,
            "attention_mask" : attention_mask,

        }

   
        return batch    
    


class VqaGenDataset(Dataset):
    def __init__(
        self,
        split,
        folder,
        max_src_length=128,
        max_object_length=30,
        max_tgt_length=30,
        patch_image_size=224,
        add_object=False,
        imagenet_default_mean_and_std=False,
        prompt_type="none"
    ):
        self.split = split
        self.dataset = folder
        self.max_src_length = max_src_length
        self.max_object_length = max_object_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size

        self.add_object = add_object
        self.prompt_type = prompt_type

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((patch_image_size, patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def pre_question(self, question, max_ques_words=None):
        question = question.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ')

        question = re.sub(
            r"\s{2,}",
            ' ',
            question,
        )
        question = question.rstrip('\n')
        question = question.strip(' ')

        # truncate question
        question_words = question.split(' ')
        if max_ques_words is not None and len(question_words) > max_ques_words:
            question = ' '.join(question_words[:max_ques_words])

        return question

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]

        image, question, ref, predict_objects = item

        image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
        patch_image = self.patch_resize_transform(image)
        patch_mask = torch.tensor([True])

        question = self.pre_question(question, self.max_src_length)
        question = question + '?' if not question.endswith('?') else question
        src_item = (' {}'.format(question))

        ref_dict = {item.split('|!+')[1]: float(item.split('|!+')[0]) for item in ref.split('&&')}
        answer = max(ref_dict, key=ref_dict.get)
        conf = torch.tensor([ref_dict[answer]])
        tgt_item = (" {}".format(answer))

        # if self.add_object and predict_objects is not None:
        #     predict_object_seq = ' '.join(predict_objects.strip().split('&&')[:self.max_object_length])
        #     predict_object_item = " object: {}".format(predict_object_seq)
        #     src_item = src_item + predict_object_item

        example = {
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": tgt_item,
            "prompt_type": self.prompt_type,
        }

        return example





class VqaDataset(Dataset):
    def __init__(self, ann_file, vqa_root, patch_image_size = 224,imagenet_default_mean_and_std = False, split="train", max_ques_words=128,max_obj_length = 30, answer_list='', read_local_data=True, add_object=False,prompt_type="none"):
        self.split = split        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))


        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((patch_image_size, patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        self.vqa_root = vqa_root
        self.add_object = add_object
        self.prompt_type = prompt_type
        self.max_ques_words = max_ques_words
        self.max_obj_length = max_obj_length
        self.read_local_data = read_local_data

        
        if split=='test':
            self.max_ques_words = 128 # do not limit question length during test
            self.answer_list = json.load(open(answer_list,'r'))    

        

    def pre_question(self, question, max_ques_words=None):
        question = question.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ')

        question = re.sub(
            r"\s{2,}",
            ' ',
            question,
        )
        question = question.rstrip('\n')
        question = question.strip(' ')

        # truncate question
        question_words = question.split(' ')
        if max_ques_words is not None and len(question_words) > max_ques_words:
            question = ' '.join(question_words[:max_ques_words])

        return question           
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        if ann['dataset']=='vqa':
            image_path = os.path.join(self.vqa_root,ann['image'])    
       
            
        image = Image.open(image_path.replace("_img" , "")).convert('RGB')
        image = self.patch_resize_transform(image)
        patch_mask = torch.tensor([True])


        question = ann['question']
        question = self.pre_question(question, self.max_src_length)
        question = question + '?' if not question.endswith('?') else question
        question = (' {}'.format(question))

     
        if self.split == 'test':
            question_id = ann['question_id']            
            return image, question, question_id


        elif self.split=='train':                       
            
            
            if ann['dataset']=='vqa':
                
                answer_weight = {}
                for answer in ann['answer']:
                    if answer in answer_weight.keys():
                        answer_weight[answer] += 1/len(ann['answer'])
                    else:
                        answer_weight[answer] = 1/len(ann['answer'])

                answers = list(answer_weight.keys())
                weights = list(answer_weight.values())

                answer = max(answer_weight, key=answer_weight.get)
                conf = torch.tensor([answer_weight[answer]])


           

            # answers = [answer+self.eos for answer in answers]

            example = {
            "source": question,
            "patch_image": image,
            "patch_mask": patch_mask,
            "target": answer,
            "conf":conf,
            "prompt_type": self.prompt_type,
            "ref_dict":answer_weight,
            }
                
            return example
        #image, question, answers, weights
