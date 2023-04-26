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
import h5py


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

        src = [sample["source"] for sample in samples]
        src_tokens = self.tokenizer( src, return_tensors="pt", max_length=self.max_seq_length, truncation=True, padding=True).input_ids
        src_lengths = torch.LongTensor([s.ne(self.pad).long().sum() for s in src_tokens])
        
        patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
        patch_masks = torch.cat([sample['patch_mask'] for sample in samples])

        tgt = [sample["target"] for sample in samples]
        tgt_tokenised = self.tokenizer( tgt, return_tensors="pt", max_length=self.max_seq_length, truncation=True, padding=True)
        attention_mask = tgt_tokenised.attention_mask
        tgt_tokens = tgt_tokenised.input_ids
        tgt_lengths = torch.LongTensor([s.ne(self.pad).long().sum() for s in tgt_tokens])
        # ntokens = tgt_lengths.sum().item()
        if samples[0]["prompt_type"] == 'none':
            target_item = tgt_tokens[:,1:].clone()
            prev_output_tokens = tgt_tokens[:,:-1].clone()
            prev_output_tokens[prev_output_tokens==self.eos] = self.pad
            attention_mask = attention_mask[:,1:]
            decoder_prompt_ids = torch.repeat_interleave(self.bos , target_item.shape[0]).reshape(-1,1)



        # prefix_tokens = None
        # if samples[0].get("decoder_prompt", None) is not None:
        #     prefix_tokens = decoder_prompt_ids[:, 1:]

        ref_dict = None
        # print(samples[0].get("ref_dict", "no ref dict"))
        if samples[0].get("ref_dict", None) is not None:
            ref_dict = np.array([s['ref_dict'] for s in samples])

        conf = None
        if samples[0].get("conf", None) is not None:
            conf = torch.cat([s['conf'] for s in samples], dim=0)


        batch = {
            "input_ids": src_tokens,
            "patch_images": patch_images,
            "patch_masks": patch_masks,
            "decoder_input_ids": prev_output_tokens,
            "target": target_item,
            "attention_mask" : attention_mask,
            "conf": conf,
            "ref_dict": ref_dict,

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
            "ref_dict":ref_dict,
            "conf": conf,

        }

        return example


class VqaDataset(Dataset):
    def __init__(self, ann_file, vqa_root, patch_image_size = 224,imagenet_default_mean_and_std = False, split="train", max_ques_words=128,max_obj_length = 30, answer_list='', read_local_data=True, add_object=False,prompt_type="none"):
        self.split = split        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))

        print()


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

        
        # if split=='test':
        #     self.max_ques_words = 128 # do not limit question length during test
            # self.answer_list = json.load(open(answer_list,'r'))    

        

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

    def pre_answer(self, answer):
        answer = answer.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ')

        answer = re.sub(
            r"\s{2,}",
            ' ',
            answer,
        )
        answer = answer.rstrip('\n')
        answer = answer.strip(' ')
        return answer           
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        if ann['dataset']=='vqa':
            image_path = os.path.join(self.vqa_root,ann['image'])
            image = Image.open(image_path.replace("_img" , "")).convert('RGB')
 

        if  ann['dataset'] == 'stack':   
            image_path = os.path.join(self.vqa_root,ann['image_filename'])
            image = Image.open(image_path).convert('RGB')

       
            
        image = self.patch_resize_transform(image)
        patch_mask = torch.tensor([True])


        question = ann['question']
        question = self.pre_question(question, self.max_ques_words)
        question = question + '?' if not question.endswith('?') else question
        question = (' {}'.format(question))

     
        # if self.split == 'test':
        #     question_id = ann['question_id']            
        #     return image, question, question_id


        if self.split=='train':                       
            
            
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

            if ann['dataset']=='stack':
            
                answer = self.pre_answer(str(ann['answer']))
                conf = torch.tensor([1])


           

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


class VqaStackDataset(Dataset):
    def __init__(self, ann_file, vqa_root, patch_image_size = 224,imagenet_default_mean_and_std = False, split="train", max_ques_words=128,max_obj_length = 30, answer_list='', read_local_data=True, add_object=False,prompt_type="none"):
        self.split = split        
        with open(ann_file) as f:
            self.ann = json.load(f)

        self.ann = self.ann["questions"]

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((patch_image_size, patch_image_size), interpolation=Image.BICUBIC),
            transforms.ColorJitter(brightness=.3),
            transforms.GaussianBlur(kernel_size=(5, 5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
           
        ])
        self.vqa_root = vqa_root
        self.add_object = add_object
        self.prompt_type = prompt_type
        self.max_ques_words = max_ques_words
        self.max_obj_length = max_obj_length
        self.read_local_data = read_local_data

        
        # if split=='test':
        #     self.max_ques_words = 128 # do not limit question length during test
            # self.answer_list = json.load(open(answer_list,'r'))    

        

    def pre_question(self, question, max_ques_words=None):
        question = question.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ')

        question = re.sub(
            r"\s{2,}",
            ' ',
            question,
        )
        question = question.rstrip('\n')
        question = question.strip(' ')

        question = question.replace("most" , "furthest")

        

        # truncate question
        question_words = question.split(' ')
        if max_ques_words is not None and len(question_words) > max_ques_words:
            question = ' '.join(question_words[:max_ques_words])

        return question           
    
    def pre_answer(self, answer):
        answer = answer.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ')

        answer = re.sub(
            r"\s{2,}",
            ' ',
            answer,
        )
        answer = answer.rstrip('\n')
        answer = answer.strip(' ')
        return answer           
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
      
        image_path = os.path.join(self.vqa_root,ann['image_filename'])

        # try:
        image = Image.open(image_path).convert('RGB')
        # print(f"loading image {image_path} failed")
        # except:
            # new_index = np.random.choice(len(self.ann))
            # ann = self.ann[new_index]
            # image_path = os.path.join(self.vqa_root,ann['image_filename'])
            # image = Image.open(image_path).convert('RGB')



    
            
        image = self.patch_resize_transform(image)
        patch_mask = torch.tensor([True])


        question = ann['question']
        question = self.pre_question(question, self.max_ques_words)
        question = question + '?' if not question.endswith('?') else question
        question = (' {}'.format(question))

     
        # if self.split == 'test':
        #     question_id = ann['question_id']            
        #     return image, question, question_id


        # if self.split=='train':                       
        answer = self.pre_answer(str(ann['answer']))
        conf = torch.tensor([1])

        if "false" in answer.lower():
            
            answer = answer.replace("False" , "no")
            # question.replace("false" , "no")

        if "true" in answer.lower():
            answer = answer.replace("True" , "yes")
            # question.replace("true" , "yes")

        example = {
        "source": question,
        "patch_image": image,
        "patch_mask": patch_mask,
        "target": answer,
        "conf":conf,
        "prompt_type": self.prompt_type,
        }
            
        return example


class CLEVRVQADataset(Dataset):
    def __init__(self, data_dir = '/srv/scratch/sanisetty3/DLM/sornet/data/clevr_cogent', 
                 split = "trainA", patch_image_size = 224, 
                 imagenet_default_mean_and_std = False,max_nobj = 10 , 
                 max_ques_words = 128, prompt_type="none"):

        self.scene_file_path = os.path.join(data_dir , split+".h5")
        self.question_file_path = os.path.join(data_dir ,f"questions/CLEVR_{split}_questions.json")
        
        with h5py.File(self.scene_file_path) as scene_h5:
            self.scene_keys = list(scene_h5.keys())
        self.scene_h5 = h5py.File(self.scene_file_path)
        
            
        with open(self.question_file_path ) as f:
            self.quesions = json.load(f)["questions"]
            
        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((patch_image_size, patch_image_size), interpolation=Image.BICUBIC),
            transforms.ColorJitter(brightness=.3),
            transforms.GaussianBlur(kernel_size=(5, 5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
        self.max_nobj = max_nobj
        self.max_ques_words=max_ques_words
        self.prompt_type = prompt_type
        

    def __len__(self):
        return len(self.quesions)
    
    
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

    def pre_answer(self, answer):
        answer = answer.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ')

        answer = re.sub(
            r"\s{2,}",
            ' ',
            answer,
        )
        answer = answer.rstrip('\n')
        answer = answer.strip(' ')
        return answer                  
  

    def __getitem__(self, idx):
        
        ##questions
        blacklist_indx = [3458, 33105, 47245, 62568,16164, 29041, 30429, 30512, 45493, 47196,45934]

        blacklist_imgs = ['000387','003458','005388','005634','006415','006580','008480','009088','012042','013284','013637','016054','022761','024193','027561','029041','030512','033105','039147','042811','043362','043664','047196','047245','049774','053507','054034','055662','056542','058618','058852','060724','061020','062185','062568','063503']
        #if idx in blacklist_indx:
                            
        question = self.quesions[idx]
        
        question_text = question["question"]
        answer_text = self.pre_answer(str(question["answer"]))
        image_indx = question["image_filename"].split("_")[-1].split(".")[0]
        
        if image_indx in blacklist_imgs:
            idx = np.random.randint(0,len(self.scene_keys))
            question = self.quesions[idx]
            question_text = question["question"]
            answer_text = str(question["answer"])
            image_indx = question["image_filename"].split("_")[-1].split(".")[0]
            
            

        question_text = self.pre_question(question_text, self.max_ques_words)
        question_text = question_text + '?' if not question_text.endswith('?') else question_text
        question_text = (' {}'.format(question_text))
        
        scene = self.scene_h5[image_indx]
        image = (Image.open(BytesIO(scene['image'][()])).convert('RGB'))
        objects = scene['objects'][()].decode().split(',')
        
        image = self.patch_resize_transform(image)
        patch_mask = torch.tensor([True])
        
        
        example = {
            "source": question_text,
            "patch_image": image,
            "patch_mask": patch_mask,
            "target": answer_text,
            "conf":torch.tensor([1]),
            "prompt_type": self.prompt_type,
            }

        return example


def build_predicates(objects, unary, binary):
    pred_names = [pred % obj for pred in unary for obj in objects]
    obj1 = [obj for _ in range(len(objects) - 1) for obj in objects]
    obj2 = [obj for i in range(1, len(objects)) for obj in np.roll(objects, -i)]
    pred_names += [pred % (o1, o2) for pred in binary for o1, o2 in zip(obj1, obj2)]
    return pred_names

class LeonardoVQADataset(Dataset):
    def __init__(
        self, data_dir = '/srv/scratch/sanisetty3/DLM/sornet/data/leonardo/',
        split = 'valid',n_objects = 4,
         view=3, randview=True, max_ques_words = 128,prompt_type="none",
        patch_image_size = 224, imagenet_default_mean_and_std = False,
    ):
        
        unary_pred = [
            'on_surface(%s, left)', 'on_surface(%s, right)', 'on_surface(%s, far)',
            'on_surface(%s, center)', 'has_obj(robot, %s)', 'top_is_clear(%s)',
        ]
        binary_pred = ['stacked(%s, %s)']

        objects = [f'object{i:02d}' for i in range(n_objects)]
        predicates = build_predicates(objects, unary_pred, binary_pred)

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        with open(f'{data_dir}/{split}_nframes.json') as f:
            n_frames = json.load(f)
            self.sequences = list(n_frames.keys())
            n_frames = list(n_frames.values())
            self.cum_n_frames = np.cumsum(n_frames)
        with h5py.File(f'{data_dir}/{split}.h5') as data:
            all_predicates = data['predicates'][()].decode().split('|')
            pred_ids = {pred: i for i, pred in enumerate(all_predicates)}
            self.pred_ids = [pred_ids[pred] for pred in predicates]

        self.data_dir = data_dir
        self.split = split
        self.h5 = h5py.File(f'{self.data_dir}/{self.split}.h5', 'r')
        
        self.all_predicates = all_predicates
        self.view = view
        self.randview = randview
        self.patch_image_size = patch_image_size
        self.max_ques_words = max_ques_words
        self.prompt_type = prompt_type
        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((patch_image_size, patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __len__(self):
        return self.cum_n_frames[-1]

    def load_h5(self, idx):
        if self.h5 is None:
            self.h5 = h5py.File(f'{self.data_dir}/{self.split}.h5', 'r')
       
        # Get H5 file index and frame index
        file_idx = np.argmax(idx < self.cum_n_frames)
        data = self.h5[self.sequences[file_idx]]
        frame_idx = idx
        if file_idx > 0:
            frame_idx = idx - self.cum_n_frames[file_idx - 1]
        return data, frame_idx
        
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

    def pre_answer(self, answer):
        answer = answer.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ')

        answer = re.sub(
            r"\s{2,}",
            ' ',
            answer,
        )
        answer = answer.rstrip('\n')
        answer = answer.strip(' ')
        return answer                    
  

    def get_rgb(self, data, idx):
        v = torch.randint(self.view, ()).item() if self.randview else self.view
        rgb = Image.open(BytesIO(data[f'rgb{v}'][idx])).convert('RGB')
        return self.patch_resize_transform(rgb)

    
    def process_template(self, template, objects,answer):
        predicate_name = template.split("(")[0]

        if predicate_name == 'on_surface':
            obj1, loc = template.split('(')[1].split(')')[0].split(',')
#             print("on_surface" , obj1 , loc)
            obj1 = objects[int(obj1[-2:])]
            ques = [f'Is {obj1} colored block on the {loc} of the robot?',
                    f'Where is the {obj1} colored block on the table?'
                   ]
            ans = ['yes' if answer==1 else 'no', loc]
            
            qa = list(zip(ques,ans))
            
            return qa[np.random.choice(len(qa))]

        if predicate_name == 'has_obj':
            _, obj1 = template.split('(')[1].split(')')[0].split(',')
#             print("has_obj" , obj1 ,  objects[int(obj1[-2:])])
            obj1 = objects[int(obj1[-2:])]
            ques = [f'Does the robot have the {obj1} colored block in the gripper?', 
                    f'Is the robot picking up {obj1} colored block?',
                    f'Did the robot pick up {obj1} colored block using the gripper?']

            ans = ['yes' if answer==1 else 'no' , 
                   'yes' if answer==1 else 'no' , 
                   'yes' if answer==1 else 'no',
                  ]
            qa = list(zip(ques,ans))
            
            return qa[np.random.choice(len(qa))]

        if predicate_name == "top_is_clear":
            obj1 = template.split('(')[1].split(')')[0].split(',')[0]
#             print("top_is_clear" , obj1 , objects[int(obj1[-2:])])
            obj1 = objects[int(obj1[-2:])]
            ques = [f'Is there anything on the {obj1} colored block?',
                   f'Is the top of {obj1} colored block clear?',
                   ]
            ans = ['no' if answer==1 else 'yes',
                  'yes' if answer==1 else 'no' ]
            
            qa = list(zip(ques,ans))
            
            return qa[np.random.choice(len(qa))]

        if predicate_name == "stacked":
            obj1 , obj2 = template.split('(')[1].split(')')[0].split(',')  
#             print("stacked" , obj1,obj2)
            other_objs = list(set(np.arange(len(objects))) - set([int(obj1[-2:]) , int(obj2[-2:]) ]))
            
            obj1 = objects[int(obj1[-2:])]
            obj2 = objects[int(obj2[-2:])]
            obj3 = objects[other_objs[0]]
            obj4 = objects[other_objs[1]]

            ques = [f'Is the {obj2} block on the {obj1} colored block?',
                    f'Is the {obj3} block on the {obj1} colored block?',
                    f'Is the {obj4} block on the {obj1} colored block?',
                    f'Is the {obj1} block on the {obj2} colored block?',
                    f'Which color block on the {obj1} colored block?'
                   ]

            ans = ['yes' if answer==1 else 'no' ,
                  'no' if answer==1 else 'yes',
                  'no' if answer==1 else 'yes',
                  'no' if answer==1 else 'yes',
                  obj2]
            
            qa = list(zip(ques,ans))
            
            return qa[np.random.choice(len(qa) , p = [0.5,0.1,0.1,0.1,0.2])]
            
        

    
    def __getitem__(self, idx):
        data, frame_idx = self.load_h5(idx)
        
        rand_ques_id = np.random.choice(self.pred_ids)
        
        objects = data['colors'][()].decode().split(',')
        answers = data['logical'][frame_idx][rand_ques_id]
        
        question,answer = self.process_template(self.all_predicates[rand_ques_id] , objects , answers)
            
        question = self.pre_question(question, self.max_ques_words)
        question = question + '?' if not question.endswith('?') else question
        question = (' {}'.format(question))

        # Load RGB from H5 file
        image = self.get_rgb(data, frame_idx)
        patch_mask = torch.tensor([True])

        answer= self.pre_answer(answer)

        
        example = {
            "source": question,
            "patch_image": image,
            "patch_mask": patch_mask,
            "target": answer,
            "conf":torch.tensor([1]),
            "prompt_type": self.prompt_type,
            }

        return example
