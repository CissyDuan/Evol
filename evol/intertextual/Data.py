import torch
import json
import re
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("ethanyt/guwenbert-base")
import random


class Dataset(torch.utils.data.Dataset):
    def __init__(self,data,max_length):
        self.data_all=data
        self.number = len(self.data_all)
        self.max_length=max_length

    def __len__(self):
        return  self.number

    def __getitem__(self, index):
        text= self.data_all[index]

        text_mini=re.split(r'[,，]', text)
        length=len(text_mini)
        if length>2:
            sample_num=random.randint(2,length-1)
            text_sample=random.sample(text_mini,sample_num)
            text_sample='，'.join(text_sample)
            if len(text_sample)<=3:
                text_sample=text
        else:
            text_sample=text

        token = tokenizer(text, padding=True, pad_to_multiple_of=self.max_length, return_tensors='pt')
        ids = token['input_ids'][0]
        mask_pad = token['attention_mask'][0]

        token_sample = tokenizer(text_sample, padding=True, pad_to_multiple_of=self.max_length, return_tensors='pt')
        ids_sample = token_sample['input_ids'][0]
        mask_pad_sample = token_sample['attention_mask'][0]

        example = Example(ids,mask_pad,ids_sample,mask_pad_sample)

        return example

class Collate:
    def __init__(self,ept):
        self.ept=ept

    def __call__(self, example_list):
        return Batch(example_list)

class Batch:
    def __init__(self, example_list):

        self.ids=[e.ids for e in example_list]
        self.mask_pad=[e.mask_pad for e in example_list]

        self.ids_sample = [e.ids_sample for e in example_list]
        self.mask_pad_sample = [e.mask_pad_sample for e in example_list]


class Example:
    def __init__(self,ids,mask_pad,ids_sample,mask_pad_sample):
        self.ids=ids
        self.mask_pad=mask_pad

        self.ids_sample=ids_sample
        self.mask_pad_sample=mask_pad_sample



class DataLoader:

    def __init__(self, data_path,batch_size,max_length):
        data=json.load(open(data_path))
        self.batch_size=batch_size
        self.dataset = Dataset(data,max_length)

    def __call__(self):
        dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, pin_memory=True,
                                                     collate_fn=Collate(1), shuffle=True)

        return dataloader

