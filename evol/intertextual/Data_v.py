import torch
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("./data")

class Dataset_v(torch.utils.data.Dataset):
    def __init__(self,data,max_length):
        self.data_all=data
        self.number = len(self.data_all)
        self.max_length=max_length

    def __len__(self):
        return  self.number

    def __getitem__(self, index):
        text= self.data_all[index]
        token = tokenizer(text, padding=True, pad_to_multiple_of=self.max_length, return_tensors='pt')
        ids = token['input_ids'][0]
        mask_pad = token['attention_mask'][0]
        example = Example_v(ids,mask_pad)

        return example

class Collate_v:
    def __init__(self,ept):
        self.ept=ept

    def __call__(self, example_list):
        return Batch_v(example_list)

class Batch_v:
    def __init__(self, example_list):

        self.ids=[e.ids for e in example_list]
        self.mask_pad=[e.mask_pad for e in example_list]

class Example_v:
    def __init__(self,ids,mask_pad):
        self.ids=ids
        self.mask_pad=mask_pad


class DataLoader_v:

    def __init__(self, data,batch_size,max_length):
        self.batch_size=batch_size
        self.dataset = Dataset_v(data,max_length)

    def __call__(self):

        dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, pin_memory=True,
                                                     collate_fn=Collate_v(1), shuffle=False)

        return dataloader

def get_dataloader_v(data,batch_size,max_length):
        dataloader= DataLoader_v(data,batch_size,max_length)
        return dataloader()