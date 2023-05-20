import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from tool import *
#=================================================
def bert_vector(dataloader,model,use_cuda,device):
    vectors=[]
    for batch in tqdm(dataloader):
        memory=model.infer(batch,device)
        memory=memory.tolist()

        vectors+=memory
    return vectors
from transformers import  RobertaForMaskedLM ,AutoModel

class model_CL(nn.Module):
    def __init__(self, use_cuda):
        super(model_CL, self).__init__()
        self.encoder = RobertaForMaskedLM.from_pretrained("ethanyt/guwenbert-base").roberta
        # self.encoder = AutoModel.from_pretrained("ethanyt/guwenbert-base")
        # self.encoder =  AutoModel

        self.use_cuda = use_cuda

    def forward(self, batch):
        ids = batch.ids
        mask = batch.mask_pad
        ids = torch.stack(ids, dim=0)  # [b,s]
        mask = torch.stack(mask, dim=0)  # [b,s]

        batch_size = ids.size(0)
        seq_len = ids.size(1)

        ids_dropout = torch.cat((ids.unsqueeze(1), ids.unsqueeze(1)), dim=1).reshape([batch_size * 2, seq_len])
        mask_dropout = torch.cat((mask.unsqueeze(1), mask.unsqueeze(1)), dim=1).reshape([batch_size * 2, seq_len])

        if self.use_cuda:
            ids_dropout, mask_dropout = ids_dropout.cuda(), mask_dropout.cuda()

        input = {'input_ids': ids_dropout, 'attention_mask': mask_dropout}
        memory = self.encoder(**input)[0]
        memory = (memory * mask_dropout.unsqueeze(-1)).sum(1) / mask_dropout.sum(-1).unsqueeze(-1)



        return memory



    def infer(self, batch,device):
        ids = batch.ids
        mask = batch.mask_pad


        ids = torch.stack(ids, dim=0)  # [b,s]
        mask = torch.stack(mask, dim=0)  # [b,s]

        if self.use_cuda:
            ids, mask = ids.to(device), mask.to(device)

        input = {'input_ids': ids, 'attention_mask': mask}
        memory = self.encoder(**input)[0]
        memory = (memory * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)
        return memory


def loss_CL(batch_emb):
    # loss_func = nn.CrossEntropyLoss

    # 构造标签
    batch_size = batch_emb.size(0)
    y_true = torch.cat([torch.arange(1, batch_size, step=2, dtype=torch.long).unsqueeze(1),
                        torch.arange(0, batch_size, step=2, dtype=torch.long).unsqueeze(1)],
                       dim=1).reshape([batch_size, ])
    # print(y_true)

    cross = torch.eye(batch_size) * 1e12

    y_true = y_true.cuda()
    cross = cross.cuda()

    # 计算score和loss
    norm_emb = F.normalize(batch_emb, dim=1, p=2)
    sim_score = torch.matmul(norm_emb, norm_emb.transpose(0, 1))

    sim_score = sim_score - cross
    # print(sim_score)

    sim_score = sim_score * 20  # 温度系数为 0.05，也就是乘以20
    # print(sim_score)
    # assert 1 == 0
    loss = F.cross_entropy(sim_score, y_true, reduction='sum')
    # loss = loss_func(sim_score, y_true,reduction=mean)

    return loss



