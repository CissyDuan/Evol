import torch
import os
import json
from tool import *
import numpy as np
from tqdm import tqdm
from Data_v import *
import os
import faiss
import csv

def search(vectors,vectors_index,device,d=768,nlist = 1000,k = 100,batch_size_index=1000,use_cuda=True):
    length=len(vectors)

    print(vectors.shape)
    print(vectors_index.shape)

    if use_cuda:
        res = faiss.StandardGpuResources()
        #co = faiss.GpuClonerOptions()
        #co.useFloat16 = False
        quantizer = faiss.IndexFlatL2(d)  # the other index
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        index = faiss.index_cpu_to_gpu(res, device, index)#, co)

    else:
        quantizer = faiss.IndexFlatL2(d)  # the other index
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    # here we specify METRIC_L2, by default it performs inner-product search
    print('start train')
    assert not index.is_trained
    index.train(vectors_index)
    assert index.is_trained
    print('train end')
    index.add(vectors_index)                  # add may be a bit slower as well

    index.nprobe = 100              # default nprobe is 1, try a few more

    idx=0
    print('start index')

    print_count=0
    result_vector=[]
    #with open(path_write, 'w', encoding='utf-8', newline='') as csvfile:
        #csvwriter = csv.writer(csvfile)
    while idx+batch_size_index <length:
        query=vectors[idx:idx+batch_size_index,:]
        D, I = index.search(query, k)
        query_id=list(range(idx,idx+batch_size_index))
        for q,i,d in zip(query_id,I,D):
            i=i.tolist()
            d=d.tolist()
            row=[q,i,d]
            #csvwriter.writerow(row)
            result_vector.append(row)
        idx+=batch_size_index
        print_count+=1
        if print_count%50==0:
            print(print_count*batch_size_index/length*100)
    if idx!=length-1:
        query = vectors[idx:, :]
        D, I = index.search(query, k)
        query_id = list(range(idx,length))
        for q, i, d in zip(query_id, I, D):
            i = i.tolist()
            d = d.tolist()
            row = [q, i, d]
            #csvwriter.writerow(row)
            result_vector.append(row)
    return result_vector

    #with open(path_write_j, "w") as f:
     #   json.dump(result_vector, f)

def get_vector(text_clean,model,batch_size,max_length,use_cuda,device):

    dataloader=get_dataloader_v(text_clean,batch_size,max_length)
    vectors=bert_vector(dataloader,model,use_cuda,device)
    return vectors

def get_dist_thr(result,id2book,ratio):
    dist_sum = []

    for row in tqdm(result):
        s1 = row[0]
        sent1 = id2book[s1]
        sents2 = row[1]
        dists = row[2]
        for s2, d in zip(sents2, dists):
            if s2 != s1:
                sent2 = id2book[s2]
                book_idx1 = sent1[0]
                book_idx2 = sent2[0]

                if book_idx1 != book_idx2:
                    dist_sum.append(d)
    dist_sum.sort()
    all_dist = len(dist_sum)
    print(all_dist)
    dist_thr = dist_sum[int(all_dist * ratio)]
    print(dist_thr)
    return dist_thr


def filter_thr(result,dist_thr,id2book1,id2book2):
    result_id=[]
    for row in tqdm(result):
        s1=row[0]
        sent1=id2book1[s1]
        sents2=row[1]
        dists=row[2]
        for s2,d in zip(sents2,dists):
            if s2!=s1 and d<=dist_thr:
                sent2=id2book2[s2]
                book_idx1 = sent1[0]
                book_idx2 = sent2[0]
                if book_idx1 != book_idx2:
                    result_id.append([sent1,sent2,d])
    result_id.sort()
    return result_id

def add_reverse(result_id):
    reverse_id=[[pair[1],pair[0],pair[2]] for pair in result_id]
    result_id+=reverse_id
    result_id.sort()
    result_id_uni=[]
    result_id_uni.append(result_id[0])
    for i in range(1,len(result_id)):
        if result_id[i]!=result_id[i-1]:
            result_id_uni.append(result_id[i])
    return result_id_uni

def vector_uni(vector):
    vector_thr=sorted(vector)
    vector_thr_uni=[]
    for i in range(1,len(vector_thr)):
        if vector_thr[i]!=vector_thr[i-1]:
            vector_thr_uni.append(vector_thr[i])
    return vector_thr_uni
