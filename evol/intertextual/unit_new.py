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

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#log_path='./log/1loss-10e'
#restore='105630_updates_checkpoint.pt'
#log_path='./log/0.2d0.1a'
#restore='111420_updates_checkpoint.pt'
log_path='./log/sub0.1d0.5a'
restore='262630_updates_checkpoint.pt'

path='../data/base'
#path='../data/test'
path_index='./base/index_book.json'
#path_index_id_sent='./base/index_v_id_sent.json'
#path_index_id_subsent='./base/index_v_id_subsent.json'

write_path_sent='./base/result_text_sent.csv'
write_path_id_sent='./base/result_id_sent.json'
write_path_text_sent='./base/result_text_sent.json'

write_path_subsent='./base/result_text_subsent.csv'
write_path_id_subsent='./base/result_id_subsent.json'
write_path_text_subsent='./base/result_text_subsent.json'

write_path_sent_unit='./base/result_text_sent_unit.csv'
write_path_id_sent_unit='./base/result_id_sent_unit.json'
write_path_text_sent_unit='./base/result_text_sent_unit.json'
f_thr = "./base/dist-thr.txt"

use_cuda=True
max_length_sent=52
max_length_subsent=52
ratio_sent=0.02
ratio_subsent=0.02
batch_size=128
device_index=2
device_bert= torch.device("cuda:2")
book_i=0
book_index,books_sent,books_subsent,book_i,bookname_index=read_data_unit(path,max_length_sent-2,book_i)

with open(path_index, "w") as f:
    json.dump(book_index, f)
print('sent_all')
print(len(books_sent))
print(len(books_subsent))

checkpoints = torch.load(os.path.join(log_path,restore))
model = model_CL(use_cuda)
model.load_state_dict(checkpoints['model'])
model = model.to(device_bert)
model.eval()


text_clean_sent,id2book_sent=build_data(books_sent,max_length_sent,bookname_index)

#text_clean_sent_thr=list(set(text_clean_sent))
vectors_sent=get_vector(text_clean_sent,model,batch_size,max_length_sent,use_cuda,device_bert)

vectors_sent_thr=vector_uni(vectors_sent)
vectors_sent_thr = np.array(vectors_sent_thr ).astype('float32')
#torch.cuda.empty_cache()
result_sent=search(vectors_sent_thr,vectors_sent_thr,device_index,d=768,nlist = 1000,k = 100,batch_size_index=1000,use_cuda=True)
#torch.cuda.empty_cache()
thr_sent=get_dist_thr(result_sent,id2book_sent,ratio_sent)


vectors_sent = np.array(vectors_sent ).astype('float32')
#torch.cuda.empty_cache()
result_sent=search(vectors_sent,vectors_sent,device_index,d=768,nlist = 1000,k = 200,batch_size_index=1000,use_cuda=True)
#torch.cuda.empty_cache()

result_id_sent=filter_thr(result_sent,thr_sent,id2book_sent,id2book_sent)
result_id_sent=add_reverse(result_id_sent)
result_id_sent.sort()
result_text_sent=id2text(result_id_sent,book_index,book_index,is_sent=True)
write_result(write_path_sent, write_path_id_sent, write_path_text_sent,result_text_sent, result_id_sent)

with open(f_thr,"a") as file:
    file.write(str(thr_sent)+'/'+log_path +"\n")


text_clean_subsent,id2book_subsent=build_data(books_subsent,max_length_subsent,bookname_index)
vectors_subsent=get_vector(text_clean_subsent,model,batch_size,max_length_subsent,use_cuda,device_bert)

vectors_subsent_thr=vector_uni(vectors_subsent)
vectors_subsent_thr = np.array(vectors_subsent_thr ).astype('float32')
torch.cuda.empty_cache()
result_subsent=search(vectors_subsent_thr,vectors_subsent_thr,device_index,d=768,nlist = 1000,k = 100,batch_size_index=1000,use_cuda=True)
torch.cuda.empty_cache()
thr_subsent=get_dist_thr(result_subsent,id2book_subsent,ratio_subsent)


vectors_subsent = np.array(vectors_subsent ).astype('float32')
torch.cuda.empty_cache()
result_subsent=search(vectors_subsent,vectors_subsent,device_index,d=768,nlist = 1000,k = 200,batch_size_index=1000,use_cuda=True)
torch.cuda.empty_cache()

result_id_subsent=filter_thr(result_subsent,thr_subsent,id2book_subsent,id2book_subsent)
result_id_subsent=add_reverse(result_id_subsent)
result_id_subsent.sort()
result_text_subsent=id2text(result_id_subsent,book_index,book_index,is_sent=False)
write_result(write_path_subsent, write_path_id_subsent, write_path_text_subsent,result_text_subsent, result_id_subsent)

with open(f_thr,"a") as file:
    file.write(str(thr_sent)+'/'+str(thr_subsent)+'/'+log_path +"\n")

print('start combine')
result_id_sent_unit=unit_subsent(result_id_sent,result_id_subsent)
result_text_sent_unit=id2text(result_id_sent_unit,book_index,book_index,is_sent=True)
write_result(write_path_sent_unit, write_path_id_sent_unit, write_path_text_sent_unit,result_text_sent_unit, result_id_sent_unit)
