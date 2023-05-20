
import os
import json
from tool import *
import numpy as np
from tqdm import tqdm

path='../data/base'
path_data1='./data/base.json'

max_length=52

book_i=0
book_index,books_sent,books_subsent,book_i,bookname_index=read_data_unit(path,max_length-2,book_i)

print('sent_all')
print(len(books_sent))

text_clean_sent,id2book_sent=build_data(books_sent,max_length,bookname_index)

with open(path_data1, "w") as f:
    json.dump(text_clean_sent, f)


'''
path_data2='./data/base_subsent.json'
text_clean_subsent,id2book_subsent=build_data(books_subsent,max_length,bookname_index)
print('sent_chosen')
print(len(text_clean_subsent))
with open(path_data2, "w") as f:
    json.dump(text_clean_subsent, f)
'''