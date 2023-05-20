import json
import os
import csv
from tool import *
path1='../data/base'
path2='../data/history'
max_length_sent=52
max_length_subsent=52

book_i=0
book_index,books_sent,books_subsent,book_i,bookname_index=read_data_unit(path1,max_length_sent-2,book_i)

text_clean_sent1,id1=build_data(books_sent,max_length_sent,bookname_index)
#text_clean_sent1,id1=build_data(books_subsent,max_length_sent,bookname_index)

book_index,books_sent,books_subsent,book_i,bookname_index=read_data_unit(path2,max_length_sent-2,book_i)

text_clean_sent2,id2=build_data(books_sent,max_length_sent,bookname_index)
#text_clean_sent2,id2=build_data(books_subsent,max_length_sent,bookname_index)

id=id1+id2
text=text_clean_sent1+text_clean_sent2
#每本书有效句子数量

print(len(id))
num_count={}
for j in range(len(id)):
    sent=id[j]
    if sent[0] not in num_count:
        num_count[sent[0]]={}
        num_count[sent[0]]['sum']=[]
    if sent[1] not in  num_count[sent[0]]:
        num_count[sent[0]][sent[1]]=[]

    num_count[sent[0]]['sum'].append(text[j])
    num_count[sent[0]][sent[1]].append(text[j])

num_sum=0
for i in range(book_i):
    if i not in num_count:
        num_count[i]={}
        num_count[i]['sum']=0
    else:
        for k in num_count[i]:
            num_count[i][k]=len(list(set(num_count[i][k])))
            #num_count[i][k] = len(num_count[i][k])
            num_sum+=num_count[i][k]
print(num_sum)

with open('sent_num.json', "w") as f:
    json.dump(num_count, f)

