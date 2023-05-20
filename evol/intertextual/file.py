from tqdm import tqdm
import opencc
import os
import re
import json
from tool import *
import csv
#=================================================
def combine_subsent(subsents):
    subsents_new=[]
    unit_str=''
    flag=0
    mark=[',','，']
    for s in subsents:
        if flag==0:
            if len(s)<=3 and s not in mark:
                flag=1
                unit_str+=s

            else:
                subsents_new.append(s)

        else:
            if len(s)>3:
                subsents_new.append(unit_str)
                unit_str=''
                flag=0
                subsents_new.append(s)
            else:
                unit_str+=s

    if unit_str!='':
        subsents_new.append(unit_str)
    return subsents_new


def read_data(file_path,book_index,books_sent,books_subsent,max_length,book_i):
    #print(file_path)
    book_msg={}
    book_pages={}
    f=open(file_path)
    book = json.load(f)
    bookname=list(book.keys())[0]
    book_msg['bookname'] = bookname
    book=book[bookname]
    pages=list(book.keys())
    page_i=0
    for page in pages:
        page_msg={}
        pagename=remove_diff (page)
        page_msg['pagename']=pagename

        para_msg = {}
        para_i=0

        for para in book[page]:
            sent_i = 0
            sent_msg = {}

            sents=re.split(r'([;；。.？?！!：:「」『』\s])\s*',para)

            for sent in sents:
                sent=sent_split(sent,max_length)
                for s in sent:
                    s=remove_diff(s)
                    #s=remove_mark(s)
                    s=remove_multimark(s)


                    sub_s=re.split(r'([,，])', s)
                    sub_s=combine_subsent(sub_s)
                    sub_s=list(filter(lambda x:x!='',sub_s))

                    #if len(sub_s)>2:

                    book_text_id = {}
                    book_text_id['text'] = s
                    book_text_id['bookid'] = book_i
                    book_text_id['index'] = [book_i, page_i, para_i, sent_i]
                    books_sent.append(book_text_id)

                    subsent_msg = {}
                    subsent_i = 0
                    for sub_s_i in sub_s:
                        subsent_msg[subsent_i] = sub_s_i

                        sub_s_i_clean=re.sub('[,，]', '', sub_s_i)

                        book_text_id = {}
                        book_text_id['text'] = sub_s_i_clean
                        book_text_id['bookid'] = book_i
                        book_text_id['index'] = [book_i, page_i, para_i,sent_i,subsent_i]
                        books_subsent.append(book_text_id)

                        subsent_i+=1


                    sent_msg[sent_i]=subsent_msg
                    sent_i+=1

            para_msg[para_i]=sent_msg
            para_i+=1

        page_msg['para']=para_msg
        book_pages[page_i] = page_msg
        page_i+=1

    book_msg['pages'] = book_pages
    book_index[book_i] = book_msg
    book_i += 1

    return book_index,books_sent,books_subsent,book_i

def read_data_unit(path,max_length,book_i):
    book_index = {}
    books_sent = []
    books_subsent = []
    bookname_index={}

    for root, dirs, files in os.walk(path):
        files.sort()
        for file_name in tqdm(files):
            if file_name != '.DS_Store':
                file_path = os.path.join(root, file_name)
                bookname=file_name[:-5]
                bookname_index[book_i]=bookname
                book_index, books_sent, books_subsent, book_i=read_data(file_path,book_index,books_sent,books_subsent,max_length,book_i)
    return book_index,books_sent,books_subsent,book_i,bookname_index



def build_data(books,max_length,bookname_index):
    longer_count = 0
    idx=0
    text_clean=[]
    idmsg=[]

    for text in tqdm(books):
        if text==None:
            print('None')
            continue
        textclean=textToCode(text['text'],bookname_index[text['bookid']])
        if textclean!=None:
            if len(textclean)>max_length-2:
                #print(textclean)
                longer_count+=1
            else:
                text_clean.append(textclean)
                idmsg.append(text['index'])
                idx+=1

                #if len(textclean) > max_sent:
                    #max_sent = len(textclean)

    print('sent_chosen')
    print(idx)

    print('sent_longer')
    print(longer_count)

    # with open(path_index_id, "w") as f:
    # json.dump(idmsg, f)

    return text_clean,idmsg

def read_index_sent(books,index):
    book_idx,page_idx,para_idx,sent_idx=index[0],index[1],index[2],index[3]
    bookname=books[book_idx]['bookname']
    pagename=books[book_idx]['pages'][page_idx]['pagename']
    sents=books[book_idx]['pages'][page_idx]['para'][para_idx][sent_idx]
    sents=list(sents.values())
    sent=''.join(sents)

    return bookname,pagename,sent,book_idx

def read_index_subsent(books,index):
    book_idx,page_idx,para_idx,sent_idx,subsent_idx=index[0],index[1],index[2],index[3],index[4]
    bookname=books[book_idx]['bookname']
    pagename=books[book_idx]['pages'][page_idx]['pagename']
    subsent=books[book_idx]['pages'][page_idx]['para'][para_idx][sent_idx][subsent_idx]

    return bookname,pagename,subsent,book_idx

def unit_subsent(result_id_sent,result_id_subsent):
    result_id_subsent=[[pair[0][:-1],pair[1][:-1],pair[2]] for pair in result_id_subsent]

    result_id_sent += result_id_subsent
    result_id_sent.sort()
    result_id_uni = []
    result_id_uni.append(result_id_sent[0])
    for i in range(1, len(result_id_sent)):
        if result_id_sent[i][:-1] != result_id_sent[i - 1][:-1]:
            result_id_uni.append(result_id_sent[i])
        else:
            result_id_uni[-1][-1]=min(result_id_uni[-1][-1],result_id_sent[i][-1])
    return result_id_uni


def write_result(write_path, write_path_id,write_path_text, result_text, result_id):
    with open(write_path_id, "w") as f:
        json.dump(result_id, f)

    with open(write_path_text, "w") as f:
        json.dump(result_text, f)


    with open(write_path, 'w', encoding='utf-8', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in tqdm(result_text):
            csvwriter.writerow(row)

def id2text(result_id,books_1,books_2,is_sent=True):
    result_text = []

    for pair in result_id:
        sent1=pair[0]
        sent2=pair[1]
        d=pair[2]

        if is_sent==True:
            bookname1,pagename1,ngram1,book_idx1=read_index_sent(books_1,sent1)
            bookname2, pagename2, ngram2,book_idx2 = read_index_sent(books_2, sent2)
        else:
            bookname1, pagename1, ngram1, book_idx1 = read_index_subsent(books_1, sent1)
            bookname2, pagename2, ngram2, book_idx2 = read_index_subsent(books_2, sent2)


        text=[bookname1,pagename1,ngram1,bookname2, pagename2, ngram2,sent1,sent2,d]
        result_text.append(text)

    return result_text