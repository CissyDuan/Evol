import json
from tqdm import tqdm


import csv

'''
result_path='./base/result_text_sent_unit.json'

f=open(result_path, 'r')
result=json.load(f)

bookname={}

for row in tqdm(result):
    source_title = row[0]
    if source_title not in bookname:
        bookname[source_title]=[]
        bookname[source_title].append(row)
    else:
        bookname[source_title].append(row)

for source_title in bookname.keys():
    rows=bookname[source_title]
    write_path='./base/sep_test/'+source_title+'.csv'
    with open(write_path, 'w', encoding='utf-8', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in rows:
            csvwriter.writerow(row)

'''
result_path='./history/result_text_sentcombine.json'

f=open(result_path, 'r')
result=json.load(f)

bookname={}

for row in tqdm(result):
    source_title = row[0]
    if source_title not in bookname:
        bookname[source_title]=[]
        bookname[source_title].append(row)
    else:
        bookname[source_title].append(row)

for source_title in bookname.keys():
    rows=bookname[source_title]
    write_path='./history/sep_sum/'+source_title+'.csv'
    with open(write_path, 'w', encoding='utf-8', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in rows:
            csvwriter.writerow(row)
