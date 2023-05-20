import json
from tqdm import tqdm
books=json.load(open('index_book_sum.json','r'))
name_msg=[]
name_msg_book=[]
for book_id in books:
    book_msg=books[book_id]
    bookname=book_msg['bookname']
    name_msg_book.append([book_id,bookname])
    pagenames=[]
    page_msg=book_msg['pages']
    for page_id in page_msg:
        name=[book_id,bookname,page_id,page_msg[page_id]['pagename']]
        name_msg.append(name)

import csv
write_path='name_page.csv'
with open(write_path, 'w', encoding='utf-8', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    for row in tqdm(name_msg):
        csvwriter.writerow(row)

write_path='name_book.csv'
with open(write_path, 'w', encoding='utf-8', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    for row in tqdm(name_msg_book):
        csvwriter.writerow(row)