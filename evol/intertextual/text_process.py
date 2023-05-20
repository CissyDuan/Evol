from tqdm import tqdm
import opencc
import os
import re
from tool import *
import json

converter = opencc.OpenCC('t2s')

# =============================================================================
# 异体字替换+繁简体转换+过滤标点符号
# =============================================================================

med=['傷寒論',
'急就篇',
'新刊王氏脈經',
'神農本草經',
'葛仙翁肘後備急方',
'金匱要略',
'養性延命錄',
'難經',
'靈樞經',
'黃帝內經',]

#med=[0,1,2,3,4,5]

#if word in variantDict.keys():  # 替换异体字
#    outText = outText + str(variantDict[word])
#else:
#    outText = outText + str(word)

mark=['、','‘','’','“','”','\'','\"']
def cuttingText(text):

    outText = ''
    text = converter.convert(text)
    try:

        while text[0] in stop_punc or text[0]== ' ':
            text=text[1:]
        while text[-1] in stop_punc or text[-1]== ' ':
            text=text[:-1]

        while (text != ''):
            word = text[0:1]
            if word != ' ' and word not in mark:  # 过滤标点符号
                outText = outText + str(word)

            text = text[1:]

        return outText
    except:
        #print(text)
        return ''
def filtstop(text):
    outText = ''
    while (text != ''):
        word = text[0:1]
        if word not in stopwords and word not in stop_punc:
            outText = outText + str(word)
        text = text[1:]

    '''if outText!='':
        while outText[0] in stop_punc or outText[0]== ' ':
            outText=outText[1:]
            if outText=='':
                break
    if outText!='':
        while outText[-1] in stop_punc or outText[-1]== ' ':
            outText=outText[:-1]
            if outText=='':
                break'''

    return outText

def read_punc():  # 读取标点符号
    mark = [line.strip() for line in open('../punc.txt', 'r', encoding='utf-8')]
    return mark
def read_stopword():  # 读取标点符号
    stopword = [line.strip() for line in open('../stopword.txt', 'r', encoding='utf-8')]
    return stopword
def read_variant():  # 读取异体字
    variantDict = {}
    for line in open('../variance.txt', 'r', encoding='utf-8'):
        value = line[0:1]
        line = line[1:]
        while (line != ''):
            key = line[0:1]
            line = line[1:]
            if key != '':
                variantDict[key] = value
    return variantDict

stop_punc = read_punc()
variantDict = read_variant()
stopwords=read_stopword()

bookmark='<>《》'
def is_book(text):
    for mark in bookmark:
        if mark in text:
            return True
    return False

#去无效

def ismed(text):
    unit=['升', '两', '枚','丸','味','片','斗','合','斤','服','分','钱']
    number = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '半','再']
    text_mini=re.split(r'[,，]', text)
    mark=0
    for t in text_mini:
        if len(t)>=2:
            if t[-1] in unit and t[-2] in number:
                mark+=1
            elif t[-1] in number and t[-2] in unit:
                mark+=1
            elif t[-2:] in ['汤方']:
                mark+=1
        if len(t)>=3:
            if t[-3:]=='方寸匕':
                mark+=1

    if mark>0:
        return True
    else:
        return False

def istime(text):
    time = ['年', '月', '日','岁']
    month = ['月']
    number = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '正','元']
    #if len(text)>10:
    #    return False
    if text[-1] in time and text[-2] in number:
        return True
    elif text[0] in number and text[1] in month:
        return True
    elif text[1] in number and text[2] in month:
        return True
    else:
        return False


def islength(text):
    lengthword = ['尺', '丈','寸','人','篇','家']
    number = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十','百','余','许','数']
    #if len(text)>10:
    #    return False
    if text[-1] in lengthword and text[-2] in number:
        return True
    elif text[-1] in number and text[-2] in lengthword and text[-3] in number:
        return True
    else:
        return False

def isnumbers(text):
    #if len(text)>10:
    #    return False
    #else:
    time = ['年', '月', '日','春','夏','秋','冬','岁','地']
    lengthword = ['尺', '丈','寸','长','高','里','东','南','西','北','方','度','分','广','步']
    unitword=['人','石','篇','种','家','卷','口','户','秩','中','凡','次','食','邑','疋','匹','州','兵','城','府','马','槲','斛','部']
    connectword=['至','去','以','为','之','有','上','比','者']
    number = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '余','数','正','百','千','万','亿','元','许']
    tiandi=['甲','乙','丙','丁','戊','己','庚','辛','壬','子','丑','寅','卯','辰','巳','午','未','申','酉','戌','亥']
    com=time+lengthword+number+tiandi+unitword+connectword
    count=0
    length=0
    for char in text:
        if char in com:
            count+=1
        if char not in stop_punc:
            length+=1
    if length<=3:
        return True
    elif count/length>=0.7:
        return True

number=['一','二','三','四','五','六','七','八','九','零']


f=open('../nomeaning.txt')
nomeaning = f.read().splitlines()
def isaction(text):
    name=['孔子','太公','桓子','管子','晏子','仲尼','夫子','桓公','孟子','莊子','哀公','壺子','列子']
    action=['問','見','告']
    flag=0
    if text[-2:] in name:
        if len(text)==2:
            flag=1
        bound=min(5,len(text))
        before=text[-bound:-2]
        for char in before:
            if char in action:
                flag=1
    if flag==0:
        return False
    else:
        return True

def isfilter(sent,bookname):
    if sent == '' or len(sent) <= 3:
        return True
    elif isaction(sent):
        return True
    elif bookname in med and ismed(sent):
        return True
    elif sent[-1] == '曰':
        return True
    elif sent[-3:] == '樂所奏':
        return True
    elif sent[-1] == '年' and sent[-2] in number:
        return True
    elif istime(sent) or islength(sent) or isnumbers(sent):
        return True
    elif sent[-2:] in ['年卒', '即位', '既丧', '既殁', '既殡', '百石', '千石', '十斛', '百斛', '千斛', '千匹', '帝时', '闻之', '请之']:
        return True
    elif sent[0] == '年' and sent[-1] == '卒':
        return True
    elif len(sent) <= 10:
        sent_filt = filtstop(sent)
        if sent_filt == '' or len(sent_filt) < 3:
            return True
    else:
        return False

def textToCode(text,bookname):
    if isfilter(text,bookname):
        return None
    elif is_book(text):
        return None
    elif text in nomeaning:
        return None

    else:
        sent = cuttingText(text)
        if isfilter(text, bookname):
            return None
        else:
            return sent



def sent_split(sent,max_length):
    if len(sent) <= max_length:
        return [sent]

    else:
        sent_mini = re.split(r'[,，]', sent)
        sent_num = len(sent_mini)
        #print(sent_mini)
        if sent_num >= 3:
            mid = int(sent_num / 2)
            a = '，'.join(sent_mini[:mid])+'，'
            #print(a)
            a=sent_split(a,max_length)

            c = '，'.join(sent_mini[mid:])
            #print(c)
            c=sent_split(c,max_length)
            sent_mini =a+c
        else:
            sent_mini = [sent]
        return sent_mini

del_char=['(',')','（','）','〔','〕','〈','〉']
del_char_left=['(','（']
del_char_right=[')','）']
def remove_diff(text):
    length=len(text)
    outtext=''
    i=0

    while i<length:

        if i<length-2 and i>2 and text[i]=='〔' and text[i+2]=='〕' and text[i-1] in del_char_right and text[i-3] in del_char_left:
            i+=3
        elif i<length-3 and i>3 and text[i]=='〔' and text[i+3]=='〕' and text[i-1] in del_char_right and text[i-4] in del_char_left:
            i+=4
        elif i<length-4 and i>4 and text[i]=='〔' and text[i+4]=='〕' and text[i-1] in del_char_right and text[i-5] in del_char_left:
            i+=5
        elif text[i] in del_char:
            i+=1
        else:
            outtext+=text[i]
            i+=1
    return outtext

remove_char=['〔','〕','〈','〉']

#remove_char=['(',')','（','）']
#del_char_left=['〔']
#del_char_right=['〕']
def remove_mark(text):
    outtext=''
    del_flag=0
    for char in text:
        if del_flag==1 and char not in del_char_right:
            continue
        elif del_flag==1 and char in del_char_right:
            del_flag=0
            continue
        elif char in del_char_left:
            del_flag=1
            continue
        elif char in remove_char:
            continue
        else:
            outtext+=char
    return outtext

conflict_mark='、,，。！？!?：:;'
def remove_multimark(text):
    if text=='' or len(text)==1:
        return text
    else:
        outtext=''
        skip_flag=0
        for i in range(len(text)-1):
            if skip_flag==1:
                if text[i+1] not in conflict_mark:
                    skip_flag=0
                continue

            elif text[i] in conflict_mark and text[i+1] in conflict_mark:
                skip_flag=1
            outtext+=text[i]
        if skip_flag==0:
            outtext+=text[-1]
        return outtext
















