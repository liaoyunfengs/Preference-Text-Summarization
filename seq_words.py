import re
import jieba.posseg as pos
import os

'''读取文本，按词性分词'''
def pos_words(text):
    pstr = []
    words = pos.cut(text)
    for w in words:
        if w.flag in ['n', 'nr', 'ns', 'nt', 'nz']:
            if w.word not in pstr:
                pstr.append(w.word)
    print(pstr)
    return (pstr)


'''分类词典'''
def read_words(word):
    words = []
    with open('2_n_27294 - 副本.txt', 'r', encoding='utf-8')as file:
        lines = file.readlines()
        for line in lines:
            for i in range(len(word)):
                if word[i] in line:
                    words.append(line)
    print(words)
    return (words)


'''抽取类别'''
def extra_words(re_word, r):
    words = []
    for j in range(len(re_word)):
        for i in range(len(r)):
            p = re.compile('\w*' + ' ' + '\w*' + r[i])
            if p.findall(re_word[j]) not in words:
                words.append(p.findall(re_word[j]))
    words.remove([])
    print(words)
    return (words)


'''提取类别数字'''
def ex_num(ex_words):
    num = []
    for i in range(len(ex_words)):
        p = re.compile('\d+')
        pp = p.findall(str(ex_words[i]))
        list_int = list(map(int, pp))  # 仅提取列表中的数字
        for j in list_int:
            if j not in num:
                num.append(j)
    print(num)
    return (num)


'''按照类别生成列表，将对应的词放入对应的列表'''
def list_num(num, ex_words):
    a = []
    for i in range(len(num)):
        a.append([])
        for j in range(len(ex_words)):
            p = re.compile('\w*' + ' ' + '\w*' + str(num[i]))
            pp = p.findall(str(ex_words[j]))
            if pp not in a[i]:
                a[i].append(pp)
    print(a)
    return (a)

'''提取句子'''
def seq_words(num_list, text):
    seq = []
    for i in range(len(num_list)):
        seq.append([])  # 装分类的句子
        num_list[i].remove([])
        print(num_list[i])
        for j in range(len(num_list[i])):
            p = re.compile('\w*' + str(num_list[i][j]) + '\w*')
            seq[i].append(p.findall(text))
        print(seq)
    return (seq)

r = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
with open('text.txt') as file:
    lines = file.readlines()
    for line in lines:
        words = pos_words(line)  # 读取分好词性的词
        re_words = read_words(words)  # 从词典中查找带有类别的词
        ex_words = extra_words(re_words, r)  # 提取有类别的词语
        num = ex_num(ex_words)  # 提取类别数字
        num_list = list_num(num, ex_words)  # 按类别放入不同的列表
        seq = seq_words(num_list, line)  # 按类别提取句子

