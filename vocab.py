import os
import re
import charade
import jieba

error_box = []


def word_cut(origin, sync=True):
    gen = jieba.cut(origin)
    words = list(gen)
    if sync:
        with open('sync.txt') as f_obj:
            line = f_obj.read().split('\n')
            line.append('\n')
        for i in line:
            while i in words:
                words.remove(i)
    return words


def word_freq(words):
    words_s = list(set(words))
    len_s = len(words_s)
    words_f = [0] * len_s
    for i in range(len_s):
        words_f[i] = words.count(words_s[i])
    most = max(words_f)
    freq = {}
    for k in range(most + 1)[::-1]:
        for i in range(len_s):
            if words_f[i] == k:
                freq[words_s[i]] = words_f[i]
    return freq


def text_reader(file):
    try:
        with open(file, 'rb') as f_obj:
            origin = f_obj.read()
            chartype = charade.detect(origin)
            try:
                if 'GB' in chartype['encoding']:
                    article = origin.decode('gbk')
                else:
                    article = origin.decode(chartype['encoding'])
            except UnicodeDecodeError and TypeError:
                error_box.append('Error - File decode is failed: ' + file)
                article = None
    except FileNotFoundError:
        error_box.append('Error - File is not exist: ' + file)
        article = None
    if article:
        while '=' in article:
            article = article.replace('=', '等于')
    return article


def file_list(root):
    file_dir = os.listdir(root)
    if '__output__' in file_dir:
        file_dir.remove('__output__')
    if '__log__' in file_dir:
        file_dir.remove('__log__')
    return file_dir


class WordFreq:
    def __init__(self, filename):
        self.store = filename
        self.freq = {}

    def load(self):
        try:
            with open(self.store, 'r', encoding='utf-8') as fp:
                for line in fp.readlines():
                    key_val = line.split(' ')
                    key_val[0] = re.sub('\s', '', key_val[0])
                    if key_val[0] in self.freq.keys():
                        self.freq[key_val[0]] += key_val[1]
                    else:
                        if key_val[0] != '':
                            self.freq[key_val[0]] = key_val[1]
        except FileNotFoundError or UnicodeDecodeError:
            pass

    def extract(self, folder):
        filelist = file_list(folder)
        for file in filelist:
            try:
                article = text_reader(folder + '/' + file)
            except FileNotFoundError:
                error_box.append('Error - File is not exist: ' + file)
                article = None
            if article:
                words = word_cut(article, sync=False)
                freq = word_freq(words)
                for i, j in freq.items():
                    i = re.sub('\s', '', i)
                    if i in self.freq.keys():
                        self.freq[i] += j
                    else:
                        if i != '':
                            self.freq[i] = j

    def prompt(self):
        with open(self.store, 'w', encoding='utf-8') as fp:
            for i, j in self.freq.items():
                rcd = i + ' ' + str(j) + '\n'
                fp.write(rcd)


myWord = WordFreq(r'D:\自然语言处理\LCSTS_ORIGIN\NLP数据集准备\simple-300\__vocab__')
myWord.extract(r'D:\自然语言处理\LCSTS_ORIGIN\NLP数据集准备\simple-300\start')
myWord.extract(r'D:\自然语言处理\LCSTS_ORIGIN\NLP数据集准备\simple-300\start_abstract')
myWord.prompt()
