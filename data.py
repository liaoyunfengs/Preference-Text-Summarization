import glob
import random
import struct
import sys

from tensorflow.core.example import example_pb2

# Special tokens
PARAGRAPH_START = '<p>'
PARAGRAPH_END = '</p>'
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
UNKNOWN_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
DOCUMENT_START = '<d>'
DOCUMENT_END = '</d>'


class Vocab(object):
    """Vocabulary class for mapping words and ids.用于映射单词和ID的词汇类"""

    def __init__(self, vocab_file, max_size):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0

        with open(vocab_file, 'r', encoding='UTF-8') as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 2:
                    sys.stderr.write('Bad line: %s\n' % line)
                    continue
                if pieces[0] in self._word_to_id:
                    raise ValueError('Duplicated word: %s.' % pieces[0])
                self._word_to_id[pieces[0]] = self._count
                self._id_to_word[self._count] = pieces[0]
                #pieces[0] 从list的第一个输出
                self._count += 1
                if self._count > max_size:
                    raise ValueError('Too many words: >%d.' % max_size)

    def CheckVocab(self, word):
        if word not in self._word_to_id:
            return None
        return self._word_to_id[word]

    def WordToId(self, word):
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def IdToWord(self, word_id):
        if word_id not in self._id_to_word:
            raise ValueError('id not found in vocab: %d.' % word_id)
        return self._id_to_word[word_id]

    def NumIds(self):
        return self._count


def ExampleGen(data_path, num_epochs=None):
    """Generates tf.Examples from path of data files.
      Binary data format: <length><blob>. <length> represents the byte size
      of <blob>. <blob> is serialized tf.Example proto. The tf.Example contains
      the tokenized article text and summary.
    Args:
      data_path: path to tf.Example data files.
      num_epochs: Number of times to go through the data. None means infinite.
    Yields:
      Deserialized tf.Example.
    If there are multiple files specified, they accessed in a random order
    从数据文件的路径生成TF.示例。
    二进制数据格式：<长度> BLUB>。<长度>表示字节大小
    < BLBB >.<BLUB>是序列化的TF。示例PROTO。TF示例包含
    记号化的文章正文和摘要。
    ARG:
    DATAYPATH：TF的路径。示例数据文件。
    NoimeEn历险记：通过数据的次数。没有意味着无限。
    产量：
    反序列化TF.示例
    如果指定了多个文件，则以随机顺序访问它们。.
    """
    epoch = 0
    while True:
        if num_epochs is not None and epoch >= num_epochs:
            break
        filelist = glob.glob(data_path)
        assert filelist, 'Empty filelist.'
        random.shuffle(filelist)
        #随机打乱列表
        for f in filelist:
            reader = open(f, 'rb')
            while True:
                len_bytes = reader.read(8)
                if not len_bytes: break
                str_len = struct.unpack('q', len_bytes)[0]
                #struct.pack(fmt,v1,v2,……)　
                #将v1,v2等参数的值进行一层包装，包装的方法由fmt指定。被包装的参数必须严格符合fmt。最后返回一个包装后的字符串
                #struct.unpack(fmt,string)
                #返回一个由解包数据(string)得到的一个元组(tuple), 即使仅有一个数据也会被解包成元组。其中len(string) 必须等于 calcsize(fmt)，这里面涉及到了一个calcsize函数。
                #q 长整数(long) s 字符串(string)
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                yield example_pb2.Example.FromString(example_str)

        epoch += 1


def Pad(ids, pad_id, length):
    """Pad or trim list to len length.
    Args:
      ids: list of ints to pad
      pad_id: what to pad with
      length: length to pad or trim to
    Returns:
      ids trimmed or padded with pad_id
      填充或修剪列表到LeN长度。
    ARG:
      IDS：ITO到PAD的列表
      PADYID:用什么填充
      长度：填充或修剪长度
    返回：
      用PADYID修剪或填充的IDS
    """
    assert pad_id is not None
    assert length is not None
    #assert 断言 假设
    if len(ids) < length:
        a = [pad_id] * (length - len(ids))
        return ids + a
    else:
        return ids[:length]


def GetWordIds(text, vocab, pad_len=None, pad_id=None):
    """Get ids corresponding to words in text.
    Assumes tokens separated by space.
    Args:
      text: a string
      vocab: TextVocabularyFile object
      pad_len: int, length to pad to
      pad_id: int, word id for pad symbol
    Returns:
      A list of ints representing word ids.
    获取对应于文本中的单词的ID。
    假设由空间分隔的令牌。
    ARG:
    文本：字符串
    文本文件对象
    PADYLLIN: INT，长度到PAD
    PADYID:INT，PAD符号的字ID
    返回：
    表示单词ID的int类型的列表。
    """
    ids = []
    for w in text.split():
        i = vocab.WordToId(w)
        if i >= 0:
            ids.append(i)
        else:
            ids.append(vocab.WordToId(UNKNOWN_TOKEN))
    if pad_len is not None:
        return Pad(ids, pad_id, pad_len)
    return ids
"""
    import jieba
    ids = []
    for w in jieba.cut(text)
       i = vocab.WordToId(w)
        if i >= 0:
            ids.append(i)
        else:
            ids.append(vocab.WordToId(UNKNOWN_TOKEN))
    if pad_len is not None:
        return Pad(ids, pad_id, pad_len)
    return ids 
"""

def Ids2Words(ids_list, vocab):
    """Get words from ids.
    Args:
      ids_list: list of int32
      vocab: TextVocabulary object
    Returns:
      List of words corresponding to ids.
    从IDS中获取单词。
    ARG:
    IDS3列表：32的列表
    文本词汇对象
    返回：
    与IDS对应的单词列表。
    """
    assert isinstance(ids_list, list), '%s  is not a list' % ids_list
    return [vocab.IdToWord(i) for i in ids_list]


def SnippetGen(text, start_tok, end_tok, inclusive=True):
    """Generates consecutive snippets between start and end tokens.
    Args:
      text: a string
      start_tok: a string denoting the start of snippets
      end_tok: a string denoting the end of snippets
      inclusive: Whether include the tokens in the returned snippets.
    Yields:
      String snippets
    在起始令牌和结束令牌之间生成连续代码段。
    ARG:
    文本：字符串
    StrutsTok:一个表示代码段开始的字符串
    Enthotok：一个表示片段结尾的字符串
    包含：是否在返回的片段中包含令牌。
    产量：
    字符串片段
    """
    #key_1 = 'b"'
    cur = 0
    #text.features.feature[key_1].bytes_list.value[0]
    #GetExFeatureText(text, 'b"')
    while True:
        try:
            start_p = text.index(start_tok, cur)
            end_p = text.index(end_tok, start_p + 1)
            cur = end_p + len(end_tok)
            if inclusive:
                yield text[start_p:cur]
            else:
                yield text[start_p + len(start_tok):end_p]
        except ValueError as e:
            raise StopIteration('no more snippets in text: %s' % e)


def GetExFeatureText(ex, key):
    return ex.features.feature[key].bytes_list.value[0]
#返回特征值


def ToSentences(paragraph, include_token=True):
    """Takes tokens of a paragraph and returns list of sentences.
    Args:
      paragraph: string, text of paragraph
      include_token: Whether include the sentence separation tokens result.
    Returns:
      List of sentence strings.
    取一个段落的记号并返回句子的列表。
    ARG:
    段落：段落的字符串、文本
    包含令牌：是否包括句子分离令牌的结果。
    返回：
    句子字符串表。
    """
    s_gen = SnippetGen(paragraph, SENTENCE_START, SENTENCE_END, include_token)
    return [s for s in s_gen]