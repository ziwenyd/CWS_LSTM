import gensim
from gensim.models import word2vec
import pandas as pd
import pdb


def get_char_seg_corpus(word_seg_dir, char_seg_dir):
    """
    使用word_seg 文件
    写入char_seg_dir
    :param char_seg_dir:
    :return:
    """
    f = open(char_seg_dir, 'w', encoding='utf-8')
    for line in [line.replace(" ", '').replace('\n', '') for line in open(word_seg_dir, 'r').readlines()]:
        new = ""
        for i in range(len(line)):
            new = new + line[i] + " "
        f.write(new.strip() + '\n')
    f.close()


def get_char_model(corpus_dir, model_dir,wordVecLen):
    num_features = wordVecLen  # Word vector dimensionality, 如果是小于100M的数据用默认值（100）就可以,数据越多纬度越大训练效果越好
    min_word_count = 3  # Minimum word count 需要计算词向量的最小词频。这个值可以去掉一些很生僻的低频词，默认是5。如果是小语料，
    # 可以调低这个值。可以对字典做截断， 词频少于min_count次数的单词会被丢弃掉。
    num_workers = 16  # Number of threads to run in parallel
    context = 5  # Context window size 窗口大小，即词向量上下文最大距离，这个参数在我们的算法原理篇中标记为c。window越大，
    # 则和某一词较远的词也会产生上下文关系。默认值为5。在实际使用中，可以根据实际的需求来动态调整这个window的大小。
    # 如果是小语料则这个值可以设的更小。对于一般的语料这个值推荐在[5,10]之间。个人理解应该是某一个中心词可能与前后多个词相关，
    # 也有的词在一句话中可能只与少量词相关（如短文本可能只与其紧邻词相关）。
    downsampling = 1e-1  # Downsample setting for frequent words
    sentences = word2vec.LineSentence(corpus_dir)
    # print(sentences[:3])
    model = word2vec.Word2Vec(sentences, workers=num_workers,
                              size=num_features, min_count=min_word_count,
                              window=context, sg=1, sample=downsampling)
    model.init_sims(replace=True)
    # 保存模型，供日後使用
    model.save(model_dir)


def is_stop_word(context_word):
    # stopwords = [line.strip() for line in open('stop_words.txt', encoding='utf-8').readlines()]
    stopwords = []
    if context_word in stopwords:
        return True
    else:
        return False


def write_embedding_lookup_table(lookup_dir, model_dir):
    file = open(lookup_dir, 'w', encoding='utf-8')
    model = word2vec.Word2Vec.load(model_dir)
    n_dict = len(model.wv.vocab)
    n_dim = 100  # dimentionality of the character vector
    file.write(str(n_dict) + " " + str(n_dim) + '\n')
    for char in model.wv.vocab.keys():
        vector = ""
        for num in model.wv[char].tolist():
            vector = vector + str(num) + " "
        vector.strip()
        file.write(char + ' ' + vector + '\n')
    file.close()

def test(lookup_dir):
    line = lookup_dir.readline()
    print(line.split())
    line = lookup_dir.readline()
    print(line.split())


if __name__ == "__main__":
    word_seg_corpus_dir = '/home/midea/projects/ChineseWordSeg/CWS_LSTM/sighan2005/origin/msr_training.utf8'
    char_seg_corpus_dir = '/home/midea/projects/ChineseWordSeg/CWS_LSTM/sighan2005/origin/msr_char_seg.txt'
    model_dir = '/home/midea/projects/ChineseWordSeg/CWS_LSTM/msr.model'
    wordVecLen = 10
    lookup_dir = '/home/midea/projects/ChineseWordSeg/CWS_LSTM/PreTrainedWordEmbedding/charactor_OOVthr_50_%dv.txt' % wordVecLen
    # get_char_seg_corpus(word_seg_corpus_dir, char_seg_corpus_dir)
    print('getting model')
    get_char_model(char_seg_corpus_dir, model_dir,wordVecLen)
    print("---------------model is ready------------")
    write_embedding_lookup_table(lookup_dir, model_dir)
    print('DONE')
    test(open(lookup_dir))
