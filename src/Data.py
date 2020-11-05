import numpy as np
from collections import OrderedDict


class Data(object):
    def __init__(self, path_lookup_table, wordVecLen, path_train_data, path_test_data,
                 flag_random_lookup_table, dic_label, use_bigram_feature, random_seed, flag_toy_data,
                 path_dev_data=None):
        # numpy.random.RandomState()是一个伪随机数生成器。那么伪随机数是什么呢？
        # 伪随机数是用确定性的算法计算出来的似来自[0,1]均匀分布的随机数序列。并不真正的随机，但具有类似于随机数的统计特征，如均匀性、独立性等。(摘自《百度百科》)
        self.rng = np.random.RandomState(random_seed)

        self.dic_c2idx = {}
        self.dic_idx2c = {}
        self.wordVecLen = wordVecLen
        f = open(path_lookup_table, 'r')
        li = f.readline()# get the FIRST line in f, is a string, should contain 2 numbers seperated by space, number of words & word vec dimention
        # print("type of li: ")
        # print(type(li))
        # print("size of li: ")
        # print(len(li))
        # print(li)
        # print("type of li[0]:")
        # print(type(li[0]))
        # print(li[0])
        li = li.split()
        n_dict = int(li[0]) #word dictionary size
        self.n_unigram = n_dict

        # RandomState.normal :
        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.RandomState.normal.html#numpy.random.RandomState.normal
        # Draw random samples from a normal (Gaussian) distribution.
        #[Parameters]
        # loc: float or array_like of floats Mean (“centre”) of the distribution.
        # scale: float or array_like of floats; Standard deviation (spread or “width”) of the distribution. Must be non-negative.
        # size: int or tuple of ints, optional
        #       Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn.
        #       If size is None (default), a single value is returned if loc and scale are both scalars.
        #       Otherwise, np.broadcast(loc, scale).size samples are drawn.
        #[Returns]
        # out: ndarray or scalarDrawn samples from the parameterized normal distribution.


        v_lt = self.rng.normal(loc=0.0, scale=0.01, size=(n_dict, wordVecLen))

        # lookup_table = np.zeros([n_dict, 25],dtype = np.float32)
        self.unigram_table = np.asarray(v_lt, dtype=np.float32)
        n_dim = int(li[1])
        # print(li)
        print('---------------------preparing c2idx & idx2c dictionary--------------------')
        for i in range(n_dict):
            li = f.readline() # get the SECOND line for the first iteration of this for loop
            li = str(li)
            li = li.split()
            if (len(li) != wordVecLen + 1):
                print("len(li) != wordVecLen+1")
                print("len(li) is : " + str(len(li)))
                print("wordVecLen is " + str(wordVecLen))
                continue

            self.dic_c2idx[li[0]] = i
            # print(self.dic_c2idx)
            self.dic_idx2c[i] = li[0]
            if (flag_random_lookup_table == True): continue
            for j in range(n_dim):
                self.unigram_table[i][j] = float(li[j + 1])
        f.close()
        if (use_bigram_feature == True):
            v_lt = self.rng.normal(loc=0.0, scale=0.01, size=(n_dict * n_dict, wordVecLen))
            self.bigram_table = np.asarray(v_lt, dtype=np.float32)
            for i in range(n_dict):
                for j in range(n_dict):
                    self.bigram_table[i * n_dict + j] = 0.5 * (self.unigram_table[i] + self.unigram_table[j])
        print('GOT: ----c2idx, idx2c dictionaries---------')
        self.dic_c2idx['<OOV>'] = 00 #added by ziwen, might be wrong.
        self.dic_c2idx['<BOS>'] = 000 #added by ziwen, might be wrong.
        self.dic_c2idx['<EOS>'] = 0000 #added by ziwen, might be wrong.
        data_train = []
        data_sentence = []
        label_train = []
        label_sentence = []
        # f = open('pkutrain_noNUMENG.utf8', 'r')
        f = open(path_train_data, 'r') # the file contains gold-segmented sentences.
        li = f.readlines()
        print("li at here is: ")
        print(type(li))
        print(len(li))
        print(li[0])

        f.close()

        for line in li:
            # print('processing one line------')
            # print(line)
            line = str(line)
            line_t = line.split()
            # print('line_t size is : ' + str(len(line_t)))
            # print('line_t is : ' + str(line_t))

            if (len(line_t) == 0):
                # print('len(line_t) == 0')
                if (len(data_sentence) == 0):
                    # print('len(data_sentence) == 0')
                    continue
                # print('len(data_sentence) != 0')
                data_train.append(data_sentence)
                label_train.append(label_sentence)
                # print('appended label_sentence to label_train: ' + 'label_sentence: ' + str(label_sentence))
                data_sentence = []
                label_sentence = []
                continue
            ch = line_t[0]
            # print('ch (line_t[0]) is : ' + ch)

            if (self.dic_c2idx.get(ch) == None):
                print(ch +' is not in the dic_c2idx')
                ch = self.dic_c2idx['<OOV>']

            else:
                ch = self.dic_c2idx[ch]
            data_sentence += [ch]

            # temp = self.dic_c2idx[line_t[1]]
            # print('temp is : ' + str(temp))

            # print('line_t[1] is: ' + line_t[1])
            label_sentence += [dic_label[line_t[1]]]

        if (path_dev_data == None):
            l_len = len(data_train)
            thr = int(l_len * 0.9)
            data_dev = data_train[thr:] # 10%作为development set
            label_dev = label_train[thr:]
            data_train = data_train[:thr]# 90%作为training set
            label_train = label_train[:thr]
            print('90% training set label_train: size = ' + str(len(label_train)))
            print('first element in label_train: ' + str(label_train[0])+',type = ' + str(type(label_train[0])))

        else:
            data_dev = []
            data_sentence = []
            label_dev = []
            label_sentence = []
            # f = open('pkutrain_noNUMENG.utf8', 'r')
            f = open(path_dev_data, 'r')
            li = f.readlines()
            f.close()

            for line in li:
                line = str(line)
                line_t = line.split()
                if (len(line_t) == 0):
                    if (len(data_sentence) == 0):
                        continue
                    data_dev.append(data_sentence)
                    label_dev.append(label_sentence)
                    data_sentence = []
                    label_sentence = []
                    continue
                ch = line_t[0]
                if (self.dic_c2idx.get(ch) == None):
                    ch = self.dic_c2idx['<OOV>']
                else:
                    ch = self.dic_c2idx[ch]
                data_sentence += [ch]

                label_sentence += [dic_label[line_t[1]]]

        data_test = []
        label_test = []
        data_sentence = []
        label_sentence = []
        # f = open('pkutest_noNUMENG.utf8', 'r')
        f = open(path_test_data, 'r')
        li = f.readlines()
        f.close()
        for line in li:
            line = str(line)
            line_t = line.split()
            if (len(line_t) == 0):
                if (len(data_sentence) == 0):
                    continue
                data_test.append(data_sentence)
                label_test.append(label_sentence)
                data_sentence = []
                label_sentence = []
                continue
            ch = line_t[0]
            if (self.dic_c2idx.get(ch) == None):
                ch = self.dic_c2idx['<OOV>']
            else:
                ch = self.dic_c2idx[ch]
            data_sentence += [ch]
            label_sentence += [dic_label[line_t[1]]]

        if (flag_toy_data == False):
            pass
        else:
            l_len = len(data_train)
            thr = int(l_len * flag_toy_data)
            data_train = data_train[:thr]
            label_train = label_train[:thr]

            data_dev = data_train[:]
            label_dev = label_train[:]
            l_len = len(data_test)
            thr = int(l_len * flag_toy_data)
            data_test = data_test[:thr]
            label_test = label_test[:thr]

        self.data_train = data_train
        self.label_train = label_train
        self.data_dev = data_dev
        self.label_dev = label_dev
        self.data_test = data_test
        self.label_test = label_test

    def shuffle(self, data_in, label_in):
        l_len = len(data_in)
        # 尝试使用range()创建整数列表（导致“TypeError: ‘range’ object does not support item assignment”）有时你想要得到一个有序的整数列表，所以range()
        # 看上去是生成此列表的不错方式。然而，你需要记住range() 返回的是“range object”，而不是实际的list 值。
        # 将上面例子的代码： a = range(0,N)改为a = list(range(0,N)) 就好啦！
        #from: https://blog.csdn.net/wanglin_lin/article/details/50819657
        permu = list(range(l_len))
        self.rng.shuffle(permu)
        data_out = []
        label_out = []
        for i in range(l_len):
            data_out.append(data_in[permu[i]])
            label_out.append(label_in[permu[i]])
        return (data_out, label_out)

    def bigram2id(self, id_bigram):
        (m1, m2) = id_bigram
        return m1 * self.n_unigram + m2

    def display(self, sentences):
        n = len(sentences)
        for sentence in sentences:
            s = ''
            for ch in sentence:
                # s += (self.dic_idx2c[ch].encode('utf-8') + ' ')
                s +=self.dic_idx2c[ch] + " "
            print(s + '\n')
