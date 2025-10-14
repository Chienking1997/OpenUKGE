from collections import deque
import random


class Word2VecData:
    def __init__(self, input_data, min_count):
        self.index = 0
        self.input_data = input_data
        self.input_triples = self.convert_triples()
        self.min_count = min_count  # 要淘汰的低频数据的频度
        self.wordId_frequency_dict = dict()  # 词id-出现次数 dict
        self.word_count = 0  # 单词数（重复的词只算1个）
        self.word_count_sum = 0  # 单词总数 （重复的词 次数也累加）
        self.sentence_count = 0  # 句子数
        self.id2word_dict = dict()  # 词id-词 dict
        self.word2id_dict = dict()  # 词-词id dict
        self._init_dict()  # 初始化字典
        self.sample_table = []
        self._init_sample_table()  # 初始化负采样映射表
        self.word_pairs_queue = deque()
        # 结果展示
        print('Word Count is:', self.word_count)
        print('Word Count Sum is', self.word_count_sum)
        print('Sentence Count is:', self.sentence_count)

    def convert_triples(self):
        """
        Convert the relationship indices in triples to distinguish them from entity indices.
        The function adds an 'r' prefix to the relationship index in each triple.

        Args:
            self.input_data (list of lists): A list of triples, where each triple is [head, relation, tail].

        Returns:
            list of lists: A new list of triples with the relation index converted.
        """
        converted_triples = []

        for triple in self.input_data:
            head, relation, tail, _ = triple
            # Convert relation index to a string with 'r' prefix
            relation_str = f"r{relation}"
            # Append the converted triple to the new list
            converted_triples.append([head, relation_str, tail])

        return converted_triples

    def _init_dict(self):
        word_freq = dict()
        # 统计 word_frequency
        for line in self.input_triples:
            self.word_count_sum += len(line)
            self.sentence_count += 1
            for word in line:
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1
        word_id = 0
        # 初始化 word2id_dict,id2word_dict, wordId_frequency_dict字典
        for per_word, per_count in word_freq.items():
            if per_count < self.min_count:  # 去除低频
                self.word_count_sum -= per_count
                continue
            self.id2word_dict[word_id] = per_word
            self.word2id_dict[per_word] = word_id
            self.wordId_frequency_dict[word_id] = per_count
            word_id += 1
        self.word_count = len(self.word2id_dict)

    def _init_sample_table(self):
        sample_table_size = int(1e8)
        pow_frequency = [count ** 0.75 for count in self.wordId_frequency_dict.values()]  # 词频指数为3/4
        word_pow_sum = sum(pow_frequency)  # 所有词的总词频
        ratio_array = [freq / word_pow_sum for freq in pow_frequency]  # 词频比率
        word_count_list = [int(round(ratio * sample_table_size)) for ratio in ratio_array]
        for word_index, word_freq in enumerate(word_count_list):
            self.sample_table.extend([word_index] * word_freq)  # 生成list，内容为各词的id，list中每个id重复多次

    # 获取mini-batch大小的 正采样对 (Xw,w) Xw为上下文id数组，w为目标词id。上下文步长为window_size，即2c = 2*window_size

    def get_batch_pairs(self, batch_size, window_size):
        # 确保 word_pairs_queue 有足够的正采样对
        while len(self.word_pairs_queue) < batch_size:
            # 先填充 10000 条三元组
            for _ in range(10000):
                # 检查是否需要重新循环三元组
                if self.index >= len(self.input_triples):
                    self.index = 0  # 如果已经遍历到末尾，重置指针从头开始

                # 获取当前的三元组
                triple = self.input_triples[self.index]
                self.index += 1  # 更新指针，准备下一次选择下一个三元组
                wordId_list = []  # 一句中的所有word 对应的 id
                for word in triple:
                        wordId_list.append(self.word2id_dict[word])
                # 寻找正采样对 (context(w), w) 加入正采样队列
                for i, wordId_w in enumerate(wordId_list):
                    context_ids = []
                    for j, wordId_u in enumerate(wordId_list[max(i - window_size, 0):i + window_size + 1]):
                        assert wordId_w < self.word_count
                        assert wordId_u < self.word_count
                        if i == j:  # 上下文=中心词 跳过
                            continue
                        elif max(0, i - window_size + 1) <= j <= min(len(wordId_list), i + window_size - 1):
                            context_ids.append(wordId_u)
                    if len(context_ids) == 0:
                        continue
                    self.word_pairs_queue.append((context_ids, wordId_w))

        result_pairs = []  # 返回mini-batch大小的正采样对
        for _ in range(batch_size):
            result_pairs.append(self.word_pairs_queue.popleft())  # 取出正采样对

        return result_pairs

    def get_batch_pairs_sg(self, batch_size, window_size):
        while len(self.word_pairs_queue) < batch_size:
            for _ in range(10000):  # 先加入10000条，减少循环调用次数

                if self.index >= len(self.input_triples):
                    self.index = 0  # 如果已经遍历到末尾，重置指针从头开始

                # 获取当前的三元组
                triple = self.input_triples[self.index]
                self.index += 1  # 更新指针，准备下一次选择下一个三元组
                wordId_list = []  # 一句中的所有word 对应的 id
                for word in triple:
                    if word in self.word2id_dict:
                        wordId_list.append(self.word2id_dict[word])
                # 寻找正采样对 (w, v) 加入正采样队列
                for i, wordId_w in enumerate(wordId_list):
                    for j, wordId_v in enumerate(wordId_list[max(i - window_size, 0):i + window_size + 1]):
                        assert wordId_w < self.word_count
                        assert wordId_v < self.word_count
                        if i == j:  # 上下文 = 中心词 跳过
                            continue
                        self.word_pairs_queue.append((wordId_w, wordId_v))
        result_pairs = [self.word_pairs_queue.popleft() for _ in range(batch_size)]
        return result_pairs

    # 获取负采样 输入正采样对数组 positive_pairs，以及每个正采样对需要的负采样数 neg_count 从采样表抽取负采样词的id
    def get_negative_sampling(self, len_positive_pairs, neg_count):
        neg = [random.choices(self.sample_table, k=neg_count) for _ in range(len_positive_pairs)]
        return neg

    # 估计数据中正采样对数，用于设定batch
    def evaluate_pairs_count(self, window_size):
        return self.word_count_sum * (2 * window_size - 1) - (self.sentence_count - 1) * (1 + window_size) * window_size
