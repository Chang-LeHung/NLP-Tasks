import re
import plotly.graph_objs as go
import random


class TTRCompare(object):
    """
    Usage Example:
    ```
    ttrs_obj = TTRCompare("Pubmed.txt", "Brow Corpus.txt", repeat=15)
    ttrs_obj.shape()
    ttrs_obj.visual()
    ttrs_obj.ttrs_detail()
    ```
    """
    def __init__(self, path1, path2, repeat=1, encoding="utf-8", max_words_num=40000):
        self._path1 = path1
        self._path2 = path2
        self.repeat = repeat
        self.nums1 = None
        self.nums2 = None
        self.max_length = 0
        self.ttrs1 = None
        self.ttrs2 = None
        self.words1 = None
        self.words2 = None
        self.encoding = encoding
        self.max_words_num = max_words_num

    def shape(self):
        self.words1 = self._load_data(self._path1)
        self.words2 = self._load_data(self._path2)
        self.max_length = min(len(self.words1), len(self.words2), self.max_words_num)
        self.nums1, self.ttrs1 = self.find_ttr_continuous(self.words1, repeat=self.repeat)
        self.nums2, self.ttrs2 = self.find_ttr_continuous(self.words2, repeat=self.repeat)

    def _words_count(self, word_sequence: list) -> list:
        """
        统计每个单词数出现的字数
        返回单词和其出现的次数
        return example : [("apple", 5)]
        """
        words = dict()
        for word in word_sequence:
            if word == "":
                continue
            words[word] = words.get(word, 0) + 1
        return list(words.items())

    def _load_data(self, path):
        """
        通过路径加载数据
        return example : ["apple", "banana", "apple"]
        """
        with open(path, "r+", encoding=self.encoding) as fp:
            data = re.split(r"[ \'\"(){}!-=/@#$%^&*~?\n_.]", fp.read())
            ans = []
            for char in data:
                if not char == "":
                    ans.append(char)
            return ans

    def _random_choice(self, words: list, num: int) -> list:
        """
        从words中随机抽取num个单词
        """
        if num > len(words):
            raise ValueError("num is bigger than length of words list")
        return random.choices(words, k=num)

    def find_ttr_continuous(self, word_count, repeat=10):
        """
        从word_count抽取k个单词重复计算repeat词，去ttr的平均值
        k = 1, 2, 3, ..., len(word_count)
        """
        nums = [i for i in range(1, self.max_length)]
        ttrs = []
        for num in nums:
            ttr = 0
            for i in range(repeat):
                ttr += self.calculate_ttr(self._words_count(self._random_choice(word_count,
                                                                              num)))
            ttrs.append(ttr / repeat)
        return nums, ttrs

    def visual(self):
        """
        可视化ttr随单词数目变化情况
        """
        fig = go.Figure([go.Scatter(x=self.nums1, y=self.ttrs1, name="Average" + self._path1.split(r".")[0]),
                         go.Scatter(x=self.nums1, y=[sum(self.ttrs1) / len(self.ttrs1)] * len(self.nums1)
                                    , name="Average" + self._path1.split(r".")[0]),
                         go.Scatter(x=self.nums1, y=self.ttrs2, name="Average" + self._path2.split(r".")[0]),
                         go.Scatter(x=self.nums1, y=[sum(self.ttrs2) / len(self.ttrs2)] * len(self.nums1)
                                    , name="Average" + self._path2.split(r".")[0])])
        fig.update_layout(xaxis_title="words_num", yaxis_title="ttr",
                          title="ttr随单词数目变化情况")
        fig.show()

    def calculate_ttr(self, words):
        return len(words) / sum([item[1] for item in words])

    def ttrs_detail(self):
        return {
            "path1_detail": self.ttrs1,
            "path1_average": sum(self.ttrs1) / len(self.ttrs1),
            "path2_detail": self.ttrs2,
            "path2_average": sum(self.ttrs2) / len(self.ttrs2),
        }
