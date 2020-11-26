# NLP Course-Tasks
Tasks of NLP Course in Nov.2020
## 朴素贝叶斯和文本分类
### Introduction to dataset
实验中使用的数据集是`苏轼`和`杜甫`的诗词，具体内容查看[https://github.com/Chang-LeHung/NLP-Tasks/tree/main/datasets/poems](https://github.com/Chang-LeHung/NLP-Tasks/tree/main/datasets/poems)中的dufu.json和sushi.json
实验中使用的数据集已经提取出来了，是`final set.npz`中的数据，具体加载和提取数据方式如下

```python
import numpy as np
datasets = np.load("final set.npz")
datasets.allow_pickle = True
my_mapping = datasets["use_word2idx"]
dataset = datasets["dataset"]
dufu = dataset.item()["dufu"]
sushi = dataset.item()["sushi"]
word2idx = datasets["word2idx"]
idx2word = datasets["idx2word"]
```
`杜甫`的诗词一共1150首
```python
dataset.item()["dufu"]
array([[   0,    0,    0, ..., 4024, 3507, 4682],
       [   0,    0,    0, ..., 1200, 3507, 4682],
       [   0,    0,    0, ..., 1834, 3507, 4682],
       ...,
       [   0,    0,    0, ..., 2815, 3507, 4682],
       [   0,    0,    0, ..., 2699, 3507, 4682],
       [   0,    0,    0, ..., 3168, 3507, 4682]])
```
`苏轼`的诗词一共1150首
```python
dataset.item()["sushi"]
array([[   0,    0,    0, ..., 2456, 3507, 4682],
       [   0,    0,    0, ..., 1447, 3507, 4682],
       [   0,    0,    0, ..., 3166, 3507, 4682],
       ...,
       [   0,    0,    0, ..., 3124, 3507, 4682],
       [   0,    0,    0, ...,  216, 3507, 4682],
       [   0,    0,    0, ..., 4316, 3507, 4682]])
```
