# NLP Course-Tasks
Tasks of NLP Course in Nov.2020
## 朴素贝叶斯和文本分类
### Introduction to dataset
实验中使用的数据集是`苏轼`和`杜甫`的诗词，具体内容查看[https://github.com/Chang-LeHung/NLP-Tasks/tree/main/datasets/poems](https://github.com/Chang-LeHung/NLP-Tasks/tree/main/datasets/poems)中的dufu.json和sushi.json
实验中使用的数据集已经提取出来了，是`final set.npz`中的数据，具体加载和提取数据方式如下

```python
datasets = np.load("final set.npz")
datasets.allow_pickle = True
my_mapping = datasets["use_word2idx"]
dataset = datasets["dataset"]

dufu = torch.from_numpy(dataset.item()["dufu"])
sushi = torch.from_numpy(dataset.item()["sushi"])
word2idx = datasets["word2idx"]
idx2word = datasets["idx2word"]
```
