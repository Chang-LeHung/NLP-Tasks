# NLP Course-Tasks
Tasks of NLP Course in Nov.2020
## 朴素贝叶斯和文本分类
### Introduction to dataset
实验中使用的数据集是`苏轼`和`杜甫`的诗词，具体内容查看[https://github.com/Chang-LeHung/NLP-Tasks/tree/main/datasets/poems](https://github.com/Chang-LeHung/NLP-Tasks/tree/main/datasets/poems)中的dufu.json和sushi.json 这里面还有一些其他的数据，可以用来做多分类，比如分类多个诗人的诗词，分类写人、写景的诗词等等，实验为了方便
和朴素贝叶斯进行对比，只进行了二分类。

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
构造`label`，并将其变成tensor类型数据
```python
dufu = torch.from_numpy(dataset.item()["dufu"])
sushi = torch.from_numpy(dataset.item()["sushi"])

label_dufu = torch.full((len(dufu), ), fill_value=0)
label_sushi = torch.full((len(sushi), ), fill_value=1)

final_label = torch.cat((label_dufu, label_sushi), dim=0)
final_dataset = torch.cat((dufu, sushi), dim=0)

final_label = final_label.type(torch.LongTensor)
```
构建数据集，重写`Dataset`里面的__len__，__getitem__函数方便后数据集的构造
```python
class MyDataset(Dataset):
    
    def __init__(self, dataset, label):
        
        self.datasets = dataset
        self.labels = label

    def __getitem__(self, idx):
        return self.datasets[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)
```
构建网络模型，本次分类使用 `LSTM` 模型，实验过程中比较需要注意的就是数据在网络过程中传递的时候，`shape` 的变化, 代码中 `batch_first=True`，具体的 `shape` 变化如下：

`input` : batch_size, seqence_length

`after embedding` : `batch_size, seqence_length, embedding_size`

`after LSTM` : `batch_size, seqence_length, hidden_size`

`return` : `batch_szie, 1, hidden_size `

因为我们只需要返回最后一个节点的输出，用这个输出来做分类即可，因为这个节点的信息是由前面所有节点得来的，所以 `reuturn` 的结果 `shape` 是 `batch_szie, 1, hidden_size `

```python
class PoemClassifier(nn.Module):
    
    def __init__(self, words_num, embedding_size, hidden_size, classes, num_layers,
                    batch_size, sequence_length):
        super(PoemClassifier, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.words_num = words_num
        self.sequence_length = sequence_length
        self.emb = nn.Embedding(words_num, embedding_size)
        self.LSTM = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, hidden=None):
        batch_size, sequence_length = x.shape # x batch_size, sequence_length
        if hidden is None:
            h, c = self.init_hidden(x, batch_size)
        else:
            h. c = hidden
        out = self.emb(x) # batch_size, sequence_length, embedding_size
        out, hidden = self.LSTM(out, (h, c)) # batch_size, sequence_length, hidden_size
        out = out[:, -1, :]# batch_size, last sequence, hidden_size
        out = self.fc1(out)
        return out, hidden

    def init_hidden(self, ipt, batch_size):
        h = ipt.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        c = ipt.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        h = Variable(h)
        c = Variable(c)
        return (h, c)
```
更多具体信息请参考[https://github.com/Chang-LeHung/NLP-Tasks/blob/main/Text%20Classification/Big%20TaskI.ipynb](https://github.com/Chang-LeHung/NLP-Tasks/blob/main/Text%20Classification/Big%20TaskI.ipynb)
