# Tasks of NLP Course in Autumn 2020

* [Naive Bayes & Text Classification](#Naive-Bayes-and-Text-Classification)
* [LSTM for Sequence Labeling](#LSTM-for-Sequence-Labeling)
* [Word2vec & Embedding](#Word2vec-Embedding)
## Naive Bayes and Text Classification
### Introduction to dataset
实验中使用的数据集是`苏轼`和`杜甫`的诗词，具体内容查看[https://github.com/Chang-LeHung/NLP-Tasks/tree/main/datasets/poems](https://github.com/Chang-LeHung/NLP-Tasks/tree/main/datasets/poems)中的 `dufu.json` 和 `sushi.json` 这里面还有一些其他的数据，可以用来做多分类，比如分类多个诗人的诗词，分类写人、写景的诗词等等，实验为了方便
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
`my_mapping` 可以将一个数据数组变成相应的诗词，<START>和<END>分别作为诗词的开始和结束。
```python
my_mapping[dataset.item()["sushi"]]
array([['<START>', '<START>', '<START>', ..., '去', '。', '<END>'],
       ['<START>', '<START>', '<START>', ..., '思', '。', '<END>'],
       ['<START>', '<START>', '<START>', ..., '工', '。', '<END>'],
       ...,
       ['<START>', '<START>', '<START>', ..., '谗', '。', '<END>'],
       ['<START>', '<START>', '<START>', ..., '夸', '。', '<END>'],
       ['<START>', '<START>', '<START>', ..., '章', '。', '<END>']],
      dtype='<U7')
```
```python
my_mapping[dataset.item()["sushi"][100]][-26:] 
# 将苏轼的第100首诗拿出来， <START>前面都是<START>作为填充使用，因为诗词长度不一致所以需要填充
array(['<START>', '雨', '细', '方', '梅', '夏', '，', '风', '高', '已', '麦', '秋',
       '。', '应', '怜', '百', '花', '尽', '，', '绿', '叶', '暗', '红', '榴', '。',
       '<END>'], dtype='<U7')
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
### Classify with LSTM 
        
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
分割测试集和训练集，训练集的比例是0.9
```python
dataset_train, dataset_test, label_train, label_test = train_test_split(final_dataset,
                                                            final_label, test_size=0.1,
                                                                       random_state=1)
```
构建数据集，重写`Dataset`里面的`__len__`，`__getitem__`函数方便后数据集的构造
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

`input` : `batch_size, seqence_length`

`after embedding` : `batch_size`, `seqence_length`, `embedding_size`

`after LSTM` : `batch_size`, `seqence_length`, `hidden_size`

`return` : `batch_szie`, `1`, `hidden_size` == `batch_szie`, `hidden_size `

因为我们只需要返回最后一个节点的输出，用这个输出来做分类即可，因为这个节点的信息是由前面所有节点得来的，它包含前面所有的信息，所以 `reuturn` 的结果 `shape` 是 `batch_szie`, `1`, `hidden_size `

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

    def forward(self, x, hidden=None):
        batch_size, sequence_length = x.shape # x batch_size, sequence_length
        if hidden is None:
            h, c = self.init_hidden(x, batch_size)
        else:
            h, c = hidden
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
模型训练结果如下如所示:

`Softmax + CrossEntropy` on testset

<img src="/images/sof.png" width = "800"  alt="Softmax" align=center />

`Sigmoid + Binary Cross Entropy` on testset

<img src="/images/sig.png" width = "800"  alt="Sigmoid" align=center />

更多具体代码信息请参考[LSTM for text classification](https://github.com/Chang-LeHung/NLP-Tasks/blob/main/Text%20Classification/Big%20TaskI.ipynb) 
预训练模型 [pretrained model](https://github.com/Chang-LeHung/NLP-Tasks/blob/main/Text%20Classification/PoemClassify.pth)

关于朴素贝叶斯的工程实现比较简单，具体的代码可参考：[https://github.com/Chang-LeHung/NLP-Tasks/blob/main/Text%20Classification/NaiveBayes.ipynb](https://github.com/Chang-LeHung/NLP-Tasks/blob/main/Text%20Classification/NaiveBayes.ipynb)
## LSTM for Sequence Labeling
实验中所用的数据集太大不能传送到github，如有需要请访问[https://gitee.com/Chang-LeHung/NLP-Tasks](https://gitee.com/Chang-LeHung/NLP-Tasks)
## Word2vec-Embedding
目前实现词嵌入主要有两种主流的模型分别是：SkipGram模型和CBOW模型，。两种模型很相似，相当于是一种对称模型。其主要结构如下所示：

<img src="/images/skipgram.png" width = "800"  alt="Softmax" align=center />

<img src="/images/cbow.png" width = "800"  alt="Softmax" align=center />

上述两种模型只需要具体实现一个模型即可，另外一个模型很容易就能理解和实现了，下面主要介绍CBOW模型。
### 数据集准备
CBOW模型主要是通过上下文预测中心词，因此在采集数据的时候根据中心词来采集，通过设置窗口大小来决定上下文的范围。比如 "I like natural language processing very much"，如果中心词为
"language",上下文窗口为2，那么它的范围就是"like natural language processing very"，从这中间采集上下文，切中心词左右采集的单词数目要一致。具体实现如下：
```python
def dataset_extractor(article, left_window=8, right_window=8, num_words_selected=8):
    '''
    article : 文章对应的数字列表
    num_words_selected : 每个中心词需要抽取多少个对应的上下文单词
    left_window : 中心词左边带抽取单词的范围
    right_window : 中心词右边边带抽取单词的范围
    '''
    length = len(article)
    dataset = []
    labels = []
    for idx, word_num in enumerate(article[1:-1]):
        idx = idx + 1
        if idx <= left_window:
            temp = np.array([idx - 1, idx + 1])
            num_idx = list(np.arange(0, idx - 1)) + list(temp) + list(np.arange(idx + 2, idx + 1 + right_window))
        elif length - idx -1 <= right_window:
            temp = np.array([idx - 1, idx + 1])
            num_idx = list(np.arange(idx - left_window, idx - 1)) + list(temp) + list(np.arange(idx + 2, length))
        else:
            temp = np.array([idx - 1, idx + 1])
            num_idx = list(np.arange(idx - left_window, idx - 1)) + list(temp) + list(np.arange(idx + 2, idx + 1 + right_window))
        nums = article[num_idx]
        if len(nums) > num_words_selected:
            nums = np.random.choice(nums, size=num_words_selected, replace=False)
        labels += [article[idx].tolist()]
        dataset += nums.tolist()
    return labels, dataset
# 获取数据
labels, dataset = dataset_extractor(article=num_article, left_window=8, right_window=8,
                                               num_words_selected=8)
dataset = torch.from_numpy(np.array(dataset))
labels = torch.from_numpy(np.array(labels))
# 构建数据模型
class MyDataset(Dataset):
    
    def __init__(self, data, label):
        super(MyDataset, self).__init__()
        self.data = data
        self.label = label
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
# 得到训练迭代数据
dataset = MyDataset(dataset, labels)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True, num_workers=4)
```

### 网络结构设计
CBOW模型网络结构其实很简单，就是一个简单的三层前馈网络模型，输入通过词嵌入在通过全连接层再加上一层softmax即可。具体代码如下：
```python
class CBOW(nn.Module):
    
    def __init__(self, embedding_size, vocab_size, squence_length):
        
        super(CBOW, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.squence_length = squence_length
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size * squence_length, vocab_size)
        self.logosoftmax = nn.LogSoftmax(dim=1)
   
    def forward(self, x):
        out = self.embedding(x)
        out = out.view(-1, self.squence_length * self.embedding_size)
        out = self.linear(out)
        return self.logosoftmax(out)
```
其实模型组件还是比较简单的，只需要简单修改一下数据的维度即可。之前SkipGram模型是一个词对应一个词，现在是一组词对应一个词。假如原始的输入维度为(batch size, context length)，经过embedding层之后的维度变成，(batchm size, context length, embedding size)，所以在经过全连接层之前需要进行Reshape过程，在经过$Reshape$过程之后的维度变成(batch size, context length * embedding size)，再将这个数据经过全连接层即可，注意：logosoftmax + NLLoss = sotfmax + crossentropy。
