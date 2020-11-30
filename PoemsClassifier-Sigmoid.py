


class PoemClassifier(nn.Module)
    
    def __init__(self, words_num, embedding_size, hidden_size, classes, num_layers,
                    batch_size, sequence_length)
        super(PoemClassifier, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.words_num = words_num
        self.sequence_length = sequence_length
        self.emb = nn.Embedding(words_num, embedding_size)
        self.LSTM = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden=None)
        batch_size, sequence_length = x.shape # x batch_size, sequence_length
        if hidden is None
            h, c = self.init_hidden(x, batch_size)
        else
            h. c = hidden
        out = self.emb(x) # batch_size, sequence_length, embedding_size
        out, hidden = self.LSTM(out, (h, c)) # batch_size, sequence_length, hidden_size
        out = out[, -1, ]# batch_size, last sequence, hidden_size
        out = self.fc1(out)
        out = self.sigmoid(out)
        return out, hidden

    def init_hidden(self, ipt, batch_size)
        h = ipt.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        c = ipt.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        h = Variable(h)
        c = Variable(c)
        return (h, c)
