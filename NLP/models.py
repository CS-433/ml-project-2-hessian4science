import torch.nn as nn
class SentimentBaselineModel(nn.Module):

    def __init__(self, features):
        super(SentimentAnalysisModel, self).__init__()
        self.layers = []
        size= features
        self.layers.append(nn.Linear(size, size//2))
        self.layers.append(nn.ReLU())
        for i in range(1,7):

            self.layers.append(nn.Linear(size//((i)*2), size//((i + 1)*2)))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(0.25))
        
        self.layers.append(nn.Linear(size//((i+1)*2), 3))
        self.linear = nn.Sequential(*self.layers)
        self.linear.apply(self.init_weights)


    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, text):
    
        out = self.linear(text.float())
        return out
    


    
class VanillaLSTM(nn.Module):
    def __init__(self, 
                 embedding_dim,
                 hidden_dim,
                 num_layers, 
                 dropout_rate=0.2):
                
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # YOUR CODE HERE
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, input_id):
        # YOUR CODE HERE
        output, _ = self.lstm(input_id.float())
        output = self.fc(output)
        return output