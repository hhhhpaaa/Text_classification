import torch
from torch import nn
import numpy as np


class NetWork(nn.Module):

    def __init__(self, total_words, embedding_len, units, num_class):

        super(NetWork, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=total_words, embedding_dim=embedding_len)
        self.lstm1 = nn.LSTM(input_size=embedding_len, hidden_size=units, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=units, hidden_size=128, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_class)

    def forward(self, x):

        x = self.embedding(x)
        x, (_, _) = self.lstm1(x)
        _, (h, _) = self.lstm2(x)
        x = self.dropout(self.fc1(h))
        x = self.fc2(nn.functional.relu(x)).type(torch.float64)

        return x


if __name__ == '__main__':

    total_words = 151304 + 3
    embedding_len = 512
    units = 256
    num_class = 9

    input_ = torch.randint(high=10000, size=(16, 200))
    label_ = torch.tensor(np.random.randint(9, size=16), dtype=torch.long)

    print("input.shape", input_.shape)
    print("label_.shape", label_.shape)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = NetWork(total_words=total_words, embedding_len=embedding_len, units=units, num_class=num_class)
    criterion = nn.CrossEntropyLoss()

    output_ = model(input_)
    loss = criterion(output_.squeeze(dim=0), label_)

    print("output_.shape", output_.shape)
    print("loss", loss)