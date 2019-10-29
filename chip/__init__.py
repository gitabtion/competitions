"""
              ┏┓      ┏┓ + +
             ┏┛┻━━━━━━┛┻┓ + +
             ┃          ┃
             ┃    ━     ┃ ++ + + +
            ██████━██████ +
             ┃          ┃ +
             ┃    ┻     ┃
             ┃          ┃ + +
             ┗━┓      ┏━┛
               ┃      ┃
               ┃      ┃ + + + +
               ┃      ┃   
               ┃      ┃ + 　　　　神兽保佑,loss->0
               ┃      ┃        
               ┃      ┃  +
               ┃      ┗━━━━━┓ + +
               ┃            ┣┓
               ┃            ┏┛
               ┗━┓┓┏━━━━┳┓┏━┛ + + + +
                 ┃┫┫    ┃┫┫
                 ┗┻┛    ┗┻┛ + + + +

    author: abtion
    email: abtion@outlook.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from chip.config import CONFIG
from torch.utils.data import Dataset
import pandas as pd


class BiDAFModel(nn.Module):
    def __init__(self):
        super(BiDAFModel, self).__init__()
        self.linear1 = nn.Linear(CONFIG.bert_size * 4, CONFIG.hidden_size)
        self.trans_layer = nn.TransformerEncoderLayer(CONFIG.hidden_size, 8)
        self.trans_encoder = nn.TransformerEncoder(self.trans_layer, 2)
        self.linear2 = nn.Linear(CONFIG.hidden_size, 2)

    def forward(self, C, Q):
        # C(batch_size,bert_size), Q(batch_size,bert_size)
        _C = C.unsqueeze(1)
        _Q = Q.unsqueeze(2)
        S = _C * _Q  # size(batch_size, bert_size, bert_size)
        S_1 = torch.softmax(S, dim=1)
        S_2 = torch.softmax(S.transpose(1, 2), dim=1)
        A = torch.matmul(S_1, Q.unsqueeze(2)).view(-1, CONFIG.bert_size)
        B = torch.matmul(S_2, C.unsqueeze(2)).view(-1, CONFIG.bert_size)

        X = torch.cat([C, Q, A, B], dim=1).view(-1, CONFIG.bert_size * 4)
        X = self.linear1(X)
        X = F.gelu(X)
        X = X.unsqueeze(0)
        X = self.trans_encoder(X).squeeze()
        X = self.linear2(X)
        return X


class ChipDataset(Dataset):
    def __init__(self, file):
        data = pd.read_csv(file)
        self.text1 = list(data.iloc[:, 0])
        self.text2 = list(data.iloc[:, 1])
        self.label = list(data.iloc[:, 2])
        self.num = len(self.label)

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        return self.text1[index], self.text2[index], self.label[index]


def get_loader(file):
    dataset = ChipDataset(file)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              CONFIG.batch_size,
                                              shuffle=True,
                                              num_workers=4)
    return data_loader
