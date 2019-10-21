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
