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
from torch.utils.data import Dataset
import pandas as pd

from chip.config import CONFIG


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
