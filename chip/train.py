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
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from bert_serving.client import BertClient

from chip.chip_dataset import get_loader
from chip.config import CONFIG
from chip.test import test_entry


def train(bc, model, dataset, optimizer):
    from config import FLAGS
    model = model.to(FLAGS.device)
    model.train()
    losses = []
    accs = []
    f1s = []
    loss_func = torch.nn.CrossEntropyLoss()

    # for text1, text2, label in tqdm(dataset):
    for text1, text2, label in dataset:
        optimizer.zero_grad()
        text1 = torch.tensor(bc.encode(list(text1))).to(FLAGS.device)
        text2 = torch.tensor(bc.encode(list(text2))).to(FLAGS.device)
        label = label.to(FLAGS.device).long()
        preds = model(text1, text2)
        loss = loss_func(preds, label)
        acc = 1 - torch.sum(torch.abs(torch.argmax(preds, dim=1) - label)).float() / label.shape[0]
        print(f'{torch.sum(label).item()}/{label.shape[0]}, loss: {loss.item():.4f}, acc: {acc.item():.4f}')
        losses.append(loss.item())
        accs.append(acc.item())

        loss.backward()
        optimizer.step()

    print(f'train\t'
          f'loss:\t{np.mean(losses):.4f},\t'
          f'acc:\t{np.mean(accs):.4f}')

    torch.save(model.state_dict(), CONFIG.checkpiont_file)


def train_entry():
    from chip.bidaf_model import BiDAFModel
    model = BiDAFModel()
    if CONFIG.model_from_file:
        model.load_state_dict(torch.load(CONFIG.checkpiont_file))
    model.train()
    bc = BertClient()
    optimizer = torch.optim.Adamax(model.parameters(), lr=CONFIG.lr)
    data = get_loader(CONFIG.train_file)
    for e in tqdm(range(CONFIG.epochs)):
        train(bc, model, data, optimizer)
        test_entry()
