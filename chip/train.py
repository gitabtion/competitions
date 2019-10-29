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
import math

import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from bert_serving.client import BertClient
import torch.optim as optim
from chip import get_loader
from chip.config import CONFIG
from chip.test import test_entry


def train(bc, model, dataset, optimizer, scheduler):
    device_ids = [1, 2, 3, 0]

    from config import FLAGS
    model = model.to(FLAGS.device)
    model.train()
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    optimizer = torch.nn.DataParallel(optimizer, device_ids=device_ids)
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
        # print(f'{torch.sum(label).item()}/{label.shape[0]}, loss: {loss.item():.4f}, acc: {acc.item():.4f}')
        losses.append(loss.item())
        accs.append(acc.item())
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), CONFIG.grad_clip)
        optimizer.module.step()

        scheduler.step()

    print(f'train\t'
          f'loss:\t{np.mean(losses):.4f},\t'
          f'acc:\t{np.mean(accs):.4f}')

    torch.save(model.module.state_dict(), CONFIG.checkpiont_file)


def train_entry():
    from chip import BiDAFModel
    model = BiDAFModel()
    if CONFIG.model_from_file:
        model.load_state_dict(torch.load(CONFIG.checkpiont_file))
    model.train()
    bc = BertClient()
    base_lr = 1
    lr_warm_up_num = 100
    lr = CONFIG.lr
    parameters = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = optim.Adamax(lr=base_lr, betas=(0.9, 0.999), eps=1e-7, weight_decay=5e-8, params=parameters)
    cr = lr / math.log2(lr_warm_up_num)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ee: cr * math.log2(ee + 1) if ee < lr_warm_up_num else lr)
    data = get_loader(CONFIG.train_file)
    for e in tqdm(range(CONFIG.epochs)):
        train(bc, model, data, optimizer, scheduler)
        test_entry()
