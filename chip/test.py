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
from chip.chip_dataset import get_loader
from chip.config import CONFIG
from bert_serving.client import BertClient
import numpy as np
import torch


def test_entry():
    from chip.bidaf_model import BiDAFModel
    from config import FLAGS
    model = BiDAFModel().to(FLAGS.device)
    model.load_state_dict(torch.load(CONFIG.checkpiont_file))
    model.eval()
    bc = BertClient()
    data = get_loader(CONFIG.test_file)
    losses = []
    accs = []
    f1s = []
    loss_func = torch.nn.CrossEntropyLoss()

    for text1, text2, label in data:
        text1 = torch.tensor(bc.encode(list(text1))).to(FLAGS.device)
        text2 = torch.tensor(bc.encode(list(text2))).to(FLAGS.device)
        label = label.to(FLAGS.device).long()

        preds = model(text1, text2)
        loss = loss_func(preds, label)
        acc = 1 - torch.sum(torch.abs(torch.argmax(preds, dim=1) - label)).float() / label.shape[0]
        losses.append(loss.item())
        accs.append(acc.item())

    print(f'test\t'
          f'loss:\t{np.mean(losses):.4f},\t'
          f'acc:\t{np.mean(accs):.4f}')


def eval_entry():
    pass
