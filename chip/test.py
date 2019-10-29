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
from chip import get_loader
from chip.config import CONFIG
from bert_serving.client import BertClient
import numpy as np
import torch


def test_entry():
    from chip import BiDAFModel
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
    from chip import BiDAFModel
    from config import FLAGS
    import pandas as pd
    model = BiDAFModel().to(FLAGS.device)
    model.load_state_dict(torch.load(CONFIG.checkpiont_file))
    model.eval()
    bc = BertClient()

    data = pd.read_csv(CONFIG.eval_file)
    text1s = [_data for _data in list(data.iloc[:, 1])]
    text2s = [_data for _data in list(data.iloc[:, 2])]
    encoded_text1s = torch.tensor(bc.encode(text1s)).to(FLAGS.device)
    encoded_text2s = torch.tensor(bc.encode(text2s)).to(FLAGS.device)
    batch_size = 500
    num = len(text1s) // batch_size
    preds = model(encoded_text1s[:batch_size], encoded_text2s[:batch_size])
    labels = torch.argmax(preds, dim=1).cpu().data.numpy()
    for i in range(1, num):
        _preds = model(encoded_text1s[i * batch_size:(i + 1) * batch_size],
                       encoded_text2s[i * batch_size:(i + 1) * batch_size])
        _labels = torch.argmax(preds, dim=1).cpu().data.numpy()
        labels = np.append(labels, _labels)
    print(labels.shape[0])
    data = data.drop(['question1', 'question2', 'category'], axis=1)
    data = data.astype(int)
    data.to_csv(CONFIG.submit_file, index=False)
    print("eval done!")
