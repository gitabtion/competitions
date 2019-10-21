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
import os

import torch


class CONFIG:
    work_dir = 'chip'
    hidden_size = 256
    bert_size = 768
    train_file = os.path.join(work_dir, 'data', 'train.csv')
    test_file = os.path.join(work_dir, 'data', 'test.csv')
    eval_file = os.path.join(work_dir, 'data', 'eval_original.csv')
    submit_file = os.path.join(work_dir, 'data', 'result.csv')
    batch_size = 256
    lr = 1e-3
    weight_decay = 0.1
    epochs = 100
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    checkpiont_file = os.path.join(work_dir, 'checkpoint', 'model.ckpt')
    model_from_file = os.path.exists(checkpiont_file)
    # model_from_file = False
    testset_len = 2048
