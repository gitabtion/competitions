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
from absl import app


def main(_):
    from config import FLAGS
    if FLAGS.competition == 'chip':
        if FLAGS.mode == 'data':
            from chip.preproc import preproc_entry
            preproc_entry()
        if FLAGS.mode == 'train':
            from chip.train import train_entry
            train_entry()
        if FLAGS.mode == 'test':
            from chip.test import test_entry
            test_entry()
        if FLAGS.mode == 'eval':
            from chip.test import eval_entry
            eval_entry()


if __name__ == '__main__':
    app.run(main)
