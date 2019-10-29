from absl import flags
import torch
import logging

FLAGS = flags.FLAGS

flags.DEFINE_string('competition', 'chip', 'chip/fake_news')
flags.DEFINE_string('device', "cuda:1" if torch.cuda.is_available() else "cpu", '训练所用设备')
flags.DEFINE_string('mode', 'train', 'data/train/test/eval')

log = logging
