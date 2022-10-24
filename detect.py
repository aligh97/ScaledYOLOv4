import torch

from utils.general import (strip_optimizer)
from train.inference import detect
import yaml
from yaml.loader import SafeLoader

with open('params.yaml') as f:
    opt = yaml.load(f, SafeLoader)

if __name__ == '__main__':

    with torch.no_grad():
        if opt['update']:  # update all models (to fix SourceChangeWarning)
            for opt['weights'] in ['']:
                detect(opt)
                strip_optimizer(opt['weights'])
        else:
            detect(opt)
