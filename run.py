import argparse
import os
import random
import time
from pathlib import Path
# from config import opt

import numpy as np
import torch.distributed as dist
import torch.utils.data
import yaml
from yaml.loader import SafeLoader
from torch.utils.tensorboard import SummaryWriter

from utils.general import (
    get_latest_run, check_git_status, check_file, increment_dir, print_mutation, plot_evolution)
from utils.torch_utils import select_device
from train.train import train

with open('params.yaml') as f:
    opt = yaml.load(f, SafeLoader)

if __name__ == '__main__':

    # Resume
    if opt['resume']:
        last = get_latest_run() if opt['resume'] == 'get_last' else opt['resume']  # resume from most recent run
        if last and not opt['weights']:
            print(f'Resuming training from {last}')
        opt['weights'] = last if opt['resume'] and not opt['weights'] else opt['weights']
    if opt['local_rank'] == -1 or ("RANK" in os.environ and os.environ["RANK"] == "0"):
        check_git_status()

    opt['hyp'] = opt['hyp'] or ('data/hyp.finetune.yaml' if opt['weights'] else 'data/hyp.scratch.yaml')
    opt['data'], opt['cfg'], opt['hyp'] = check_file(opt['data']), check_file(opt['cfg']), check_file(opt['hyp'])  # check files
    assert len(opt['cfg']) or len(opt['weights']), 'either --cfg or --weights must be specified'

    opt['img_size'].extend([opt['img_size'][-1]] * (2 - len(opt['img_size'])))  # extend to 2 sizes (train, test)
    device = select_device(opt['device'], batch_size=opt['batch_size'])
    opt['total_batch_size'] = opt['batch_size']
    opt['world_size'] = 1
    opt['global_rank'] = -1

    # DDP mode
    if opt['local_rank'] != -1:
        assert torch.cuda.device_count() > opt['local_rank']
        torch.cuda.set_device(opt['local_rank'])
        device = torch.device('cuda', opt['local_rank'])
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        opt['world_size'] = dist.get_world_size()
        opt['global_rank'] = dist.get_rank()
        assert opt['batch_size'] % opt['world_size'] == 0, '--batch-size must be multiple of CUDA device count'
        opt['batch_size'] = opt['total_batch_size'] // opt['world_size']

    print(opt)
    with open(opt['hyp']) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps

    # Train
    if not opt['evolve']:
        tb_writer = None
        if opt['global_rank'] in [-1, 0]:
            print('Start Tensorboard with "tensorboard --logdir %s", view at http://localhost:6006/' % opt['logdir'])
            tb_writer = SummaryWriter(log_dir=increment_dir(Path(opt['logdir']) / 'exp', opt['name']))  # runs/exp

        train(hyp, opt, device, tb_writer)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'momentum': (0.1, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'giou': (1, 0.02, 0.2),  # GIoU loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (1, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (0, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (1, 0.0, 1.0),  # image flip left-right (probability)
                'mixup': (1, 0.0, 1.0)}  # image mixup (probability)

        assert opt['local_rank'] == -1, 'DDP mode not implemented for --evolve'
        opt['notest'], opt['nosave'] = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path('runs/evolve/hyp_evolved.yaml')  # save best result here
        if opt['bucket']:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt['bucket'])  # download evolve.txt if exists

        for _ in range(100):  # generations to evolve
            if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.9, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device)

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, opt['bucket'])

        # Plot results
        plot_evolution(yaml_file)
        print('Hyperparameter evolution complete. Best results saved as: %s\nCommand to train a new model with these '
              'hyperparameters: $ python train.py --hyp %s' % (yaml_file, yaml_file))
