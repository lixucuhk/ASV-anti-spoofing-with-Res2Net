import numpy as np
import os
import random
import shutil
import time
import warnings
from collections import defaultdict
from functools import reduce
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from model import Detector
from data_reader.dataset_v1 import SpoofDatsetSystemID

from local import datafiles, prediction, trainer, validate

import argparse

def main(pretrained, data_files, model_params, training_params, device):
    """ forward pass dev and eval data to trained model  """
    batch_size = training_params['batch_size']
    test_batch_size = training_params['test_batch_size']
    epochs = training_params['epochs']
    start_epoch = training_params['start_epoch']
    n_warmup_steps = training_params['n_warmup_steps']
    log_interval = training_params['log_interval']

    kwargs = {'num_workers': 4, 'pin_memory': True} if device == torch.device('cuda') else {}

    # create model
    model = Detector(**model_params).to(device) 
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)    
    print('===> Model total parameter: {}'.format(num_params))

    if device == torch.device('cuda') and torch.cuda.device_count() > 1:
        print('multi-gpu')
        model = nn.DataParallel(model).cuda()
    
    if pretrained:
        epoch_id = pretrained.split('/')[2].split('_')[0]
        pretrained_id = pretrained.split('/')[1]
        if os.path.isfile(pretrained):
            print("===> loading checkpoint '{}'".format(pretrained))
            checkpoint = torch.load(pretrained, map_location=lambda storage, loc: storage) # load for cpu
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            print("===> loaded checkpoint '{}' (epoch {})"
                  .format(pretrained, checkpoint['epoch']))
        else:
            print("===> no checkpoint found at '{}'".format(pretrained))
            exit()
    else: raise NameError 

    # Data loading code (class analysis for multi-class classification only)
    val_data    = SpoofDatsetSystemID(data_files['dev_scp'], data_files['dev_utt2index'], binary_class=False)
    eval_data = SpoofDatsetSystemID(data_files['eval_scp'], data_files['eval_utt2index'], binary_class=False)

    val_loader  = torch.utils.data.DataLoader(
        val_data, batch_size=test_batch_size, shuffle=False, **kwargs)
    eval_loader = torch.utils.data.DataLoader(
        eval_data, batch_size=test_batch_size, shuffle=False, **kwargs)

    os.makedirs(data_files['scoring_dir'], exist_ok=True)
    # forward pass for dev
    print("===> forward pass for dev set")
    score_file_pth = os.path.join(data_files['scoring_dir'], str(pretrained_id) + '-epoch%s-dev_scores.txt' %(epoch_id))
    print("===> dev scoring file saved at: '{}'".format(score_file_pth))
    prediction.prediction(val_loader, model, device, score_file_pth, data_files['dev_utt2systemID'])

    # forward pass for eval
    print("===> forward pass for eval set")
    score_file_pth = os.path.join(data_files['scoring_dir'], str(pretrained_id) + '-epoch%s-eval_scores.txt' %(epoch_id))
    print("===> eval scoring file saved at: '{}'".format(score_file_pth))
    prediction.prediction(eval_loader, model, device, score_file_pth, data_files['eval_utt2systemID'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-feats', action='store', type=str, default='pa_spec')
    parser.add_argument('--pretrained', action='store', type=str, default=None)
    parser.add_argument('--configfile', action='store', type=str)
    parser.add_argument('--random-seed', action='store', type=int, default=0)
    args = parser.parse_args()

    pretrained = args.pretrained
    random_seed = args.random_seed

    with open(args.configfile) as json_file:
        config = json.load(json_file)

    data_files = datafiles.data_prepare[args.data_feats]
    model_params = config['model_params']
    training_params = config['training_params']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)

    ''' 
    print(run_id)
    print(pretrained)
    print(data_files)
    print(model_params)
    print(training_params)
    print(device)
    exit(0)
    '''
    main(pretrained, data_files, model_params, training_params, device)


