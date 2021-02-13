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

from local import datafiles, trainer, validate
from local.optimizer import ScheduledOptim

import argparse

def main(run_id, pretrained, data_files, model_params, training_params, device):
    best_acc1 = 0
    batch_size = training_params['batch_size']
    test_batch_size = training_params['test_batch_size']
    epochs = training_params['epochs']
    start_epoch = training_params['start_epoch']
    n_warmup_steps = training_params['n_warmup_steps']
    log_interval = training_params['log_interval']

    # model is trained for binary classification (for datalaoder) 
    if model_params['NUM_SPOOF_CLASS'] == 2: 
        binary_class = True 
    else: binary_class = False 

    kwargs = {'num_workers': 2, 'pin_memory': True} if device == torch.device('cuda') else {}
    
    # create model
    model = Detector(**model_params).to(device) 
    num_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('===> Model total parameter: {}'.format(num_model_params))
    
    # Wrap model for multi-GPUs, if necessary
    if device == torch.device('cuda') and torch.cuda.device_count() > 1:
        print('multi-gpu') 
        model = nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    optimizer = ScheduledOptim(
            torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, lr=3e-4, amsgrad=True),
        training_params['n_warmup_steps'])

    # optionally resume from a checkpoint
    if pretrained:
        if os.path.isfile(pretrained):
            print("===> loading checkpoint '{}'".format(pretrained))
            checkpoint = torch.load(pretrained)
            start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("===> loaded checkpoint '{}' (epoch {})".format(pretrained, checkpoint['epoch']))
        else:
            print("===> no checkpoint found at '{}'".format(pretrained))

    # Data loading code
    train_data = SpoofDatsetSystemID(data_files['train_scp'], data_files['train_utt2index'], binary_class)
    val_data   = SpoofDatsetSystemID(data_files['dev_scp'], data_files['dev_utt2index'], binary_class)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=test_batch_size, shuffle=True, **kwargs)

    best_epoch = 0
    early_stopping, max_patience = 0, 100 # for early stopping
    os.makedirs("model_snapshots/" + run_id, exist_ok=True) 
    for epoch in range(start_epoch, start_epoch+epochs):

        trainer.train(train_loader, model, optimizer, epoch, device, log_interval)
        acc1 = validate.validate(val_loader, data_files['dev_utt2systemID'], model, device, log_interval)    
        
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        

        # adjust learning rate + early stopping 
        if is_best:
            early_stopping = 0
            best_epoch = epoch + 1
        else:
            early_stopping += 1
            if epoch - best_epoch > 2:
                optimizer.increase_delta()
                best_epoch = epoch + 1
        if early_stopping == max_patience:
            break
        
        # save model
        trainer.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
            }, is_best,  "model_snapshots/" + str(run_id), str(epoch) + ('_%.3f'%acc1) + ".pth.tar")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', action='store', type=str, default='0')
    parser.add_argument('--data-feats', action='store', type=str, default='pa_spec')
    parser.add_argument('--pretrained', action='store', type=str, default=None)
    parser.add_argument('--configfile', action='store', type=str)
    parser.add_argument('--random-seed', action='store', type=int, default=0)
    args = parser.parse_args()

    run_id = args.run_id
    pretrained = args.pretrained
    random_seed = args.random_seed


    with open(args.configfile) as json_file:
        config = json.load(json_file)

    print(config)

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
    main(run_id, pretrained, data_files, model_params, training_params, device)

