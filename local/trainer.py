import time
import shutil
import torch
import torch.nn.functional as F

from local import optimizer

def train(train_loader, model, optim, epoch, device, log_interval):
    batch_time = optimizer.AverageMeter()
    data_time = optimizer.AverageMeter()
    losses = optimizer.AverageMeter()
    top1 = optimizer.AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (_, input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # Create vaiables
        input  = input.to(device, non_blocking=True)
        #input  = input.to(device)
        target = target.to(device, non_blocking=True).view((-1,))
        #target = target.to(device).view((-1,))

        # compute output
        output = model(input)

        # loss 
        loss = F.nll_loss(output, target)

        # measure accuracy and record loss
        acc1, = optimizer.accuracy(output, target, topk=(1, ))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))

        # compute gradient and do SGD step
        optim.zero_grad()
        loss.backward()
        optim.step()
        lr = optim.update_learning_rate()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_interval == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'LR {lr:.6f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, lr=lr, loss=losses, top1=top1))

