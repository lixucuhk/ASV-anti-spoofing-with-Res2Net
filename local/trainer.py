import time
import shutil
import torch
import torch.nn.functional as F

def train(train_loader, model, optimizer, epoch, device, log_interval):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

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
        acc1, = accuracy(output, target, topk=(1, ))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr = optimizer.update_learning_rate()

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


class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count+1e-10)


def accuracy(output, target, topk=(1,)):
    """ Computes the accuracy over the k top predictions for the specified values of k """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

