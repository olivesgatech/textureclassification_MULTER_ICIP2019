##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import print_function
import os
from tqdm import tqdm

import torch
import torch.nn as nn

from option import Options
from utils import get_transform, MINCDataset
from deepTEN import get_deepten
from lr_scheduler import LR_Scheduler
from files import save_checkpoint

from tensorboardX import SummaryWriter

from datetime import datetime

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
if not os.path.exists('logs/'):
    os.mkdir('logs/')

log_path = os.path.join('logs', ts)
os.makedirs(log_path)
writer = SummaryWriter(log_path)

# global variable
best_pred = 0.0
acclist_train = []
acclist_val = []


def main():
    # init the args
    global best_pred, acclist_train, acclist_val
    args = Options().parse()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # init datasets and dataloaders
    transform_train, transform_val = get_transform(args.dataset)
    trainset = MINCDataset(root='./data/minc-2500', split='train', transform=transform_train)
    valset = MINCDataset(root='./data/minc-2500',  split='val', transform=transform_val)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)
    
    # init the model
    model = get_deepten(dataset='minc', backbone='resnet50', pretrained=args.pretrained, root='models/')
    print(model)

    # criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.cuda:
        model.cuda()
        criterion.cuda()
        # Please use CUDA_VISIBLE_DEVICES to control the number of gpus
        model = nn.DataParallel(model)
    # check point
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] +1
            best_pred = checkpoint['best_pred']
            acclist_train = checkpoint['acclist_train']
            acclist_val = checkpoint['acclist_val']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            raise RuntimeError("=> no resume checkpoint found at '{}'".\
                format(args.resume))
    scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(train_loader), args.lr_step)
    def train(epoch):
        model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        global best_pred, acclist_train
        tbar = tqdm(train_loader, desc='\r')
        for batch_idx, (data, target) in enumerate(tbar):
            scheduler(optimizer, batch_idx, epoch, best_pred)
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))
            losses.update(loss.item(), data.size(0))
            tbar.set_description('\rLoss: %.3f | Top1: %.3f' % (losses.avg, top1.avg))
            niter = (epoch - 1) * len(train_loader) + batch_idx
            writer.add_scalar('Train/Loss', losses.avg, niter)
            writer.add_scalar('Train/Prec@1', top1.avg, niter)
            writer.add_scalar('Train/Pred@5', top5.avg, niter)

        acclist_train += [top1.avg]

    def validate(epoch):
        model.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        global best_pred, acclist_train, acclist_val
        is_best = False
        tbar = tqdm(val_loader, desc='\r')
        for batch_idx, (data, target) in enumerate(tbar):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = model(data)
                loss = criterion(output, target)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], data.size(0))
                top5.update(acc5[0], data.size(0))
                losses.update(loss.item(), data.size(0))

            tbar.set_description('Top1: %.3f | Top5: %.3f' % (top1.avg, top5.avg))
            niter = (epoch - 1) * len(val_loader) + batch_idx
            writer.add_scalar('Val/Loss', losses.avg, niter)
            writer.add_scalar('Val/Prec@1', top1.avg, niter)
            writer.add_scalar('Val/Pred@5', top5.avg, niter)

        if args.eval:
            print('Top1 Acc: %.3f | Top5 Acc: %.3f '%(top1.avg, top5.avg))
            return
        # save checkpoint
        acclist_val += [top1.avg]
        if top1.avg > best_pred:
            best_pred = top1.avg 
            is_best = True
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_pred': best_pred,
            'acclist_train': acclist_train,
            'acclist_val': acclist_val,
            }, args=args, is_best=is_best)

    if args.eval:
        validate(args.start_epoch)
        return

    for epoch in range(args.start_epoch, args.epochs + 1):
        train(epoch)
        validate(epoch)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
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


class AverageMeter(object):
    """Computes and stores the average and current value"""
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
        self.avg = self.sum / self.count

if __name__ == "__main__":
    main()
