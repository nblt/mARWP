# train sgd random weighted noise

import argparse
from torch.nn.modules.batchnorm import _BatchNorm
import os
import time
import numpy as np
import random
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


from utils import *

class CosineInc:
    def __init__(self, std: float, num_epochs:int, steps_per_epoch: int, inc: int):
        self.base = std
        self.halfwavelength_steps = num_epochs * steps_per_epoch
        self.inc = inc

    def __call__(self, step):
        scale_factor = -np.cos(step * np.pi / self.halfwavelength_steps) * 0.5 + 0.5
        self.current = self.base * (scale_factor * self.inc + 1)
        return self.current

# Parse arguments
parser = argparse.ArgumentParser(description='Regular SGD training')
parser.add_argument('--EXP', metavar='EXP', help='experiment name', default='SGD')
parser.add_argument('--arch', '-a', metavar='ARCH',
                    help='The architecture of the model')
parser.add_argument('--datasets', metavar='DATASETS', default='CIFAR10', type=str,
                    help='The training datasets')
parser.add_argument('--optimizer',  metavar='OPTIMIZER', default='SGD', type=str,
                    help='The optimizer for training')
parser.add_argument('--schedule',  metavar='SCHEDULE', default='step', type=str,
                    help='The schedule for training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 50 iterations)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--wandb', dest='wandb', action='store_true',
                    help='use wandb to monitor statisitcs')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--log-dir', dest='log_dir',
                    help='The directory used to save the log',
                    default='save_temp', type=str)
parser.add_argument('--log-name', dest='log_name',
                    help='The log file name',
                    default='log', type=str)
parser.add_argument('--randomseed', 
                    help='Randomseed for training and initialization',
                    type=int, default=1)
parser.add_argument('--std',  default=0.01, type=float,
                    metavar='STD', help='std for RWP')
parser.add_argument('--rho',  default=0.20, type=float,
                    metavar='RHO', help='rho for SAM')
parser.add_argument('--eta',  default=1, type=float,
                    metavar='ETA', help='eta for RWP')
parser.add_argument('--cutout', dest='cutout', action='store_true',
                    help='use cutout data augmentation')
parser.add_argument('--randaug', dest='randaug', action='store_true',
                    help='use randaug data augmentation')
parser.add_argument('--inc', default=15, type=int, metavar='N',
                    help='the times lpf sigma enlarges')
parser.add_argument('--times', default=1, type=int, metavar='N',
                    help='the times for generating random noise')
parser.add_argument('--beta',  default=0.9, type=float,
                    metavar='beta', help='beta for RFRWP')
            
best_prec1 = 0

# Record training statistics
train_loss = []
train_err = []
test_loss = []
test_err = []
arr_time = []

p0 = None

args = parser.parse_args()

if args.wandb:
    import wandb
    wandb.init(project="TWA", entity="nblt")
    date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
    wandb.run.name = args.EXP + date


def get_model_param_vec(model):
    # Return the model parameters as a vector

    vec = []
    for name,param in model.named_parameters():
        vec.append(param.data.detach().reshape(-1))
    return torch.cat(vec, 0)


def get_model_grad_vec(model):
    # Return the model gradient as a vector

    vec = []
    for name,param in model.named_parameters():
        vec.append(param.grad.detach().reshape(-1))
    return torch.cat(vec, 0)

def update_grad(model, grad_vec):
    idx = 0
    for name,param in model.named_parameters():
        arr_shape = param.grad.shape
        size = param.grad.numel()
        param.grad.data = grad_vec[idx:idx+size].reshape(arr_shape).clone()
        idx += size

def update_param(model, param_vec):
    idx = 0
    for name,param in model.named_parameters():
        arr_shape = param.data.shape
        size = param.data.numel()
        param.data = param_vec[idx:idx+size].reshape(arr_shape).clone()
        idx += size
        
def print_param_shape(model):
    for name,param in model.named_parameters():
        print (name, param.data.shape)

def _cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def get_cosine_annealing_scheduler(optimizer, epochs, steps_per_epoch, base_lr):
    lr_min = 0.0
    total_steps = epochs * steps_per_epoch

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_annealing(
            step,
            total_steps,
            1,  # since lr_lambda computes multiplicative factor
            lr_min / base_lr))

    return scheduler        

def main():

    global args, best_prec1, p0
    global train_loss, train_err, test_loss, test_err, arr_time
    global std, std_scheduler
    
    set_seed(args.randomseed)

    # Check the save_dir exists or not
    print ('save dir:', args.save_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Check the log_dir exists or not
    print ('log dir:', args.log_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    sys.stdout = Logger(os.path.join(args.log_dir, args.log_name))

    # Define model
    # model = torch.nn.DataParallel(get_model(args))
    model = get_model(args)
    model.cuda()
    
    # for name, param in model.named_parameters():
    #     print (name, param.shape)
    # while True: pass
    print_param_shape(model)
    
    # Optionally resume from a checkpoint
    if args.resume:
        # if os.path.isfile(args.resume):
        if os.path.isfile(os.path.join(args.save_dir, args.resume)):
            
            # model.load_state_dict(torch.load(os.path.join(args.save_dir, args.resume)))

            print ("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            print ('from ', args.start_epoch)
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print ("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print ("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Prepare Dataloader
    print ('cutout:', args.cutout)
    if args.cutout:
        train_loader, val_loader = get_datasets_cutout(args)
    elif args.randaug:
        train_loader, val_loader = get_datasets_randaug(args)
    else:
        train_loader, val_loader = get_datasets(args)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    if args.optimizer == 'ARWP':
        base_optimizer = torch.optim.SGD
        optimizer = ARWP(model.parameters(), base_optimizer, std=args.std, eta=args.eta, beta=args.beta, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    elif args.optimizer == 'ARWP_adam':
        base_optimizer = torch.optim.Adam
        optimizer = ARWP(model.parameters(), base_optimizer, std=args.std, eta=args.eta, beta=args.beta, lr=args.lr, weight_decay=args.weight_decay)

    if args.schedule == 'step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=args.start_epoch - 1)
    elif args.schedule == 'cosine':
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        lr_scheduler = get_cosine_annealing_scheduler(optimizer, args.epochs, len(train_loader), args.lr)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    is_best = 0
    print ('Start training: ', args.start_epoch, '->', args.epochs)

    # DLDR sampling
    torch.save(model.state_dict(), os.path.join(args.save_dir,  str(0) +  '.pt'))

    print ('std:', args.std)
    
    std_scheduler = CosineInc(args.std, args.epochs, len(train_loader), args.inc)
    print ('len(train_loader):', len(train_loader))
    optimizer.std = std_scheduler(0)

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        print ('current std:', optimizer.std)
        train(train_loader, model, criterion, optimizer, lr_scheduler, epoch)
        
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        
        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'model.th'))

    print ('train loss: ', train_loss)
    print ('train err: ', train_err)
    print ('test loss: ', test_loss)
    print ('test err: ', test_err)

    print ('time: ', arr_time)

    
    prec1 = validate(train_loader, model, criterion)

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

running_grad = 0
current_step = 0
g0, g1 = None, None

def train(train_loader, model, criterion, optimizer, lr_scheduler, epoch):
    """
    Run one train epoch
    """
    global train_loss, train_err, arr_time, p0, sample_idx, index
    global running_grad, std, std_scheduler, current_step, g0, g1
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()    

    total_loss, total_err = 0, 0
    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()
        
        optimizer.first_step(zero_grad=True)
        
        output = model(input_var)
        loss = criterion(output, target_var)
        loss.mean().backward()
        optimizer.second_step(zero_grad=True)

        total_loss += loss.item() * input_var.shape[0]
        total_err += (output.max(dim=1)[1] != target_var).sum().item()

        output = output.float()
        loss = loss.float()

        current_step += 1
        optimizer.std = std_scheduler(current_step)
        lr_scheduler.step()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        if i % args.print_freq == 0 or i == len(train_loader) - 1:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))

    print ('Total time for epoch [{0}] : {1:.3f}'.format(epoch, batch_time.sum))

    train_loss.append(total_loss / len(train_loader.dataset))
    train_err.append(total_err / len(train_loader.dataset)) 
    print ('train loss | acc', total_loss / len(train_loader.dataset),  1 - total_err / len(train_loader.dataset))
    # print ('train loss before | post: ', total_loss / len(train_loader.dataset), post_total_loss / len(train_loader.dataset))
    # print ('train acc before | post: ', 1 - total_err / len(train_loader.dataset), 1 - post_total_err / len(train_loader.dataset))
    
    if args.wandb:
        wandb.log({"train loss": total_loss / len(train_loader.dataset)})
        wandb.log({"train acc": 1 - total_err / len(train_loader.dataset)})
    
    arr_time.append(batch_time.sum)

def validate(val_loader, model, criterion, add=True):
    """
    Run evaluation
    """
    global test_err, test_loss

    total_loss = 0
    total_err = 0

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()
                
            total_loss += loss.item() * input_var.shape[0]
            total_err += (output.max(dim=1)[1] != target_var).sum().item()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and add:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    if add:
        print(' * Prec@1 {top1.avg:.3f}'
            .format(top1=top1))
    
        test_loss.append(total_loss / len(val_loader.dataset))
        test_err.append(total_err / len(val_loader.dataset))
        
        if args.wandb:
            wandb.log({"test loss": total_loss / len(val_loader.dataset)})
            wandb.log({"test acc": 1 - total_err / len(val_loader.dataset)})

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
