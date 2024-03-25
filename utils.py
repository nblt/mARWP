import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models_imagenet
from models.vit import ViT

import numpy as np
import random
import os
import time
import models
import sys
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
import os.path
import pickle
from PIL import Image

import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required
import math

def set_seed(seed=1): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Logger(object):
    def __init__(self,fileN ="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN,"a")
 
    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
################################ datasets #######################################

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

class Cutout:
    def __init__(self, size=16, p=0.5):
        self.size = size
        self.half_size = size // 2
        self.p = p

    def __call__(self, image):
        if torch.rand([1]).item() > self.p:
            return image

        left = torch.randint(-self.half_size, image.size(1) - self.half_size, [1]).item()
        top = torch.randint(-self.half_size, image.size(2) - self.half_size, [1]).item()
        right = min(image.size(1), left + self.size)
        bottom = min(image.size(2), top + self.size)

        image[:, max(0, left): right, max(0, top): bottom] = 0
        return image

def get_datasets(args):
    if args.datasets == 'CIFAR10':
        print ('cifar10 dataset!')
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./datasets/', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./datasets/', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    elif args.datasets == 'CIFAR100':
        print ('cifar100 dataset!')
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./datasets/', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./datasets/', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    elif args.datasets == 'ImageNet':
        traindir = os.path.join('/opt/data/common/ILSVRC2012/', 'train')
        valdir = os.path.join('/opt/data/common/ILSVRC2012/', 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers)
    
    return train_loader, val_loader


def get_datasets_randaug(args):
    print ('randaug!')
    if args.datasets == 'CIFAR10':
        print ('cifar10 dataset!')
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.Resize(32),
                transforms.ToTensor(),
                normalize,
            ])
        
        N=2; M=14;
        transform_train.transforms.insert(0, RandAugment(N, M))
        transform_test = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                normalize,
            ])

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./datasets/', train=True, transform=transform_train, download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./datasets/', train=False, transform=transform_test),
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        
    elif args.datasets == 'CIFAR10-noise':
        print('cifar10 nosie dataset!')
        cifar10_noise = cifar_dataloader(dataset='cifar10', r=args.noise_ratio, noise_mode=args.noise_mode, 
                                          batch_size=args.batch_size, num_workers=args.workers, cutout=args.cutout)
        train_loader, val_loader = cifar10_noise.get_loader()
        
    elif args.datasets == 'CIFAR100':
        print ('cifar100 dataset!')
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        
        transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4), transforms.ToTensor(), normalize,
            ])
        N=2; M=14;
        transform_train.transforms.insert(0, RandAugment(N, M))
        transform_test = transforms.Compose([transforms.ToTensor(), normalize,
            ])

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./datasets/', train=True, transform=transform_train, download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./datasets/', train=False, transform=transform_test),
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    
    elif args.datasets == 'CIFAR100-noise':
        print('cifar100 nosie dataset!')
        cifar100_noise = cifar_dataloader(dataset='cifar100', r=args.noise_ratio, noise_mode=args.noise_mode, 
                                          batch_size=args.batch_size, num_workers=args.workers, cutout=args.cutout)
        train_loader, val_loader = cifar100_noise.get_loader()
    
    return train_loader, val_loader

def get_datasets_ddp(args):
    if args.datasets == 'CIFAR10':
        print ('cifar10 dataset!')
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        
        my_trainset = datasets.CIFAR10(root='./datasets/', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True)
                                       
        train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
        train_loader = torch.utils.data.DataLoader(my_trainset, batch_size=args.batch_size, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./datasets/', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    
    elif args.datasets == 'CIFAR100':
        print ('cifar100 dataset!')
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        
        my_trainset = datasets.CIFAR100(root='./datasets/', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True)
        train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
        train_loader = torch.utils.data.DataLoader(my_trainset, batch_size=args.batch_size, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./datasets/', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        
    return train_loader, val_loader

def get_datasets_cutout(args):
    print ('cutout!')
    if args.datasets == 'CIFAR10':
        print ('cifar10 dataset!')
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./datasets/', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
                Cutout()
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./datasets/', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    elif args.datasets == 'CIFAR100':
        print ('cifar100 dataset!')
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./datasets/', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
                Cutout()
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./datasets/', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    
    return train_loader, val_loader
    
def get_datasets_cutout_ddp(args):
    if args.datasets == 'CIFAR10':
        print ('cifar10 dataset!')
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        
        my_trainset = datasets.CIFAR10(root='./datasets/', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
                Cutout()
            ]), download=True)
        train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
        train_loader = torch.utils.data.DataLoader(my_trainset, batch_size=args.batch_size, sampler=train_sampler, drop_last=True, num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./datasets/', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    
    elif args.datasets == 'CIFAR100':
        print ('cifar100 dataset!')
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        
        my_trainset = datasets.CIFAR100(root='./datasets/', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
                Cutout()
            ]), download=True)
        train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
        train_loader = torch.utils.data.DataLoader(my_trainset, batch_size=args.batch_size, sampler=train_sampler, drop_last=True, num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./datasets/', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        
    return train_loader, val_loader

def get_model(args):
    print('Model: {}'.format(args.arch))

    if args.datasets == 'ImageNet':
        return models_imagenet.__dict__[args.arch]()

    if args.datasets == 'CIFAR10':
        num_classes = 10
    elif args.datasets == 'CIFAR100':
        num_classes = 100
    elif args.datasets == 'ImageNet':
        num_classes = 1000
        

    if args.datasets == 'ImageNet':
        return models_imagenet.__dict__[args.arch]()
    
    if args.arch == 'ViT':
        model = ViT(image_size = 32, patch_size = 4, num_classes = num_classes,
            dim = int(512), depth = 6, heads = 8,
            mlp_dim = 512, dropout = 0.1, emb_dropout = 0.1)
        return model
    
    model_cfg = getattr(models, args.arch)

    return model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)



class ARWP(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, std=0.01, eta=1, beta=0.9, **kwargs):
        assert std >= 0.0, f"Invalid std, should be non-negative: {std}"

        defaults = dict(std=std, **kwargs)
        super(ARWP, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.std = std
        self.eta = eta
        self.beta = beta
        print ('ARWP std:', self.std, 'eta:', self.eta, 'beta:', self.beta)

    @torch.no_grad()
    def first_step(self, zero_grad=True):
        for group in self.param_groups:

            for p in group["params"]:
                self.state[p]["old_p"] = p.data.clone()
                sh = p.data.shape
                sh_mul = int(np.prod(sh[1:]))

                fisher = None
                pd = p.data 
                if "old_g" in self.state[p]:
                    fisher = self.state[p]["old_g"].view(sh[0], -1).sum(dim=1, keepdim=True).repeat(1, sh_mul).view(sh)

                if len(p.data.shape) > 1:
                    e_w = pd.view(sh[0], -1).norm(dim=1, keepdim=True).repeat(1, sh_mul).view(sh)
                    e_w = torch.normal(0, (self.std + 1e-16) * e_w).to(p)
                else:
                    e_w = torch.empty_like(pd).to(p)
                    e_w.normal_(0, self.std * (pd.view(-1).norm().item() + 1e-16))
                    
                if fisher is not None:
                    # _norm = e_w.norm()
                    e_w /= torch.sqrt(1 + self.eta * fisher)
                    # e_w = e_w / e_w.norm() * _norm
                    # print (_norm, e_w.norm(), p.data.norm())

                    # print (torch.sqrt((self.eta * fisher)).max().item(), end=' ')

                p.add_(e_w)  # add weight noise

        if zero_grad: self.zero_grad()


    @torch.no_grad()
    def second_step(self, zero_grad=True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
                if "old_g" not in self.state[p]:
                    self.state[p]["old_g"] = p.grad.clone() ** 2
                else:
                    self.state[p]["old_g"] = self.state[p]["old_g"] * self.beta + p.grad.clone() ** 2

        self.base_optimizer.step()  # do the actual update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "RWP requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups        

