#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import string
import os
import random
import shutil
import time
import warnings

import torchvision
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


# ONLINE LOGGING
import wandb


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')


#########
# WANDB #
#########
parser.add_argument('--notes', type=str, default='', help='wandb notes')
default_id = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(32)])
parser.add_argument('--name', type=str, default=default_id, help='wandb id/name')
parser.add_argument('--id', type=str, default=default_id, help='wandb id/name')
parser.add_argument('--wandbproj', type=str, default='autoself', help='wandb project name')

parser.add_argument('--checkpoint-interval', default=50, type=int,
                    help='how often to checkpoint')
parser.add_argument('--checkpoint_fp', type=str, default='checkpoints/', help='where to store checkpoint')


parser.add_argument('--dataid', help='id of dataset', default="cifar10", choices=('cifar10', 'imagenet'))

parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=30., type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--randomcrop', action='store_true',
                    help='use the random crop instead of randomresized crop, for FAA augmentations')

parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--mlp', action='store_true',
                    help='train a 2 layer mlp instead of a linear layer')
parser.add_argument('--task', default='classify',
                    help='which task to train', choices=("classify", "rotation"))

parser.add_argument('--kfold', default=None, type=int, 
                    help = "which fold we're looking at")

best_acc1 = 0




def main():
    args = parser.parse_args()

    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    # CIFAR 10 mod

    if args.dataid =="cifar10":
    # use the layer the SIMCLR authors used for cifar10 input conv, checked all padding/strides too.
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1,1), padding=(1,1), bias=False)
        model.maxpool = nn.Identity()
        model.fc = torch.nn.Linear(model.fc.in_features, 10) # note this is for cifar 10.

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False



    # Initialize the weights and biases in the way they did in the paper.
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    # hack the mlp into the final layer    
    if args.mlp:
        print('training mlp final layer')
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, model.fc.in_features), model.fc)
        model.fc[0].weight.data.normal_(mean=0.0, std=0.01)
        model.fc[0].bias.data.zero_()

    # load from pre-trained, before DistributedDataParallel constructor
    wandb_resume = args.resume
    name = args.id
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            
            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            only_encoder = 'encoder' in state_dict
            if only_encoder:
                state_dict = state_dict['encoder']
            if checkpoint.get('id'):
                # sync the ids for wandb
                args.id = checkpoint['id']
                name = args.id
                wandb_resume = True
            if checkpoint.get('name'):
                name = checkpoint['name']
            
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if only_encoder:
                    state_dict[k[len("module."):]] = state_dict[k]
                elif k.startswith('module.model.encoder'):
                    # remove prefix
                    state_dict[k[len("module.model.encoder."):]] = state_dict[k]
                elif k.startswith('module.encoder_q'):
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
                
            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            
            
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias
    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code


    # Chanigng this for CIFAR10.

    if args.dataid =="cifar10":
        _CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        normalize = transforms.Normalize(mean=_CIFAR_MEAN, std=_CIFAR_STD)


    #  Original normalization
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    # Readded some data augmentations for training this part.

    if args.dataid == "cifar10":
        crop_size = 28
        orig_size = 32
    else:
        orig_size = 256
        crop_size = 224

    if not args.randomcrop:
        crop_transform = transforms.RandomResizedCrop(crop_size)
    else:
        crop_transform = transforms.RandomCrop(32, padding=4)
        crop_size=32

    train_dataset = torchvision.datasets.CIFAR10(args.data,
        transform= transforms.Compose([
            crop_transform,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), download=False)


    val_transform = transforms.Compose([
            transforms.Resize(orig_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
            ])

    if args.kfold == None: 
        val_dataset = torchvision.datasets.CIFAR10(args.data, transform=val_transform,
            download=True, train=False)

    else: 
        # use the held out train data as the validation data. 
        val_dataset = torchvision.datasets.CIFAR10(args.data,
            transform= val_transform, download=True)


    if not args.kfold == None: 
        torch.manual_seed(1337)
        print('before: K FOLD', args.kfold, len(train_dataset))
        lengths = [len(train_dataset)//5]*5
        print(lengths)
        folds = torch.utils.data.random_split(train_dataset, lengths)
        folds.pop(args.kfold)
        train_dataset = torch.utils.data.ConcatDataset(folds)

        # Get the validation split
        print('pre split val', val_dataset)
        torch.manual_seed(1337)
        lengths = [len(val_dataset)//5]*5
        folds = torch.utils.data.random_split(val_dataset, lengths)
        val_dataset = folds[args.kfold]
        print('len val', len(val_dataset))

    else: 
        print("NO KFOLD ARG", args.kfold)


    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    # CR: only the master will report to wandb for now
    is_main_node = not args.multiprocessing_distributed or args.gpu == 0
    if is_main_node:
        # use lcls prefix so we don't overwrite the training args
        wandb_args = {"lcls_{}".format(key): val for key, val in args.__dict__.items()}
        wandb.init(project=args.wandbproj,
                   name=name,
                   id=args.id, resume=wandb_resume,
                   config=wandb_args, job_type='linclass')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args, is_main_node)
        return
    print("Doing task: {}".format(args.task))
    for epoch in range(args.start_epoch, args.epochs):

        print(epoch)
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, is_main_node, args.id[:5])

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)
        if is_main_node:
            wandb.log({"val-{}".format(args.task): acc1})

        # remember best acc@1 and save checkpoint
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.gpu == 0):
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            if is_best:

                if args.task == "classify": 

                    savefile = os.path.join(args.checkpoint_fp, "{}_lincls_best.tar".format(args.id[:5]))
                elif args.task == "rotation": 
                    savefile = os.path.join(args.checkpoint_fp, "{}_lincls_best_rotation.tar".format(args.id[:5]))
                torch.save({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, savefile)
                wandb.save(savefile)

def train(train_loader, model, criterion, optimizer, epoch, args, is_main_node=False, runid=""):
    batch_time = AverageMeter('LinCls Time', ':6.3f')
    data_time = AverageMeter('LinCls Data', ':6.3f')
    rot_losses = AverageMeter('Rot Val Loss', ':.4e')
    losses = AverageMeter('LinCls Loss', ':.4e')
    top1 = AverageMeter('LinCls Acc@1', ':6.2f')
    top5 = AverageMeter('LinCls Acc@5', ':6.2f')
    progress = ProgressMeter(
        is_main_node,
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, rot_losses],
        prefix="{} LinClass Epoch: [{}]".format(runid, epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        if args.task=="rotation":
            rotated_images, target = rotate_images(images)
            output = model(rotated_images)
            loss = criterion(output, target)
            rot_losses.update(loss.item(), images.size(0))            
        else:
            target = target.cuda(args.gpu, non_blocking=True)
            # compute output
            output = model(images)
            loss = criterion(output, target)
            losses.update(loss.item(), images.size(0))
            
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args, is_main_node=False):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Val Loss', ':.4e')
    rot_losses = AverageMeter('Rot Val Loss', ':.4e')
    top1 = AverageMeter('Val Acc@1', ':6.2f')
    top5 = AverageMeter('Val Acc@5', ':6.2f')
    progress = ProgressMeter(
        is_main_node,
        len(val_loader),
        [batch_time, losses, top1, top5, rot_losses],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            # compute output
         
            else:
                target = target.cuda(args.gpu, non_blocking=True)
                output = model(images)
                loss = criterion(output, target)
                losses.update(loss.item(), images.size(0))

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

def rotate_images(images):
    nimages = images.shape[0]
    n_rot_images = 4*nimages

    # rotate images all 4 ways at once
    rotated_images = torch.zeros([n_rot_images, images.shape[1], images.shape[2], images.shape[3]]).cuda()
    rot_classes = torch.zeros([n_rot_images]).long().cuda()

    rotated_images[:nimages] = images
    # rotate 90
    rotated_images[nimages:2*nimages] = images.flip(3).transpose(2,3)
    rot_classes[nimages:2*nimages] = 1
    # rotate 180
    rotated_images[2*nimages:3*nimages] = images.flip(3).flip(2)
    rot_classes[2*nimages:3*nimages] = 2
    # rotate 270
    rotated_images[3*nimages:4*nimages] = images.transpose(2,3).flip(3)
    rot_classes[3*nimages:4*nimages] = 3

    return rotated_images, rot_classes


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'module.encoder_q.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.encoder_q.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, main_node, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.main_node = main_node

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        if self.main_node:
            wandb.log({meter.name: meter.avg for meter in self.meters})

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


if __name__ == '__main__':
    main()
