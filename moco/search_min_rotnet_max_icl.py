# CODE is based of fastautoagument code here 

# https://github.com/kakaobrain/fast-autoaugment/blob/master/FastAutoAugment/search.py


import torch 
import torch.nn as nn
import ray 

import ray
from ray import tune
from ray.tune import track

from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch

from ray.tune import register_trainable, run_experiments
import wandb
import argparse
import torchvision.models as models
import sys

sys.path.append("/userdata/smetzger/all_deepul_files/deepul_proj/moco/")
import moco.loader
import moco.builder

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
import os
import ray.tune as tune



# FOR DEBUG
class Args:
    checkpoints = ['fxrZE', 'lJu2W', 'rdEIg', 'esdq2' ,'vnhKs'] # Ordered KFOLDS order. Make this nicer.
    checkpoint_fp = '/userdata/smetzger/all_deepul_files/ckpts'
    data = '/userdata/smetzger/data/cifar_10/'
    
    # Some args for the Fast Autoaugment thing. 
    num_op = 2
    num_policy=5
    num_search = 200
    dataid = 'cifar10'
    cv_ratio=1.0
    smoke_test=False
    resume=False
    arch = 'resnet50'
    distributed=False
    loss = 'icl_and_rotation'# one of rotation, supervised, icl, icl_and_rotation.
    base = 'moco' # Name for what we are saving our training runs as.

    # Moco args. 
    moco_k = 65536
    moco_m = 0.999
    moco_t = 0.2
    
    
    # Whether or not to use the MLP for mocov2
    mlp = True
    
    # Model input args for building the model head. 
    nomoco = False
    rotnet = False
    
    moco_dim = 128
    policy_dir = '/userdata/smetzger/all_deepul_files/policies'
    
    # Remember we are trying to max negative loss, so a negative here
    # is like maximizing, a positive here is like minimizing. 
    loss_weights = {'rotation': 1, 'icl': -1, 'supervised':1} # weight for ICL, and ROTATION. Divide by mean.

    
args=Args()
print('args', args)

import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from torchvision.transforms.transforms import Compose
import torchvision
import torchvision.transforms as transforms
random_mirror = True
from self_aug.autoaug_scripts import augment_list, Augmentation, Accumulator

# Define how we load our dataloaders. 
_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

def get_dataloaders(augmentations, batch=1024, kfold=0, loss_type='icl', get_train=False):

    """
    input: augmentations: the list of the augmentations you want applied to the data. 
    batch = batchsize, 
    kfold, which fold you want to look at (0, 1,2 3, or 4)
    get_train, whether or not you want the train data. Use this when loading the data to train linear classifiers, 
    slash when you're loading the final classifier. 
    """
    if args.dataid == "imagenet":
        train_dataset = datasets.ImageFolder(
            traindir,
            transformations)

    elif args.dataid == "cifar10":

        # THe default training transforms we use when training the CIFAR10 network. 
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(28, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
        

        # If we're evaluating ICL, its only fair to do so with the ICL augmentations
        if loss_type == "icl": 
            
            random_resized_crop = transforms.RandomResizedCrop(28, scale=(0.2, 1.))
            
            transform_train = transforms.Compose([
            random_resized_crop,
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
            ])


        # Insert the new transforms in to the training transforms. 
        transform_train.transforms.insert(0, Augmentation(augmentations))
        
        # Use the twocrops transform. 
        if loss_type == "icl": 
            transform_train = moco.loader.TwoCropsTransform(transform_train)

    else:
        raise NotImplementedError("Support for the following dataset is not yet implemented: {}".format(args.dataid))

    if get_train: 
        train_dataset = torchvision.datasets.CIFAR10(args.data,
                                                     transform=transform_train,
                                                     download=True)

    # NOTE THAT IN THE FAA PAPER THE USED TRANSFORM TRAIN.  
    
    
    val_dataset = torchvision.datasets.CIFAR10(args.data, transform=transform_train, 
        download=True)
    

    if get_train: 
        torch.manual_seed(1337)
        lengths = [len(train_dataset)//5]*5
        folds = torch.utils.data.random_split(train_dataset, lengths)
        folds.pop(kfold)
        train_dataset = torch.utils.data.ConcatDataset(folds)


    torch.manual_seed(1337)
    lengths = [len(val_dataset)//5]*5
    folds = torch.utils.data.random_split(val_dataset, lengths)
    val_dataset = folds[kfold]

    if get_train: 
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch, shuffle=(train_sampler is None),
            num_workers=8, pin_memory=True, sampler=train_sampler, drop_last=True)

        
    if loss_type == 'icl': 
        sampler =None
        drop_last=False
    else: 
        sampler =None
        drop_last=False

    val_loader= torch.utils.data.DataLoader(
        val_dataset, batch_size=batch, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=drop_last, 
        sampler=sampler
    )

    if not get_train: 
        train_loader = None

    return train_loader, val_loader

# Take in the augment from hyperopt and return some augmentations, in teh way that we want them. 
def policy_decoder(augment, num_policy, num_op):
    op_list = augment_list(False)
    policies = []
    for i in range(num_policy):
        ops = []
        for j in range(num_op):
            op_idx = augment['policy_%d_%d' % (i, j)]
            op_prob = augment['prob_%d_%d' % (i, j)]
            op_level = augment['level_%d_%d' % (i, j)]
            ops.append((op_list[op_idx][0].__name__, op_prob, op_level))
        policies.append(ops)
    return policies

def find_model(name, fold, epochs=750, basepath="/userdata/smetzger/all_deepul_files/ckpts"):
    """
    name = model name
    fold = which fold of the data to find. 
    epochs = how many epochs to load the checkpoint at (e.g. 750)
    
    """
    for file in os.listdir(basepath):
        if name in str(file) and 'fold_%d' %fold in str(file):
            if str(file).endswith(str(epochs-1) + '.tar'): 
                return os.path.join(basepath, file)
            
    print("COULDNT FIND MODEL")
    assert True==False # just throw an error. 

    
    

def load_model(cv_fold, loss_type): 
    
    print("HELLO")
    model = models.__dict__[args.arch]()
    # CIFAR 10 model
    
    if args.dataid =="cifar10":
    # use the layer the SIMCLR authors used for cifar10 input conv, checked all padding/strides too.
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1,1), padding=(1,1), bias=False)
        model.maxpool = nn.Identity()

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init the fc layer
    if args.dataid == "cifar10":

        if loss_type == 'supervised':
            model.fc = torch.nn.Linear(model.fc.in_features, 10) # note this is for cifar 10.
        elif loss_type =='rotation': 
            model.fc = torch.nn.Linear(model.fc.in_features, 4)


    if loss_type == 'supervised': 
        savefile = os.path.join(args.checkpoint_fp, 
                                "{}_lincls_best.tar".format(args.checkpoints[cv_fold]))
    elif loss_type == 'rotation': 
        savefile = os.path.join(args.checkpoint_fp, 
                                 "{}_lincls_best_rotation.tar".format(args.checkpoints[cv_fold]))

    elif loss_type == 'icl' or loss_type == 'icl_and_rotation': 
#         print('ICL')
        heads = {}
        if not args.nomoco:
            heads["moco"] = {
            "num_classes": args.moco_dim
        }
        
#         print(heads)

        model = moco.builder.MoCo(
            models.__dict__[args.arch],
            K=args.moco_k, m=args.moco_m, T=args.moco_t, mlp=args.mlp, dataid=args.dataid,
            multitask_heads=heads
        )
        savefile = find_model(args.checkpoints[cv_fold], cv_fold)
        
#     print('savefile', savefile)
    ckpt = torch.load(savefile, map_location="cpu")
    
    state_dict = ckpt['state_dict']
    
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module'):
            state_dict[k[len("module."):]] = state_dict[k] 
            del state_dict[k]

                
    model.load_state_dict(state_dict)
    return model


    
m = load_model(0, args.loss)    

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

def eval_augmentations(config): 
    augment = config
    print('called', augment)
    
    if args.loss == 'icl_and_rotation':
        
        losses = ['icl', 'rotation']
        
    else: 
        losses = [args.loss]
        
    
    metrics = Accumulator()
    
    for loss_type in losses: 
        # TODO MOve this out
#         metrics = Accumulator()
      
        print(loss_type)
        
        augmentations = policy_decoder(augment, augment['num_policy'], augment['num_op'])
        
        
        fold = augment['cv_fold']
        model = load_model(cv_fold, loss_type).cuda()
        model.eval()
        loaders = []

        for _ in range(args.num_policy): #TODO: 
            _, validloader = get_dataloaders(augmentations, 512, kfold=fold, loss_type=loss_type)
            loaders.append(iter(validloader))
            del _

     
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

        try: 

            i = 0
            with torch.no_grad(): 
                while True: 
                    losses = []
                    corrects = []

                    for loader in loaders:

                        if not loss_type == 'icl':

                            data, label = next(loader)
                            data = data.cuda()
                            label = label.cuda()

                            if loss_type == 'supervised':
                                pred = model(data)

                            if loss_type =="rotation":
                                rotated_images, label = rotate_images(data)
                                pred = model(rotated_images)  

                        else: 

                            images, _ = next(loader)
                            images[0] = images[0].cuda(non_blocking=True)
                            images[1] = images[1].cuda(non_blocking=True)
                            pred, label =model(head="moco", im_q=images[0], im_k=images[1], evaluate=True)
                            acc = accuracy(pred, label)

                        loss = loss_fn(pred, label)
                        losses.append(loss.detach().cpu().numpy())

                        _, pred = pred.topk(1, 1, True, True)
                        pred = pred.t()
                        correct = pred.eq(label.view(1, -1).expand_as(pred)).detach().cpu().numpy()
                        corrects.append(correct)

                        if not loss_type == 'icl':
                            del loss, correct, pred, data, label
                        else: 
                            del loss, images, pred, label, correct



                    losses = np.concatenate(losses)
#                     print('losses shape' , losses.shape)
                    losses_min = np.mean(losses) # get it so it averages out.
                    corrects = np.concatenate(corrects)
                    corrects_max = np.max(corrects, axis=0).squeeze()
#                     print('len corrects max', len(corrects_max), 'corrects shape', corrects.shape)
                    losses_min *= losses.shape[0]/5
                    
                    if loss_type == 'rotation': 
                        metrics.add_dict({ 
                            'minus_loss': -.25*np.sum(losses_min)*args.loss_weights[loss_type],
                            'plus_loss': np.sum(losses_min)*args.loss_weights[loss_type],
                            'correct': np.sum(corrects_max)*args.loss_weights[loss_type],
                            'cnt': len(corrects_max)})
                        del corrects, corrects_max
                    else: 
                        metrics.add_dict({ 
                            'minus_loss': -1*np.sum(losses_min)*args.loss_weights[loss_type],
                            'plus_loss': np.sum(losses_min)*args.loss_weights[loss_type],
                            'correct': np.sum(corrects_max)*args.loss_weights[loss_type],
                            'cnt': len(corrects_max)})
                        del corrects, corrects_max

#                     print(metrics['minus_loss']/metrics['cnt'])
        except StopIteration: 
            pass
    
    del model
    metrics = metrics/'cnt'
    tune.track.log(top_1_valid=metrics['correct'], minus_loss=metrics['minus_loss'], plus_loss=metrics['plus_loss'])
    print(metrics['correct'])
    return metrics['minus_loss']

ops = augment_list(False) # Get the default augmentation set. 
# Define the space of our augmentations. 
space = {}
for i in range(args.num_policy): 
    for j in range(args.num_op):
        space['policy_%d_%d' %(i,j)]  = hp.choice('policy_%d_%d' %(i, j), list(range(0, len(ops))))
        space['prob_%d_%d' %(i, j)] = hp.uniform('prob_%d_%d' %(i, j), 0.0, 1.0)
        space['level_%d_%d' %(i, j)] = hp.uniform('level_%d_%d' %(i, j), 0.0, 1.0)

final_policy_set = []

# if not args.loss == 'icl': 
#     reward_attr = 'minus_loss'
# else: 
#     reward_attr = 'minus_loss'
    
reward_attr = 'top_1_valid'

# TODO: let this be whatever we want. 
object_store_memory = int(0.6 * ray.utils.get_system_memory() // 10 ** 9 * 10 ** 9)

# TODO Change back
ray.init(num_gpus=4, ignore_reinit_error=True, 
    num_cpus=28
    )
# ray.init(num_gpus=1, memory=200*1024*1024*100, object_store_memory=200*1024*1024*50)
import ray
from ray import tune

cv_num = 5
num_result_per_cv = 10

for _ in range(2): 
    for cv_fold in range(cv_num): 
        name = "slm_moco_min_max_top1valid_%s_fold_%d" %(args.dataid, cv_fold)
        hyperopt_search=HyperOptSearch(space, 
            max_concurrent=4,
            metric=reward_attr,
            mode='max')


        results = tune.run(
            eval_augmentations, 
            name=name,
            num_samples=200,
            resources_per_trial={
                "gpu": 1
            },
            search_alg=hyperopt_search,
            verbose=2,
            config = { 
                'num_op': args.num_op, 
                'num_policy': args.num_policy, 
                'cv_fold': cv_fold
            },
            return_trials=True,
            stop={'training_iteration': 1},
        )
        results_copy = results
        results = [x for x in results if x.last_result is not None]
        results = sorted(results, key= lambda x: x.last_result[reward_attr], reverse=True)

        for result in results[:num_result_per_cv]: 
            final_policy = policy_decoder(result.config, args.num_policy, args.num_op)
            final_policy_set.extend(final_policy)

        print(final_policy)
print(final_policy_set)

# Start saving to a path called policies. 
import pickle
savefp = os.path.join(args.policy_dir, str(args.base+'_'+args.loss+ '_' + (str(reward_attr) + '.pkl')))
with open(savefp, 'wb') as f: 
    pickle.dump(final_policy_set, f)