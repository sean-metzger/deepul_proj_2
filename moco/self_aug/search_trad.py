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


# print(os.environ["CUDA_VISIBLE_DEVICES"])
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

# What to use for our moco model. 

# argparser.add('--checkpoint_fp', type=str, help='base file path where everything is stored.')
# argparser.add('-checkpoints' , '--list', nargs='+', help='The list of the checkpoint codes, in order by cv fold')
# # example: search.py -checkpoints rdElg fxrZE IJu2W vnhKs esdq2

# argparser.add('--data', type=str, help="Where the data directory is")
# argparser.add('--dataid', type=str, default='cifar10', help="imagenet or cifar")
 
# # Arguments for FAA.  
# parser.add_argument('--num-op', type=int, default=2, help="number of operations per subpolicy.")
# parser.add_argument('--num-policy', type=int, default=5, help="number of subpolicies in each policy")
# parser.add_argument('--num-search', type=int, default=200, help="number of hyperopt iterations to run.")
# parser.add_argument('--smoke-test', action='store_true', help="quick test of our search")
# args = parser.parse_args()


# FOR DEBUG
class Args:
    checkpoints = ['fxrZE', 'lJu2W', 'rdEIg', 'esdq2' ,'vnhKs'] # Ordered KFOLDS order. Make this nicer.
    checkpoint_fp = '/userdata/smetzger/all_deepul_files/ckpts'
    data = '/userdata/smetzger/data/cifar_10/'
    num_op = 2
    num_policy=5
    num_search = 200
    dataid = 'cifar10'
    cv_ratio=1.0
    smoke_test=True
    resume=False
    arch = 'resnet50'
    distributed=False
    loss = 'rotation' # one of rotation, supervised, ICL, icl_and_rotation.
args=Args()


import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from torchvision.transforms.transforms import Compose
import torchvision
import torchvision.transforms as transforms
random_mirror = True
from autoaug_scripts import augment_list, Augmentation, Accumulator

# Define how we load our dataloaders. 
_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


def get_dataloaders(augmentations, batch=1024, kfold=0, get_train=False):

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

        # TODO: add imagenet transforms etc. 
    elif args.dataid == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(28, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])

        transform_train.transforms.insert(0, Augmentation(augmentations))

        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])

    else:
        raise NotImplementedError("Support for the following dataset is not yet implemented: {}".format(args.dataid))

    if get_train: 
        train_dataset = torchvision.datasets.CIFAR10(args.data,
                                                     transform=transform_train,
                                                     download=True)

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

    # if args.distributed:
    #     # if get_train: 
    #     #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    #     # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) #TODO: is this necessary? 
    #     val_sampler = None

    if get_train: 
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch, shuffle=(train_sampler is None),
            num_workers=8, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader= torch.utils.data.DataLoader(
        val_dataset, batch_size=batch, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False
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

def load_base_model(cv_fold): 
    
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            if checkpoint.get('id'):
                # sync the ids for wandb
                args.id = checkpoint['id']
                name = args.id
                wandb_resume = True
            if checkpoint.get('name'):
                name = checkpoint['name']

            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.model.encoder'):
                    # remove prefix
                    state_dict[k[len("module.model.encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

def load_model(cv_fold, loss_type): 
    model = models.__dict__[args.arch]()
    # CIFAR 10 mod
    
    
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
#         print('before change', model.fc)
        model.fc = torch.nn.Linear(model.fc.in_features, 10) # note this is for cifar 10.
#         print(model.fc)


    if loss_type == 'supervised': 
        savefile = os.path.join(args.checkpoint_fp, 
                                "{}_lincls_best.tar".format(args.checkpoints[cv_fold]))
    elif loss_type == 'rotation': 
        savefile = os.path.join(args.checkpoint_fp, 
                                 "{}_lincls_best_rotation.tar".format(args.checkpoints[cv_fold]))

    ckpt = torch.load(savefile, map_location="cpu")
    
    state_dict = ckpt['state_dict']
    
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        
        if k.startswith('module'):
            # remove prefix
            
            state_dict[k[len("module."):]] = state_dict[k]
            del state_dict[k]
            
    
    model.load_state_dict(state_dict)
    
    return model

    # Load the FC layer and append it to the end. 

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


def eval_augmentations(config): 
    augment = config
    print('called', augment)
    augmentations = policy_decoder(augment, augment['num_policy'], augment['num_op'])
    # Load the model from wandb. 
    fold = augment['cv_fold']
    ckpt = args.checkpoint_fp + 'fold_%d.tar' %(fold)

    model = load_model(cv_fold, args.loss).cuda()
    model.eval()
    loaders = []
#     TODO: Undo this
    for _ in range(args.num_policy): 
    # for _ in range(2):
        _, validloader = get_dataloaders(augmentations, 128, kfold=fold)
        loaders.append(iter(validloader))
        del _

    metrics = Accumulator()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    try: 
        with torch.no_grad(): 
            while True: 
                losses = []
                corrects = []

                for loader in loaders:
                    data, label = next(loader)
                    data = data.cuda()
                    label = label.cuda()

                    if args.loss == 'supervised':
                        pred = model(data)

                    if args.loss =="rotation":
                        rotated_images, label = rotate_images(data)
                        pred = model(rotated_images)   
                   

                    loss = loss_fn(pred, label)
                    losses.append(loss.detach().cpu().numpy())

                    _, pred = pred.topk(1, 1, True, True)
                    pred = pred.t()
                    correct = pred.eq(label.view(1, -1).expand_as(pred)).detach().cpu().numpy()
                    corrects.append(correct)

                    del loss, correct, pred, data, label

                losses = np.concatenate(losses)
                losses_min = np.min(losses, axis=0).squeeze()

                corrects = np.concatenate(corrects)
                corrects_max = np.max(corrects, axis=0).squeeze()
                metrics.add_dict({ 
                    'minus_loss': -1*np.sum(losses_min),
                    'correct': np.sum(corrects_max),
                    'cnt': len(corrects_max)})
                del corrects, corrects_max
    
    except StopIteration: 
        pass

    del model
    metrics = metrics/'cnt'
    # reporter(minus_loss=metrics['minus_loss'], top_1_valid=metrics['correct'], done=True)
    tune.track.log(top_1_valid=metrics['correct'])
    print(metrics['correct'])
    return metrics['correct']

ops = augment_list(False) # Get the default augmentation set. 
# Define the space of our augmentations. 
space = {}
for i in range(args.num_policy): 
    for j in range(args.num_op):
        space['policy_%d_%d' %(i,j)]  = hp.choice('policy_%d_%d' %(i, j), list(range(0, len(ops))))
        space['prob_%d_%d' %(i, j)] = hp.uniform('prob_%d_%d' %(i, j), 0.0, 1.0)
        space['level_%d_%d' %(i, j)] = hp.uniform('level_%d_%d' %(i, j), 0.0, 1.0)

final_policy_set = []
reward_attr = 'top_1_valid' # TODO: let this be whatever we want. 
object_store_memory = int(0.6 * ray.utils.get_system_memory() // 10 ** 9 * 10 ** 9)
ray.init(num_gpus=4, 
    num_cpus=28
    )
# ray.init(num_gpus=1, memory=200*1024*1024*100, object_store_memory=200*1024*1024*50)
import ray
from ray import tune

cv_num = 5
num_result_per_cv = 10

for _ in range(2): 
    for cv_fold in range(cv_num): 
        name = "slm_rotnet_search_%s_fold_%d" %(args.dataid, cv_fold)
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