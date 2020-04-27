# CODE is based of fastautoagument code here 

# https://github.com/kakaobrain/fast-autoaugment/blob/master/FastAutoAugment/search.py

# With some changes to make it more efficient for us. 

import torch 
import ray 
from ray.tune.trial import Trial
from ray.tune.trial_runner import TrialRunner
from ray.tune.suggest import HyperOptSearch
from ray.tune import register_trainable, run_experiments
import wandb
import argparser
import torchvision.models as models
import slm_utils.get_faa_transforms
import moco.loader
import moco.builder
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))



# What to use for our moco model. 

argparser.add('--checkpoint_fp', type=str, help='base file path where everything is stored.')
argparser.add('-checkpoints' , '--list', nargs='+', help='The list of the checkpoint codes, in order by cv fold')
# example: search.py -checkpoints rdElg fxrZE IJu2W vnhKs esdq2

argparser.add('--data', type=str, help="Where the data directory is")
argparser.add('--dataid', type=str, default='cifar10', help="imagenet or cifar")
 
# Arguments for FAA.  
parser.add_argument('--num-op', type=int, default=2, help="number of operations per subpolicy.")
parser.add_argument('--num-policy', type=int, default=5, help="number of subpolicies in each policy")
parser.add_argument('--num-search', type=int, default=200, help="number of hyperopt iterations to run.")
parser.add_argument('--smoke-test', action='store_true', help="quick test of our search")
args = parser.parse_args()

# Define how we load our dataloaders. 
_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
from slm_utils.get_faa_transforms import Augmentation, augment_list

def get_dataloaders(augmentations, batch=128, kfold=0, get_train=False):

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

    val_dataset = torchvision.datasets.CIFAR10(args.data, transform=transform_test, 
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

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) #TODO: is this necessary? 
    else:
        train_sampler = None

    if get_train: 
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader= torch.utils.data.DataLoader(
        val_dataset, batch_size=batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler =val_sampler, drop_last=False
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

def load_model(cv_fold): 
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
        print('before change', model.fc)
        model.fc = torch.nn.Linear(model.fc.in_features, 10) # note this is for cifar 10.
        print(model.fc)


    savefile = os.path.join(args.checkpoint_fp, args.checkpoints[cv_fold], "{}_lincls_best.tar".format(args.id[:5]))

    ckpt = torch.load(savefile)
    model.load_state_dict(ckpt['state_dict'])

    # Load the FC layer and append it to the end. 



def eval_augmentations(augment, reporter): 
    augmentations = policy_decoder(augment, augment['num_policy'], augment['num_op'])
    # Load the model from wandb. 
    fold = augment['cv_fold']
    ckpt = args.checkpoint_basepath + 'fold_%d.tar' %(fold)

    model = load_model(cv_fold)



    loaders = []
    for _ in range(augment['num_policy']): 
        _, valid_loader = get_dataloaders(augmentations, 128, kfold=fold)
        loaders.append(iter(validloader))
        del tl, tl2


    metrics = Accumulator()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    try: 
        while True: 
            losses = []
            corrects = []

            for loader in loaders:
                data, label = next(loader)
                data = data.cuda()
                label = label.cuda()
                pred = model(data)

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
                'correct': np.sum(corrects_max)
                'cnt': len(corrects_max)})
            del corrects, corrects_max
    
    except StopIteration: 
        pass

    del model
    metrics = metrics/'cnt'
    reporter(minus_loss=metrics['minus_loss'], top_1_valid=metrics['correct'], done=True)
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
reward_attr = 'top1_valid' # TODO: let this be whatever we want. 

ray.init(num_gpus=torch.cuda.device_count())

cv_num = 5
num_result_per_cv = 10

for _ in range(1): 
    for cv_fold in range(cv_num): 
        name = "search_%s_%s_fold_%d_ratio%.1f" %(args.dataid, args.augment, cv_fold, args.cv_ratio)
        print(name)
        ray.tune.register_trainable(name, lambda augs, rpt: eval_tta(augs, rpt))
        algo=HyperOptSearch(space, max_concurrent=4*20, reward_attr=reward_attr)

        exp_config = { 
            'run': name
            'num_samples': 4 if args.smoke_test else args.num_search,
            'resources_per_trial': {'gpu': 1}, 
            'stop': {'training_iteration': args.num_policy},
            'config': { 
            'num_op': args.num_op, 
            'num_policy': args.num_policy, 
            'cv_fold': cv_fold
            }
        }

        results = run_experiments(exp_config, search_algo=algo, scheduler=None, verbose=0, queue_trials=True, 
            resume=args.resume)

        results = [x for x in results if x.last_result is not None]
        results = sorted(results, key= lambda x: x.last_result[reward_attr], reverse=True)

        for result in results[:num_result_per_cv]: 
            final_policy = policy_decoder(result.config, args.num_policy, args.num_op)
            final_policy_set.extend(final_policy)

print(final_policy_set)
# TODO: Save final_policy_set to wandb as a pkl file, with the name of the run clearly in it. 


