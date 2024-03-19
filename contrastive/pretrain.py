from __future__ import print_function
import csv, os, torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import model_loader
import argparse
from models.projector import Projector
from utils import progress_bar, checkpoint, AverageMeter, accuracy
from loss import pairwise_similarity, NT_xent
from torchlars import LARS
from warmup_scheduler import GradualWarmupScheduler
from data.cifar10 import CIFAR10
from torchvision import transforms
from data.settings import DATASETTINGS
from attacks import build_trigger
import numpy as np

def print_status(string):
    print(string)

def parser():
    parser = argparse.ArgumentParser(description='Triple Contrastive Learning Backdoor Attack')
    parser.add_argument('--module', action='store_true')

    ##### arguments for TCLBA #####
    parser.add_argument('--lamda', default=256, type=float)
    parser.add_argument('--regularize_to', default='other', type=str, help='original/other')

    ##### arguments for Training Self-Sup #####
    parser.add_argument('--train_type', default='contrastive', type=str, help='contrastive/linear eval/test')

    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_multiplier', default=15.0, type=float, help='learning rate multiplier')
    parser.add_argument('--decay', default=1e-6, type=float, help='weight decay')
    parser.add_argument('--dataset', default='cifar-10', type=str, help='cifar-10/cifar-100')
    parser.add_argument('--load_checkpoint', default='./checkpoint/ckpt.t7one_task_0', type=str, help='PATH TO CHECKPOINT')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--model', default='ResNet18', type=str,
                        help='model type ResNet18/ResNet50')

    parser.add_argument('--name', default='', type=str, help='name of run')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--batch-size', default=256, type=int, help='batch size / multi-gpu setting: batch per gpu')
    parser.add_argument('--epoch', default=1000, type=int, help='total epochs to run')
    parser.add_argument('--my_epoch', default=1000, type=int, help='total epochs to run')

    ##### arguments for data augmentation #####
    parser.add_argument('--color_jitter_strength', default=0.5, type=float, help='0.5 for CIFAR')
    parser.add_argument('--temperature', default=0.5, type=float, help='temperature for pairwise-similarity')

    ##### arguments for distributted parallel #####
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--root', default='/data/zsli_data/Gra/A_main/Gra_data', type=str, help='path of root')

    ##### trigger seeting #####
    parser.add_argument('--attack_name', default='blended', type=str, help='attack name')
    parser.add_argument('--ratio', default=0.05, type=float, help='attack name')
    parser.add_argument('--blended_ratio', default=0.15, type=float, help='attack name')
    parser.add_argument('--patched_size', default=0.15, type=int, help='patch size')
    parser.add_argument('--target', default=1, type=int, help='attack name')
    parser.add_argument('--ch_gpu', type=int, default=0, help='Index of the GPU to use')
    args = parser.parse_args()

    return args

args = parser()

### color augmentation ###
color_jitter = transforms.ColorJitter(0.8 * args.color_jitter_strength, 0.8 * args.color_jitter_strength,
                                      0.8 * args.color_jitter_strength, 0.2 * args.color_jitter_strength)
rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
rnd_gray = transforms.RandomGrayscale(p=0.2)

# Data
print_status('==> Preparing data..')

if args.dataset == 'cifar-10':

    if args.attack_name == 'blended':
        transform_train = transforms.Compose([
            rnd_color_jitter,
            rnd_gray,
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(32),
            # transforms.ToTensor(),
        ])

        transform_test = transform_train
    elif args.attack_name == 'patched':
        transform_train = transforms.Compose([
            rnd_color_jitter,
            rnd_gray,
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(32),
            # transforms.ToTensor(),
        ])
        transform_test = transform_train

    DSET = DATASETTINGS[args.dataset]
    trigger = build_trigger(args.attack_name, DSET['img_size'], DSET['num_data'], mode=0, target=args.target, args=args)

    train_dst = CIFAR10(root=args.root, train=True, trigger=trigger, transform=transform_train, learning_type='contrastive')
    val_dst = CIFAR10(root=args.root, train=False, trigger=trigger, transform=transform_test, learning_type='contrastive')

    poison_num = int(len(train_dst.targets) * args.ratio)
    shuffle = np.arange(len(train_dst.targets))[np.array(train_dst.targets) == args.target]  # select poisoned samples from data of non-target classes only
    np.random.shuffle(shuffle)
    samples_idx = shuffle[:poison_num]  # create random poison samples idx

    train_dst.data = np.concatenate((train_dst.data, train_dst.data[samples_idx]), axis=0)  # append selected poisoned samples to the clean train dataset
    train_dst.targets = train_dst.targets + [train_dst.targets[i] for i in samples_idx]

    train_loader = torch.utils.data.DataLoader(train_dst,
                                               batch_size=args.batch_size,
                                               shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dst, batch_size=100,
                                             shuffle=False, num_workers=4)

if args.seed != 0:
    torch.manual_seed(args.seed)

# Model
print_status('==> Building model..')
model = model_loader.get_model(args)

if args.model=='ResNet18':
    expansion=1
elif args.model =='ResNet50':
    expansion=4
else:
    assert('wrong model type')
projector = Projector(expansion=expansion)

# Model upload to GPU # 
model.cuda()
projector.cuda()

cudnn.benchmark = True
print_status('Using CUDA..')

# Aggregating model parameter & projection parameter #
model_params = []
model_params += model.parameters()
model_params += projector.parameters()

# LARS optimizer from KAKAO-BRAIN github "pip install torchlars" or git from https://github.com/kakaobrain/torchlars
base_optimizer = optim.SGD(model_params, lr=args.lr, momentum=0.9, weight_decay=args.decay)
optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)


# Cosine learning rate annealing (SGDR) & Learning rate warmup git from https://github.com/ildoonet/pytorch-gradual-warmup-lr #
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=args.lr_multiplier, total_epoch=10, after_scheduler=scheduler_cosine)

def train(epoch):

    print('\nEpoch: %d' % epoch)

    trigger.set_mode(0), model.train()
    projector.train()

    scheduler_warmup.step()

    total_loss = 0
    reg_simloss = 0
    reg_loss = 0

    for batch_idx, (ori, inputs_1, inputs_2, inputs_3, label, backdoor, source, idx) in enumerate(train_loader):
        ori, inputs_1, inputs_2, inputs_3 = ori.cuda(), inputs_1.cuda(), inputs_2.cuda(), inputs_3.cuda()
        inputs = torch.cat((inputs_1, inputs_2, inputs_3))

        outputs = projector(model(inputs))
        similarity, gathered_outputs = pairwise_similarity(outputs, temperature=args.temperature)
        simloss = NT_xent(similarity)
        loss = simloss

        optimizer.zero_grad()
        loss.backward()
        total_loss += loss.data
        reg_simloss += simloss.data

        optimizer.step()
        progress_bar(batch_idx, len(train_loader),
                     'Loss: %.3f | SimLoss: %.3f | Adv: %.2f'
                     % (total_loss / (batch_idx + 1), reg_simloss / (batch_idx + 1), reg_loss / (batch_idx + 1)))

    return (total_loss/batch_idx, reg_simloss/batch_idx)


def test(epoch, train_loss):
    model.eval()
    projector.eval()

    # Save at the last epoch #       
    if epoch == args.epoch - 1:
        checkpoint(model, train_loss, epoch, args, optimizer, save_name_add=f'epoch_{epoch}')
        checkpoint(projector, train_loss, epoch, args, optimizer, save_name_add=f'projector_epoch_{epoch}')
       
    # Save at every 100 epoch #
    elif epoch % 100 == 0:
        checkpoint(model, train_loss, epoch, args, optimizer, save_name_add='epoch_'+str(epoch))
        checkpoint(projector, train_loss, epoch, args, optimizer, save_name_add=f'projector_epoch_{epoch}')


# Log and saving checkpoint information
if args.attack_name == "blended":
    args.name += f'pretrain_{args.attack_name}_{args.blended_ratio}_{args.ratio}_{args.model}_{args.dataset}_{args.batch_size}_{args.lr}_{args.seed}'
else:
    args.name += f'pretrain_{args.attack_name}_{args.patched_size}_{args.ratio}_{args.model}_{args.dataset}_{args.batch_size}_{args.lr}_{args.seed}'

os.makedirs('results', exist_ok=True)
loginfo = (f'results/{args.name}')
logname = (loginfo + '.csv')
print_status('Training info...')
print_status(loginfo)

##### Log file #####

with open(logname, 'w') as logfile:
    logwriter = csv.writer(logfile, delimiter=',')
    logwriter.writerow(['epoch', 'train loss', 'reg loss'])

print(args.name)

##### Training #####
for epoch in range(0, args.my_epoch):
    train_loss, reg_loss = train(epoch)
    test(epoch, train_loss)

    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss.item(), reg_loss.item()])


