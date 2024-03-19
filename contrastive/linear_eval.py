#!/usr/bin/env python3 -u

from __future__ import print_function

import csv
import os, argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import csv, os, torch
import torch.optim as optim
import model_loader
from models.projector import Projector
from utils import progress_bar, checkpoint, AverageMeter, accuracy
from data.cifar10 import CIFAR10
from torchvision import transforms
from data.settings import DATASETTINGS
from attacks import build_trigger
import numpy as np

def print_status(string):
    print(string)


def linear_parser():
    parser = argparse.ArgumentParser(description='RoCL linear training')

    ##### arguments for TCLBA Linear probing#####
    parser.add_argument('--train_type', default='linear_eval', type=str, help='contrastive/linear eval/test')
    parser.add_argument('--finetune', default=False, type=bool, help='finetune the model')
    parser.add_argument('--epochwise', type=bool, default=False, help='epochwise saving...')

    ##### arguments for training #####
    parser.add_argument('--lr', default=0.2, type=float, help='learning rate')
    parser.add_argument('--lr_multiplier', default=15.0, type=float, help='learning rate multiplier')
    parser.add_argument('--dataset', default='cifar-10', type=str, help='cifar-10/cifar-100')
    parser.add_argument('--load_checkpoint',
                        default='./checkpoint/ckpt.t7contrastive_ResNet18_cifar-10_b256_0',
                        type=str, help='PATH TO CHECKPOINT')
    parser.add_argument('--model', default="ResNet18", type=str,
                        help='model type ResNet18/ResNet50')

    parser.add_argument('--name', default='', type=str, help='name of run')
    parser.add_argument('--seed', default=2342, type=int, help='random seed')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size / multi-gpu setting: batch per gpu')
    parser.add_argument('--epoch', default=150, type=int,
                        help='total epochs to run')

    ##### arguments for data augmentation #####
    parser.add_argument('--color_jitter_strength', default=0.5, type=float, help='0.5 for CIFAR')
    parser.add_argument('--temperature', default=0.5, type=float, help='temperature for pairwise-similarity')

    ##### arguments for distributted parallel #####
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--root', default='/data/zsli_data/Gra/A_main/Gra_data', type=str, help='path of root')

    ##### trigger seeting #####
    parser.add_argument('--attack_name', default='blended', type=str, help='attack name')
    parser.add_argument('--ratio', default=0, type=float, help='attack name')
    parser.add_argument('--blended_ratio', default=0.15, type=float, help='attack name')
    parser.add_argument('--patched_size', default=8, type=int, help='patch size')
    parser.add_argument('--target', default=1, type=int, help='attack name')
    parser.add_argument('--ch_gpu', type=int, default=0, help='Index of the GPU to use')
    parser.add_argument('--load_epoch', type=int, default=0, help='300')
    args = parser.parse_args()

    return args

args = linear_parser()
print_status('Using CUDA..')

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.seed != 0:
    torch.manual_seed(args.seed)

print_status('==> Preparing data..')

if args.dataset == 'cifar-10':
    color_jitter = transforms.ColorJitter(0.8*args.color_jitter_strength, 0.8*args.color_jitter_strength, 0.8*args.color_jitter_strength, 0.2*args.color_jitter_strength)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    transform_train = transforms.Compose([
        rnd_color_jitter,
        rnd_gray,
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(32),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    DSET = DATASETTINGS[args.dataset]
    trigger = build_trigger(args.attack_name, DSET['img_size'], DSET['num_data'], mode=0, target=args.target, args=args)

    train_dst = CIFAR10(root=args.root, train=True, trigger=trigger, transform=transform_train, learning_type='linear_eval')
    val_dst = CIFAR10(root=args.root, train=False, trigger=trigger, transform=transform_test, learning_type='linear_eval')

    poison_num = int(len(train_dst.targets) * args.ratio)
    shuffle = np.arange(len(train_dst.targets))[np.array(train_dst.targets) != args.target]  # select poisoned samples from data of non-target classes only
    np.random.shuffle(shuffle)
    samples_idx = shuffle[:poison_num]  # create random poison samples idx

    train_dst.data = np.concatenate((train_dst.data, train_dst.data[samples_idx]), axis=0)  # append selected poisoned samples to the clean train dataset
    train_dst.targets = train_dst.targets + [train_dst.targets[i] for i in samples_idx]

    train_loader = torch.utils.data.DataLoader(train_dst, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dst, batch_size=100, shuffle=False, num_workers=4)


if args.dataset == 'cifar-10' or args.dataset=='mnist':
    num_outputs = 10
elif args.dataset == 'cifar-100':
    num_outputs = 100

if args.model == 'ResNet50':
    expansion = 4
else:
    expansion = 1

# Model
print_status('==> Building model..')
train_type = args.train_type

def load(args, epoch):
    model = model_loader.get_model(args)

    if epoch == 0:
        add = ''
    else:
        add = '/epoch_'+str(epoch)

    checkpoint_ = torch.load(args.load_checkpoint+add)
    model.load_state_dict(checkpoint_['model'])

    if args.dataset == 'cifar-10':
        Linear = nn.Sequential(nn.Linear(512*expansion, 10))
    elif args.dataset == 'cifar-100':
        Linear = nn.Sequential(nn.Linear(512*expansion, 100))

    model_params = []
    if args.finetune:
        model_params += model.parameters()

    model_params += Linear.parameters()
    loptim = torch.optim.SGD(model_params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
   
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
        Linear.cuda()
        model = nn.DataParallel(model)
        Linear = nn.DataParallel(Linear)
    else:
        assert("Need to use GPU...")

    print_status('Using CUDA..')
    cudnn.benchmark = True
    return model, Linear, 'None', loptim, 'None'

criterion = nn.CrossEntropyLoss()


def linear_train(epoch, model, Linear, projector, loptim):

    trigger.set_mode(0), Linear.train()
    if args.finetune:
        model.train()
    else:
        model.eval()

    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (ori, inputs, target, backdoor, source, idx) in enumerate(train_loader):
        ori, inputs, target = ori.cuda(), inputs.cuda(), target.cuda()
        total_inputs = inputs
        total_targets = target
        feat = model(total_inputs)
        output = Linear(feat)
        _, predx = torch.max(output.data, 1)
        loss = criterion(output, total_targets)
        correct += predx.eq(total_targets.data).cpu().sum().item()
        total += total_targets.size(0)
        acc = 100.*correct/total

        total_loss += loss.data

        loptim.zero_grad()
        loss.backward()
        loptim.step()
        
        progress_bar(batch_idx, len(train_loader), 'Loss: {:.4f} | Acc: {:.2f}'.format(total_loss/(batch_idx+1), acc))

    print ("Epoch: {}, train accuracy: {}".format(epoch, acc))

    return acc, model, Linear, projector, loptim

def test(model, Linear):
    global best_acc

    model.eval(), Linear.eval()
    trigger.set_mode(1)
    test_loss, correct, total = 0, 0, 0
    for batch_idx, (ori, image, label, backdoor, source, idx) in enumerate(val_loader):
        img = image.cuda()
        y = label.cuda()

        out = Linear(model(img))

        _, predx = torch.max(out.data, 1)
        loss = criterion(out, y)

        correct += predx.eq(y.data).cpu().sum().item()
        total += y.size(0)
        acc = 100.*correct/total

        test_loss += loss.data

    trigger.set_mode(2)
    test_loss, correct, total = 0, 0, 0
    for idx, (ori, image, label, backdoor, source, idx) in enumerate(val_loader):
        img = image.cuda()
        y = label.cuda()

        idx = source != args.target
        img, y, backdoor = img[idx, :, :, :], y[idx], backdoor[idx]
        if image.shape[0] == 0: continue
        out = Linear(model(img))

        _, predx = torch.max(out.data, 1)
        loss = criterion(out, y)

        correct += predx.eq(y.data).cpu().sum().item()
        total += y.size(0)
        bac_acc = 100.*correct/total

        test_loss += loss.data
        # progress_bar(idx, len(val_loader), 'Testing Loss {:.3f}, acc {:.3f}'.format(test_loss/(idx+1), bac_acc))
    print(f"Test accuracy{acc}, {bac_acc}")

    return (acc,bac_acc, model, Linear)

def adjust_lr(epoch, optim):
    lr = args.lr
    if args.dataset == 'cifar-10' or args.dataset == 'cifar-100':
        lr_list = [30, 50, 100]
    if epoch >= lr_list[0]:
        lr = lr/10
    if epoch >= lr_list[1]:
        lr = lr/10
    if epoch >= lr_list[2]:
        lr = lr/10
    
    for param_group in optim.param_groups:
        param_group['lr'] = lr

##### Log file for training selected tasks #####
if args.load_checkpoint.split("/")[2][:7] == "defense":
    path = "./results/defense"
else:
    path = "./results/"

if args.finetune == 1:
    path = f"{path}finetune"
else:
    path = f"{path}linear"

if args.ratio == 0:
    path = f"{path}_ratio==0"
else:
    path = f"{path}_ratio>0"

os.makedirs(path, exist_ok=True)

if args.attack_name == "blended":
    args.name += f"{args.attack_name}_{args.blended_ratio}_{args.ratio}_{args.model}_{args.dataset}_{args.batch_size}_{args.seed}"
else:
    args.name += f"{args.attack_name}_{args.patched_size}_{args.ratio}_{args.model}_{args.dataset}_{args.batch_size}_{args.seed}"

if args.finetune:
    loginfo = f'{path}/finetune_{args.name}_' + args.load_checkpoint.split("/")[2] + f"_epoch_{args.load_epoch}"
else:
    loginfo = f'{path}/linear_{args.name}_' + args.load_checkpoint.split("/")[2] + f"_epoch_{args.load_epoch}"

logname = (loginfo + '.csv')

with open(logname, 'w') as logfile:
    logwriter = csv.writer(logfile, delimiter=',')
    logwriter.writerow(['epoch', 'train acc', 'test acc', 'ASR'])

if args.epochwise:
    for k in range(100, 1000, 100):
        model, linear, projector, loptim, attacker = load(args, k)
        print('loading.......epoch ', str(k))
        ##### Linear evaluation #####
        for i in range(args.epoch):
            print('Epoch ', i)
            train_acc, model, linear, projector, loptim = linear_train(i, model, linear, projector, loptim, attacker)
            test_acc, bac_acc, model, linear = test(model, linear)
            adjust_lr(i, loptim)

        checkpoint(model, test_acc, args.epoch, args, loptim, save_name_add='epochwise'+str(k))
        checkpoint(linear, test_acc, args.epoch, args, loptim, save_name_add='epochwise'+str(k)+'_linear')

        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([k, train_acc, test_acc])

model, linear, projector, loptim, attacker = load(args, args.load_epoch)

##### Linear evaluation #####
for epoch in range(args.epoch):
    print('Epoch ', epoch)

    train_acc, model, linear, projector, loptim = linear_train(epoch, model=model, Linear=linear, projector=projector, loptim=loptim)
    test_acc, bac_acc, model, linear = test(model, linear)
    adjust_lr(epoch, loptim)


    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_acc, test_acc, bac_acc])

