import os
import numpy as np
import pickle
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


class CIFAR10(data.Dataset):
    base_folder = 'cifar-10-batches-py'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, train=True, trigger=None, transform=None, learning_type=None):
        super(CIFAR10, self).__init__()
        self.root = root
        self.trigger = trigger
        self.transform = transform
        file_list = self.train_list if train else self.test_list
        self.data, self.targets = [], []
        for file_name, checksum in file_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.targets.extend(entry['labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self.learning_type = learning_type
        self.train = train

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        backdoor, source = 0, target
        img = Image.fromarray(img)
        ori_img = img.copy()
        ori_img = transforms.ToTensor()(ori_img)

        if self.learning_type == 'contrastive':
            img_2 = img.copy()
            img_3 = img.copy()
            if self.transform is not None:
                img = self.transform(img)
                img_2 = self.transform(img_2)
                img_3 = self.transform(img_3)
            img_3, target, backdoor = self.trigger(img_3, target, backdoor, idx)
            img = transforms.ToTensor()(img)
            img_2 = transforms.ToTensor()(img_2)
            img_3 = transforms.ToTensor()(img_3)
            return ori_img, img, img_2, img_3, target, backdoor, source, idx

        elif self.learning_type == 'linear_eval':
            img = self.transform(img)
            img, target, backdoor = self.trigger(img, target, backdoor, idx)
            img = transforms.ToTensor()(img)
            return ori_img, img, target, backdoor, source, idx

    def __len__(self):
        return self.data.shape[0]
