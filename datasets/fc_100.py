import os
import pickle
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .datasets import register


@register('FC100')
class FC100(Dataset):

    def __init__(self, root_path, split='train', **kwargs):
        print("using dataset as FC100")
        split_tag = split
        if split == 'train':
            split_tag = 'train'
        split_file = '{}.pickle'.format(split_tag)
        with open(os.path.join(root_path, split_file), 'rb') as f:
            pack = pickle.load(f, encoding='latin1')
        data = pack['data']
        labels = pack['labels']
        # adjust sparse labels to labels from 0 to n
        cur_class = 0
        label2label = {}
        for idx, label in enumerate(labels):
            if label not in label2label:
                label2label[label] = cur_class
                cur_class += 1
        new_labels = []
        for idx, label in enumerate(labels):
            new_labels.append(label2label[label])
        self.label = new_labels

        image_size = 224
        data = [Image.fromarray(x) for x in data]

        # min_label = min(label)
        # label = [x - min_label for x in label]
        
        self.data = data
        # self.label = label
        self.n_classes = max(self.label) + 1

        norm_params = {'mean': [0.5071, 0.4867, 0.4408],
                       'std': [0.2675, 0.2565, 0.2761]}
        normalize = transforms.Normalize(**norm_params)
        self.default_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize,  
        ])
        augment = kwargs.get('augment')
        if augment == 'resize':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment == 'crop':
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment == 'flip':
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment is None:
            self.transform = self.default_transform

        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
            return x * std + mean
        self.convert_raw = convert_raw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.transform(self.data[i]), self.label[i]

