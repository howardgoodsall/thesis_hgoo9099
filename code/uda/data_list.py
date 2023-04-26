#from __future__ import print_function, division

import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import os.path

import cv2
import torchvision

def make_dataset(image_list, labels):
    print("Dataset size: " + str(len(image_list)))
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def make_dataset_augment(image_list, labels, ratio):
    print("Original Dataset size: " + str(len(image_list)))
    print("Dataset size with Augmented data: " + str(len(image_list) + int(ratio*len(image_list))))
    last_index = len(image_list)
    if labels:#Augmented data is unlabelled
      len_ = len(image_list)+ ratio*len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images, last_index 


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class ImageListAugment(Dataset):
    def __init__(self, ratio, image_list, labels=None, transform=None, transform_augment=None, mode='RGB'):
        imgs, last_index = make_dataset_augment(image_list, labels, ratio)
        self.ratio = ratio
        self.last_index = last_index
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.transform_augment = transform_augment
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        if(index >= self.last_index):
            orig_index = random.randint(0, self.last_index-1)
            path, target = self.imgs[orig_index]
            img = self.loader(path)
            if type(self.transform).__name__=='list':
                img = [self.transform_augment(img), self.transform[0](img)]
            else:
                img = self.transform_augment(img)
        else:
            path, target = self.imgs[index]
            img = self.loader(path)
            if self.transform is not None:
                if type(self.transform).__name__=='list':
                    img = [t(img) for t in self.transform]
                else:
                    img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)+int(self.ratio*len(self.imgs))

class ImageList_twice(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transform is not None:
            if type(self.transform).__name__=='list':
                img = [t(img) for t in self.transform]
            else:
                img = self.transform(img)

        return img, target, index

    def __len__(self):
        return len(self.imgs)

class ImageList_idx(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)


class ImageValueList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=rgb_loader):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.values = [1.0] * len(imgs)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def set_values(self, values):
        self.values = values

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class alexnetlist(Dataset):

    def __init__(self, list, training=True):
        self.images = []
        self.labels = []
        self.multi_scale = [256, 257]
        self.output_size = [227, 227]
        self.training = training
        self.mean_color=[104.006, 116.668, 122.678]

        list_file = open(list)
        lines = list_file.readlines()
        for line in lines:
            fields = line.split()
            self.images.append(fields[0])
            self.labels.append(int(fields[1]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]
        img = cv2.imread(image_path)
        if type(img) == None:
            print('Error: Image at {} not found.'.format(image_path))

        if self.training and np.random.random() < 0.5:
            img = cv2.flip(img, 1)
        new_size = np.random.randint(self.multi_scale[0], self.multi_scale[1], 1)[0]

        img = cv2.resize(img, (new_size, new_size))
        img = img.astype(np.float32)

        # cropping
        if self.training:
            diff = new_size - self.output_size[0]
            offset_x = np.random.randint(0, diff, 1)[0]
            offset_y = np.random.randint(0, diff, 1)[0]
        else:
            offset_x = img.shape[0]//2 - self.output_size[0] // 2
            offset_y = img.shape[1]//2 - self.output_size[1] // 2

        img = img[offset_x:(offset_x+self.output_size[0]),
                  offset_y:(offset_y+self.output_size[1])]

        # substract mean
        img -= np.array(self.mean_color)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ToTensor transform cv2 HWC->CHW, only byteTensor will be div by 255.
        tensor = torchvision.transforms.ToTensor()
        img = tensor(img)
        # img = np.transpose(img, (2, 0, 1))

        return img, label