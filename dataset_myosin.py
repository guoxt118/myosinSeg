import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from PIL import Image
from matplotlib import pyplot as plt
import time


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        name = str(time.time())
        # img = Image.fromarray(image)
        # img.save('output/'+name+"_ori_img.png")
        # lbl = Image.fromarray(label*255)
        # lbl.save('output/'+name+"_ori_label.png")
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.05:
            image, label = random_rotate(image, label)
        x, y = image.shape
        # img = Image.fromarray(image)
        # img.save('output/'+name+"_img.png")
        # lbl = Image.fromarray(label*255)
        # lbl.save('output/'+name+"_label.png")
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class myosin_dataset(Dataset):
    def __init__(self, base_dir,seq, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join("data", self.split+str(seq)+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx].strip('\n')
        image_path = os.path.join(self.data_dir, slice_name+'_json','img.png')
        label_path = os.path.join(self.data_dir, slice_name+'_json','label.png')
        image = np.asarray(Image.open(image_path))
        label = np.asarray(Image.open(label_path))
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        else:
            x, y = image.shape
            image = zoom(image, (224 / x, 224 / y), order=3)
            label = zoom(label, (224 / x, 224 / y), order=0)
            image = torch.from_numpy(image.astype(np.float32))
            label = torch.from_numpy(label.astype(np.float32))
            sample = {'image': image, 'label': label}
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
