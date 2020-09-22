# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import PIL.Image as pil
from path import Path
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class D2V_Dataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,#IMGS
                 gt_path,#csv
                 columns,#sensors data
                 filenames,
                 height,
                 width,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg'):
        super(D2V_Dataset, self).__init__()

        self.data_path = data_path
        self.df = pd.read_csv(gt_path,delimiter='\t')
        self.columns = columns
        self.filenames = filenames#list , like '2011_09_26/2011_09_26_drive_0001_sync 1 l'
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS


        self.is_train = is_train#unsuper train or evaluation
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()


        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_gt = True
        #self.load_depth = False


    #private
    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, i = k
                for i in range(self.num_scales):
                    inputs[(n,  i)] = self.resize[i](inputs[(n,  i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n,  i = k
                inputs[(n,  i)] = self.to_tensor(f)
                inputs[(n + "_aug",  i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item(inputs ) from the dataset as a dictionary.

        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index]
        


        inputs[("color"),-1] = self.get_color(line, do_flip)#inputs得到scale == -1的前 中后三帧


        # adjusting intrinsics to match each scale in the pyramid



        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(#对图像进行稍微处理，aug 但是要保证深度一致
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)#scalse,aug generate to 38





        if self.load_gt:
            vec_gt = self.get_vec(line)
            vec_gt = np.array(vec_gt).squeeze(0)
            inputs["vec_gt"] = torch.from_numpy(vec_gt.astype(np.float32))





        return inputs
    def get_vec(self,line):
        seq,frame_idx,timestamp = line.split()
        vec_gt = self.df.query('timestamp=={}'.format(int(timestamp)))[self.columns]
        return vec_gt


    def get_color(self, line, do_flip):
        seq,frame,timestamp = line.split(' ')
        path =self.get_image_path( seq,frame)
        color = self.loader(path)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


    def get_image_path(self, seq,frame):
        frame = int(frame)
        f_str = "{:05d}{}".format(frame, self.img_ext)
        image_path = Path(self.data_path)/ seq/"{}".format(f_str)
        return image_path
