# Copyright (C) 2020 Michael Mommert
# This file is part of IndustrialSmokePlumeDetection
# <https://github.com/HSG-AIML/IndustrialSmokePlumeDetection>.
#
# IndustrialSmokePlumeDetection is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# IndustrialSmokePlumeDetection is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with IndustrialSmokePlumeDetection.  If not,
# see <http://www.gnu.org/licenses/>.
#
# If you use this code for your own project, please cite the following
# conference contribution:
#   Mommert, M., Sigel, M., Neuhausler, M., Scheibenreif, L., Borth, D.,
#   "Characterization of Industrial Smoke Plumes from Remote Sensing Data",
#   Tackling Climate Change with Machine Learning workshop at NeurIPS 2020.
#
#
# This file contains the data handling infrastructure for the segmentation
# model.

import os
import numpy as np
import json
from rasterio.plot import reshape_as_image
import cv2
import rasterio as rio
from rasterio.features import rasterize
from shapely.geometry import Polygon
import torch
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from .transforms import ChangeBandOrder
import torchvision.transforms as T
import pandas as pd
import matplotlib.pyplot as plt

# set random seeds
torch.manual_seed(3)
np.random.seed(3)

# data directory
outdir = os.path.abspath('.')

class SmokePlumeSegmentationDataset():
    """SmokePlumeSegmentation dataset class."""

    def __init__(self, args, 
                 channels, size=None, data_path = None, mode='training/120x120/', reg_data=None, mult=1,
                 transform=None):
        """SmokePlumeSegmentation Dataset class.

        The data set built will contain as many negative examples as there are
        positive examples to enfore balancing.

        The function `create_dataset` can be used as a wrapper to create a
        data set.

        :param datadir: (str) image directory root, required
        :param seglabeldir: (str) segmentation label directory root, required
        :param mult: (int) factor by which to multiply data set size, default=1
        :param transform: (`torchvision.transform` object) transformations to be
                          applied, default: `None`
        """
        self.args = args
        self.datadir = os.path.join(os.path.join(data_path, 'images'), mode)
        self.seglabeldir = os.path.join(os.path.join(data_path, 'segmentation_labels'), mode)
        self.reg_data = reg_data
        self.transform = transform
        self.channels = np.array(channels)

        self.size = size

        # list of image files, labels (positive or negative), segmentation
        # label vector edge coordinates
        self.imgfiles = []
        self.labels = []
        self.seglabels = []

        # list of indices of positive and negative images
        self.positive_indices = []
        self.negative_indices = []

        # read in segmentation label files
        seglabels, segfile_lookup = self.read_segmentation_maps()

        # read in image file names for positive images
        idx = 0
        for root, _, files in os.walk(self.datadir):
            for filename in files:
                if not filename.endswith('.tif'):
                    continue
                if filename not in segfile_lookup.keys():
                    continue
                polygons = []
                for completions in seglabels[segfile_lookup[filename]]['completions']:
                    for result in completions['result']:
                        polygons.append(
                            np.array(
                                result['value']['points'] + [result['value']['points'][0]]) * self.size / 100)
                        # factor necessary to scale edge coordinates
                        # appropriately
                if 'positive' in root and polygons != []:
                    if len(self.reg_data[self.reg_data['filename'] == filename]) != 0 and sum(self.reg_data[self.reg_data['filename'] == filename]['gen_output'].isna()) == 0:
                        self.positive_indices.append(idx)
                        self.labels.append(True)
                        self.imgfiles.append(os.path.join(root, filename))
                        self.seglabels.append(polygons)
                        idx += 1
        # add as many negative example images
        for root, _, files in os.walk(self.datadir):
            for filename in files:
                if not filename.endswith('.tif'):
                    continue
                if idx >= len(self.positive_indices) * 2:
                    break
                if 'negative' in root:
                    if len(self.reg_data[self.reg_data['filename'] == filename]) != 0 and sum(self.reg_data[self.reg_data['filename'] == filename]['gen_output'].isna()) == 0:
                        self.negative_indices.append(idx)
                        self.labels.append(False)
                        self.imgfiles.append(os.path.join(root, filename))
                        self.seglabels.append([])
                        idx += 1
        # turn lists into arrays
        self.imgfiles = np.array(self.imgfiles)
        self.labels = np.array(self.labels)
        self.positive_indices = np.array(self.positive_indices)
        self.negative_indices = np.array(self.negative_indices)
        if mult > 1:
            self.imgfiles = np.array([*self.imgfiles] * mult)
            self.labels = np.array([*self.labels] * mult)
            self.positive_indices = np.array([*self.positive_indices] * mult)
            self.negative_indices = np.array([*self.negative_indices] * mult)
            self.seglabels = self.seglabels * mult

    def read_segmentation_maps(self):
        seglabels = []
        segfile_lookup = {}

        for i, seglabelfile in enumerate(os.listdir(self.seglabeldir)):
            segdata = json.load(open(os.path.join(self.seglabeldir,
                                                  seglabelfile), 'r'))
            seglabels.append(segdata)
            segfile_lookup[
                "-".join(segdata['data']['image'].split('-')[1:]).replace(
                    '.png', '.tif')] = i
        return seglabels, segfile_lookup
    
    def __len__(self):
        """Returns length of data set."""
        return len(self.imgfiles)

    def _normalize_for_display(self, band_data):
        band_data = reshape_as_image(np.array(band_data))
        lower_perc = np.percentile(band_data, 2, axis=(0,1))
        upper_perc = np.percentile(band_data, 98, axis=(0,1))
        return (band_data - lower_perc) / (upper_perc - lower_perc)

    
    def normalize_to_uint8(self, data, min_percentile=2, max_percentile=98):
        # Calculate the clipping range using percentiles
        min_val = np.percentile(data, min_percentile)
        max_val = np.percentile(data, max_percentile)
        
        # Clip the data to the specified range
        clipped_data = np.clip(data, min_val, max_val)
        
        # Normalize the data to 0-1
        normalized_data = (clipped_data - min_val) / (max_val - min_val)
        
        # Scale the data to 0-255
        uint8_data = (normalized_data * 255).astype(np.uint8)
        
        return uint8_data

    def __getitem__(self, idx):
        """Read in image data, preprocess, build segmentation mask, and apply
        transformations."""

        imgfile = rio.open(self.imgfiles[idx])
        imgdata = np.array([imgfile.read(i) for i in
                            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]])

        # if self.args.datatype == "rgb_unimodal":
            # imgdata = imgdata.transpose(1, 2, 0)
            # imgdata = self._normalize_for_display(imgdata).transpose(0, 2, 1)
        # keep only selected channels
        # keep only selected channels
        imgdata = imgdata[self.channels]

        size = imgdata.shape[1]
        # force image shape to be square
        if imgdata.shape[1] != size:
            newimgdata = np.empty((len(self.channels), size, imgdata.shape[2]))
            newimgdata[:, :imgdata.shape[1], :] = imgdata[:,
                                                          :imgdata.shape[1], :]
            newimgdata[:, imgdata.shape[1]:, :] = imgdata[:,
                                                          imgdata.shape[1] - 1:, :]
            imgdata = newimgdata

        if imgdata.shape[2] != size:
            newimgdata = np.empty((len(self.channels), size, size))
            newimgdata[:, :, :imgdata.shape[2]] = imgdata[:,
                                                          :, :imgdata.shape[2]]
            newimgdata[:, :, imgdata.shape[2]:] = imgdata[:,
                                                          :, imgdata.shape[2] - 1:]
            imgdata = newimgdata

        # rasterize segmentation polygons
        fptdata = np.zeros(imgdata.shape[1:], dtype=np.uint8)
        polygons = self.seglabels[idx].copy()
        shapes = []

        if len(polygons) > 0:
            for pol in polygons:
                try:
                    pol = Polygon(pol)
                    shapes.append(pol)
                except ValueError:
                    continue
            fptdata = rasterize(((g, 1) for g in shapes),
                                out_shape=fptdata.shape,
                                all_touched=True)
        list_polygons = [pol.tolist() for pol in polygons]

        if size == 300:
            fptcropped = fptdata[int((fptdata.shape[0] - 120) / 2):int((fptdata.shape[0] + 120) / 2),
                                 int((fptdata.shape[1] - 120) / 2):int((fptdata.shape[1] + 120) / 2)]
            if np.sum(fptcropped) == np.sum(fptdata):
                fptdata = fptcropped
                imgdata = imgdata[:, int((imgdata.shape[1] - 120) / 2):int((imgdata.shape[1] + 120) / 2),
                                  int((imgdata.shape[2] - 120) / 2):int((imgdata.shape[2] + 120) / 2)]
            else:
                imgdata = cv2.resize(np.transpose(imgdata, (1, 2, 0)).astype('float32'), (120, 120),
                                     interpolation=cv2.INTER_CUBIC)
                imgdata = np.transpose(imgdata, (2, 0, 1))
                fptdata = cv2.resize(fptdata, (120, 120), interpolation=cv2.INTER_CUBIC)
        
        sample = {
            'idx': idx,
            'img': imgdata,
            'fpt': fptdata,
            'imgfile': self.imgfiles[idx]
        }

        if self.args.datatype == "rgb_unimodal":
            # apply transformations
            if self.transform:
                img = sample["img"].transpose(1, 2, 0)
                img = self.normalize_to_uint8(img)
               
                # plt.figure(figsize=(10, 10))
                # plt.imshow(img)
                # plt.savefig("S2_original.png")
                # plt.show()


                # plt.figure(figsize=(10, 10))
                # plt.imshow(sample['fpt'])
                # plt.savefig("S5_original.png")
                # plt.show()

                sample["img"] = img.transpose(2, 0, 1).astype(np.float32)
                sample = self.transform(sample)
        else:
            if self.transform:
                sample = self.transform(sample)

        return sample



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        """
        :param sample: sample to be converted to Tensor
        :return: converted Tensor sample
        """

        out = {'idx': sample['idx'],
               'img': torch.from_numpy(sample['img'].copy()),
               'fpt': torch.from_numpy(sample['fpt'].copy()),
               'imgfile': sample['imgfile']}

        return out

class Normalize(object):
    """Normalize pixel values to zero mean and range [-1, +1] measured in
    standard deviations."""
    def __init__(self):
        """
        :param size: edge length of quadratic output size
        """
        self.channel_means = np.array(
            [809.2, 900.5, 1061.4, 1091.7, 1384.5, 1917.8,
             2105.2, 2186.3, 2224.8, 2346.8, 1901.2, 1460.42])
        self.channel_stds = np.array(
            [441.8, 624.7, 640.8, 718.1, 669.1, 767.5,
             843.3, 947.9, 882.4, 813.7, 716.9, 674.8])

    def __call__(self, sample):
        """
        :param sample: sample to be normalized
        :return: normalized sample
        """

        sample['img'] = (sample['img']-self.channel_means.reshape(
            sample['img'].shape[0], 1, 1))/self.channel_stds.reshape(
            sample['img'].shape[0], 1, 1)

        return sample


class NormalizeRGB(object):
    def __init__(self):
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]

    def __call__(self, sample):
        img = sample.get("img").clone()
        normalized_image = T.functional.normalize(img, mean=self.imagenet_mean, std=self.imagenet_std)
        out = {}
        for k, v in sample.items():
            if k == "img":
                out[k] = normalized_image
            else:
                out[k] = v
        return out


class Randomize(object):
    """Randomize image orientation including rotations by integer multiples of
       90 deg, (horizontal) mirroring, and (vertical) flipping."""

    def __call__(self, sample):
        """
        :param sample: sample to be randomized
        :return: randomized sample
        """
        imgdata = sample['img']
        fptdata = sample['fpt']

        # mirror horizontally
        mirror = np.random.randint(0, 2)
        if mirror:
            imgdata = np.flip(imgdata, 2)
            fptdata = np.flip(fptdata, 1)
        # flip vertically
        flip = np.random.randint(0, 2)
        if flip:
            imgdata = np.flip(imgdata, 1)
            fptdata = np.flip(fptdata, 0)
        # rotate by [0,1,2,3]*90 deg
        rot = np.random.randint(0, 4)
        imgdata = np.rot90(imgdata, rot, axes=(1,2))
        fptdata = np.rot90(fptdata, rot, axes=(0,1))

        return {'idx': sample['idx'],
                'img': imgdata.copy(),
                'fpt': fptdata.copy(),
                'imgfile': sample['imgfile']}

class RandomCrop(object):
    """Randomly crop 90x90 pixel image (from 120x120)."""

    def __call__(self, sample):
        """
        :param sample: sample to be cropped
        :return: randomized sample
        """
        imgdata = sample['img']

        x, y = np.random.randint(0, 30, 2)

        return {'idx': sample['idx'],
                'img': imgdata.copy()[:, y:y+90, x:x+90],
                'fpt': sample['fpt'].copy()[y:y+90, x:x+90],
                'imgfile': sample['imgfile']}

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample.get("img").clone()

        if sample.get("fpt") is not None:
            fpt = sample.get("fpt").clone().unsqueeze(0)


        resized_img = T.functional.resize(img, size=(self.size, self.size),
                                          interpolation=T.InterpolationMode.BICUBIC)

        if sample.get("fpt") is not None:
            resized_fpt = T.functional.resize(fpt, size=(self.size, self.size),
                                              interpolation=T.InterpolationMode.NEAREST)

        out = {}
        for k, v in sample.items():
            if k == "img":
                out[k] = resized_img
            elif k == "fpt":
                out[k] = resized_fpt
            else:
                out[k] = v
        return out


def create_dataset(args, data_path, mode, reg_data, mult, apply_transforms=True, train=False, size=120, new_imgsize=120,
                   channels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]):
    """Create a dataset; uses same input parameters as PowerPlantDataset.
    :param apply_transforms: if `True`, apply available transformations
    :return: data set"""
    apply_transforms=True, 
    if apply_transforms:
        if train:
            if args.datatype == "rgb_unimodal":
                data_transforms = T.Compose([
                    # Randomize(),
                    ToTensor(),
                    Resize(new_imgsize),
                    NormalizeRGB()
                    ])
            else:
                data_transforms = T.Compose([
                    ChangeBandOrder(),
                    Normalize(),
                    Randomize(),
                    ToTensor(),
                    Resize(new_imgsize)])
        else:
            if args.datatype == "rgb_unimodal":
                data_transforms = T.Compose([
                    # Randomize(),
                    ToTensor(),
                    Resize(new_imgsize),
                    # NormalizeRGB()
                    ])
            else:
                data_transforms = T.Compose([
                    ChangeBandOrder(),
                    Normalize(),
                    ToTensor(),
                    Resize(new_imgsize)])
    else:
        data_transforms = None

    data = SmokePlumeSegmentationDataset(args, data_path = data_path, mode = mode, mult=mult, reg_data = reg_data, channels=channels, size=size,
                            transform=data_transforms)

    return data


def init_segmentation_datasets(args):
    reg_data = pd.read_csv(os.path.join(args.tf_datapath, "labels.csv"))

    data_train_120x120 = create_dataset(args, data_path = args.tf_datapath, 
                                        mode='training/120x120/', 
                                        reg_data = reg_data, 
                                        mult=4, 
                                        train=True, 
                                        channels=args.tf_channels, 
                                        new_imgsize=args.img_size)

    data_train_300x300 = create_dataset(args, data_path = args.tf_datapath, 
                                        mode='training/300x300/', 
                                        reg_data = reg_data, 
                                        mult=4, 
                                        train=True, 
                                        channels=args.tf_channels, 
                                        size=300,
                                        new_imgsize=args.img_size)

    data_val = create_dataset(args, data_path = args.tf_datapath, 
                              mode='validation/', 
                              train=False, 
                              reg_data=reg_data, 
                              mult=1, 
                              channels=args.tf_channels,
                              new_imgsize=args.img_size)

    data_train = ConcatDataset([data_train_120x120, data_train_300x300])

    print("Number of training set: {}".format(len(data_train)))
    print("Number of validation set: {}".format(len(data_val)))

    # draw random subsamples
    train_sampler = RandomSampler(data_train, replacement=True, num_samples=int(2 * len(data_train) / 3))

    # initialize data loaders
    train_dl = DataLoader(data_train, batch_size=args.batch_size, num_workers=args.num_workers,
                          pin_memory=True, sampler=train_sampler)

    val_dl = DataLoader(data_val, batch_size=args.batch_size)
    return train_dl, val_dl
