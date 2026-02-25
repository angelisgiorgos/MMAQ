import torch
import os
import numpy as np
import random
from torch.utils.data import Dataset
import rasterio as rio
import torchvision.transforms as T
from .transforms import ChangeBandOrder
from lightly.transforms.utils import IMAGENET_NORMALIZE
from rasterio.plot import reshape_as_image
from PIL import Image


class PlantDataset(Dataset):
    def __init__(self, args, channels, size=None, data_path = None, mode='training/120x120/', reg_data=None, mult=1,
                 transform=None):
        super().__init__()

        self.args = args

        fuel_type_dict = {
            'Fossil Brown coal/Lignite': 0,
            'Fossil Hard coal': 1,
            'Fossil Gas': 2,
            'Fossil Peat': 3,
            'Fossil Coal-derived gas': 2,
            'Fossil Oil': 3}

        self.datadir = os.path.join(os.path.join(data_path, 'images'), mode)
        self.reg_data = reg_data
        self.transform = transform

        self.mult = mult

        # read in image file names for positive images
        self.read_imaages(fuel_type_dict)

        self.channels = channels
        


    def read_imaages(self, fuel_type_dict):

        # list of image files, labels (positive or negative), segmentation
        # label vector edge coordinates
        self.imgfiles = []
        self.fossil_type = []

         # list of indices of positive and negative images
        self.positive_indices = []
        self.negative_indices = []

        idx = 0
        for root, _, files in os.walk(self.datadir):
            for filename in files:
                if not filename.endswith('.tif'):
                    continue

                if 'positive' in root :
                    if len(self.reg_data[self.reg_data['filename'] == filename]) != 0 and sum(self.reg_data[self.reg_data['filename'] == filename]['gen_output'].isna()) == 0:
                        self.positive_indices.append(idx)
                        self.imgfiles.append(os.path.join(root, filename))
                        self.fossil_type.append(fuel_type_dict[self.reg_data[self.reg_data['filename'] == filename]['fuel_type'].unique()[0]])
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
                            self.imgfiles.append(os.path.join(root, filename))
                            self.fossil_type.append(fuel_type_dict[self.reg_data[self.reg_data['filename'] == filename]['fuel_type'].unique()[0]])
                            idx += 1
            
            # turn lists into arrays
        self.imgfiles = np.array(self.imgfiles)
        self.fossil_type = np.array(self.fossil_type).astype(np.float32)
        self.positive_indices = np.array(self.positive_indices)
        self.negative_indices = np.array(self.negative_indices)

        if self.mult > 1:
            self.imgfiles = np.array([*self.imgfiles] * self.mult)
            self.positive_indices = np.array([*self.positive_indices] * self.mult)
            self.negative_indices = np.array([*self.negative_indices] * self.mult)
            self.fossil_type = np.array([*self.fossil_type] * self.mult)

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
        
        sample = {
            'idx': idx,
            'img': imgdata,
            'labels': self.fossil_type[idx],
            'imgfile': self.imgfiles[idx]
        }

        if self.args.datatype == "rgb_unimodal":
            # apply transformations
            if self.transform:
                img = sample["img"].transpose(1, 2, 0)
                img = self.normalize_to_uint8(img)
                
                
                img = Image.fromarray(img).resize((256, 256))
                sample["img"] = self.transform(img)
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
               'labels': torch.tensor(sample['labels']),
               'img': torch.from_numpy(sample['img'].copy()).float(),
               'imgfile': sample['imgfile']}

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

        # mirror horizontally
        func = Mirror()
        imgdata = func(imgdata)
        # flip vertically
        func = Flip()
        imgdata = func(imgdata)
        # rotate by [0,1,2,3]*90 deg
        func = Rotate()
        imgdata = func(imgdata)

        return {'idx': sample['idx'],
                'img': imgdata.copy(),
                'labels': sample['labels'],
                'imgfile': sample['imgfile']}


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample.get("img").clone()


        resized_img = T.functional.resize(img, size=(self.size, self.size),
                                          interpolation=T.InterpolationMode.BICUBIC)

        out = {}
        for k, v in sample.items():
            if k == "img":
                out[k] = resized_img.float()
            else:
                out[k] = v
        return out


class Normalize(object):
    """Normalize pixel values to zero mean and range [-1, +1] measured in
    standard deviations."""
    def __init__(self, channels):
        self.channels_means = np.array(
            [960.97437, 1110.9012, 1250.0942, 1259.5178, 1500.98,
             1989.6344, 2155.846, 2251.6265, 2272.9438, 2442.6206,
             1914.3, 1512.0585])
        self.channels_stds = np.array(
            [1302.0157, 1418.4988, 1381.5366, 1406.7112, 1387.4155, 1438.8479,
             1497.8815, 1604.1998, 1516.532, 1827.3025, 1303.83, 1189.9052])

        self.channel_means = self.channels_means[channels]
        self.channel_stds = self.channels_stds[channels]


    def __call__(self, sample):
        """
        :param sample: sample to be normalized
        :return: normalized sample
        """
        sample['img'] = (sample['img'] - self.channel_means.reshape(
            sample['img'].shape[0], 1, 1)) / self.channel_stds.reshape(
            sample['img'].shape[0], 1, 1)
        return sample


class Mirror(object):
    """Mirror image."""
    def __init__(self, p=0.5, always_apply=False):
        self.p = p
        self.always_apply = always_apply

    def __call__(self, imgdata):
        if self.always_apply:
            self.p = 0
        if self.p < random.random():
            imgdata = np.flip(imgdata, 2)
        return imgdata


class Flip(object):
    """Flip image."""
    def __init__(self, p=0.5, always_apply=False):
        self.p = p
        self.always_apply = always_apply

    def __call__(self, imgdata):
        if self.always_apply:
            self.p = 0
        if self.p < random.random():
            imgdata = np.flip(imgdata, 1)
        return imgdata


class Rotate(object):
    """Rotate image."""
    def __init__(self, p=0.5, always_apply=False):
        self.p = p
        self.always_apply = always_apply

    def __call__(self, imgdata):
        if self.always_apply:
            self.p = 0
        if self.p < random.random():
            rot = np.random.randint(0, 4)
            imgdata = np.rot90(imgdata, rot, axes=(1, 2))
        return imgdata



def create_dataset(args, data_path, mode, reg_data, mult, apply_transforms=True, train=False, size=120, new_imgsize=120,
                   channels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]):
    """Create a dataset; uses same input parameters as PowerPlantDataset.
    :param apply_transforms: if `True`, apply available transformations
    :return: data set"""
    if apply_transforms:
        if train:
            if args.datatype == "rgb_unimodal":
                data_transforms = T.Compose([
                T.Resize(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
                ])
            else:
                data_transforms = T.Compose([
                    ChangeBandOrder(),
                    Normalize(np.array(channels)),
                    Randomize(),
                    ToTensor(),
                    Resize(new_imgsize)])
                
        else:
            if args.datatype == "rgb_unimodal":
                data_transforms = T.Compose([
                    T.Resize(224),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"])])
            else:
                data_transforms = T.Compose([
                    ChangeBandOrder(),
                    Normalize(np.array(channels)),
                    ToTensor(),
                    Resize(new_imgsize)])

    data = PlantDataset(args, channels=channels, size=size, data_path = data_path, mode=mode, reg_data=reg_data, mult=mult,
                            transform=data_transforms)
    return data