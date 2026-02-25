# This file contains the data handling infrastructure for the segmentation
# model.

import os
import numpy as np
import json
import cv2
import rasterio as rio
from rasterio.features import rasterize
from shapely.geometry import Polygon
from torch.utils.data import Dataset

# data directory
outdir = os.path.abspath('.')

from utils.data_utils import normalize_to_uint8


class SmokePlumeSegmentationDataset(Dataset):
    """SmokePlumeSegmentation dataset class."""

    def __init__(self, args, 
                 channels, size=None, data_path = "", mode='training/120x120/', reg_data=None, mult=1,
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
                    if self.reg_data is not None and len(self.reg_data[self.reg_data['filename'] == filename]) != 0 and sum(self.reg_data[self.reg_data['filename'] == filename]['gen_output'].isna()) == 0:
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
                    if self.reg_data is not None and len(self.reg_data[self.reg_data['filename'] == filename]) != 0 and sum(self.reg_data[self.reg_data['filename'] == filename]['gen_output'].isna()) == 0:
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

    def __getitem__(self, idx):
        """Read in image data, preprocess, build segmentation mask, and apply
        transformations."""

        imgfile = rio.open(self.imgfiles[idx])
        imgdata = np.array([imgfile.read(i) for i in
                            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]])

        # if self.args.datatype == "rgb_unimodal":
            # imgdata = imgdata.transpose(1, 2, 0)
            # imgdata = normalize_for_display(imgdata).transpose(0, 2, 1)
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
                img = normalize_to_uint8(img)
                sample["img"] = img.transpose(2, 0, 1).astype(np.float32)
                sample = self.transform(sample)
        else:
            if self.transform:
                sample = self.transform(sample)

        return sample


