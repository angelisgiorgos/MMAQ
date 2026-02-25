import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import rasterio as rio
from utils.data_utils import normalize_to_uint8


class PlantDataset(Dataset):
    def __init__(self, args, channels, size=None, data_path = "", mode='training/120x120/', reg_data=None, mult=1,
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
                    if self.reg_data is not None and len(self.reg_data[self.reg_data['filename'] == filename]) != 0 and sum(self.reg_data[self.reg_data['filename'] == filename]['gen_output'].isna()) == 0:
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
                        if self.reg_data is not None and len(self.reg_data[self.reg_data['filename'] == filename]) != 0 and sum(self.reg_data[self.reg_data['filename'] == filename]['gen_output'].isna()) == 0:
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
                img = normalize_to_uint8(img)
                
                
                img = Image.fromarray(img).resize((256, 256))
                sample["img"] = self.transform(img)
        else:
            if self.transform:
                sample = self.transform(sample)
        return sample

