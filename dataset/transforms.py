import os
from typing import Tuple, Union
os.environ["OMP_NUM_THREADS"] = "6"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "6"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "6"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

import random
import numpy as np
import torch
from rasterio.plot import reshape_as_image
import torchvision.transforms as T
import skimage.transform as transform
import torch.nn.functional as F
import torchvision
from scipy.ndimage.filters import gaussian_filter
from torchvision.transforms import functional as TF



# define image transforms
class ChangeBandOrder(object):
    def __call__(self, sample):
        """necessary if model was pre-trained on .npy files of BigEarthNet and should be used on other Sentinel-2 images

        move the channels of a sentinel2 image such that the bands are ordered as in the BigEarthNet dataset
        input image is expected to be of shape (200,200,12) with band order:
        ['B04', 'B03', 'B02', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'B01', 'B09'] (i.e. like my script on compute01 produces)

        output is of shape (12,120,120) with band order:
        ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"] (order in BigEarthNet .npy files)
        """
        img = sample["img"].copy()
        if not img.shape[0] == 12:
            img = np.moveaxis(img, -1, 0)
        reordered_img = np.zeros(img.shape)
        reordered_img[0, :, :] = img[10, :, :]
        reordered_img[1, :, :] = img[2, :, :]
        reordered_img[2, :, :] = img[1, :, :]
        reordered_img[3, :, :] = img[0, :, :]
        reordered_img[4, :, :] = img[4, :, :]
        reordered_img[5, :, :] = img[5, :, :]
        reordered_img[6, :, :] = img[6, :, :]
        reordered_img[7, :, :] = img[3, :, :]
        reordered_img[8, :, :] = img[7, :, :]
        reordered_img[9, :, :] = img[11, :, :]
        reordered_img[10, :, :] = img[8, :, :]
        reordered_img[11, :, :] = img[9, :, :]

        if img.shape[1] != 120 or img.shape[2] != 120:
            reordered_img = reordered_img[:, 40:160, 40:160]

        out = {}
        for k, v in sample.items():
            if k == "img":
                out[k] = reordered_img
            else:
                out[k] = v
        return out


class CustomColorJitter(T.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(brightness, contrast, saturation, hue)

    def __call__(self, sample):
        img = sample.get("img").clone()
        # Assume x is a torch tensor with shape [C, H, W]
        # Apply color jittering separately to each band
        for i in range(img.size(0)):
            img[i] = super().__call__(img[i].unsqueeze(0)).squeeze(0)
        
        out = {}
        for k, v in sample.items():
            if k == "img":
                out[k] = img
            else:
                out[k] = v
        return out

 

class ToTensor(object):
    def __call__(self, sample):
        img = torch.from_numpy(sample["img"].copy())
        img = img.to(torch.float32)

        if sample.get("no2") is not None:
            no2 = torch.from_numpy(sample["no2"].copy())
            no2 = no2.to(torch.float32)
        if sample.get("o3") is not None:
            o3 = torch.from_numpy(sample["o3"].copy())
            o3 = o3.to(torch.float32)
        if sample.get("co") is not None:
            co = torch.from_numpy(sample["co"].copy())
            co = co.to(torch.float32)
        if sample.get("so2") is not None:
            so2 = torch.from_numpy(sample["so2"].copy())
            so2 = so2.to(torch.float32)
        if sample.get("pm10") is not None:
            pm10 = torch.from_numpy(sample["pm10"].copy())
            pm10 = pm10.to(torch.float32)


        if sample.get("s5p") is not None:
            s5p = torch.from_numpy(sample["s5p"].copy())
            s5p = s5p.to(torch.float32)

        out = {}
        for k, v in sample.items():
            if k == "img":
                out[k] = img
            elif k == "no2":
                out[k] = no2
            elif k == "o3":
                out[k] = o3
            elif k == "co":
                out[k] = co
            elif k == "so2":
                out[k] = so2
            elif k == "pm10":
                out[k] = pm10
            elif k == "s5p":
                out[k] = s5p
            else:
                out[k] = v

        return out



class DatasetStatistics(object):
    def __init__(self):
        self.channel_means = np.array([340.76769064, 429.9430203, 614.21682446,
                590.23569706, 950.68368468, 1792.46290469, 2075.46795189, 2218.94553375,
                2266.46036911, 2246.0605464, 1594.42694882, 1009.32729131])

        self.channel_std = np.array([554.81258967, 572.41639287, 582.87945694,
                675.88746967, 729.89827633, 1096.01480586, 1273.45393088, 1365.45589904,
                1356.13789355, 1302.3292881, 1079.19066363, 818.86747235])

        # statistics over the whole of Europe from Sentinel-5P products in 2018-2020:
        # l3_mean_europe_2018_2020_005dg.netcdf mean 1.51449095e+15 std 6.93302798e+14
        # l3_mean_europe_large_2018_2020_005dg.netcdf mean 1.23185273e+15 std 7.51052046e+14
        self.s5p_mean = 1.23185273e+15      #this will need to be updated if I change the samples file!!!
        self.s5p_std = 7.51052046e+14       #this will need to be updated if I change the samples file!!!

        #altitude values
        self.alt_mean = 246.9503722
        self.alt_std = 255.961989

        #popdense values
        self.popdense_mean = 435.719603
        self.popdense_std = 1049.972678

        # values for averages from 2018-2020 per EEA station, across stations
        self.no2_mean = 17.38866556 #updated to 3poll dataset
        self.no2_std =  8.99506794 #updated to 3poll dataset

        self.o3_mean = 54.7168629 #updated to 3poll dataset
        self.o3_std = 11.17569622 #updated to 3poll dataset

        self.co_mean = 0.337909296
        self.co_std = 0.199646936

        self.so2_mean = 4.632686498
        self.so2_std = 3.4378752763

        self.pm10_mean = 21.30139834 #updated to 3poll dataset
        self.pm10_std = 8.454853402 #updated to 3poll dataset



class Normalize(object):
    """normalize a sample, i.e. the image and NO2 value, by subtracting mean and dividing by std"""
    def __init__(self, statistics):
        self.statistics = statistics

    def __call__(self, sample):
        img = reshape_as_image(sample.get("img").copy())
        img = np.moveaxis((img - self.statistics.channel_means) / self.statistics.channel_std, -1, 0)

        if sample.get("no2") is not None:
            no2 = sample.get("no2")#.copy()
            no2 = np.array((no2 - self.statistics.no2_mean) / self.statistics.no2_std)
            #no2 = np.array((no2 - 0) / 1)

        if sample.get("o3") is not None:
            o3 = sample.get("o3")  # .copy()
            o3 = np.array((o3 - self.statistics.o3_mean) / self.statistics.o3_std)
            #o3 = np.array((o3 - 0) / 1)

        if sample.get("co") is not None:
            co = sample.get("co")  # .copy()
            co = np.array((co - self.statistics.co_mean) / self.statistics.co_std)
            #co = np.array((co - 0) / 1)

        if sample.get("so2") is not None:
            so2 = sample.get("so2")  # .copy()
            so2 = np.array((so2 - self.statistics.so2_mean) / self.statistics.so2_std)
            #so2 = np.array((so2 - 0) / 1)

        if sample.get("pm10") is not None:
            pm10 = sample.get("pm10")  # .copy()
            pm10 = np.array((pm10 - self.statistics.pm10_mean) / self.statistics.pm10_std)
            #pm10 = np.array((pm10 - 0) / 1)

        if sample.get("s5p") is not None:
            #print("Sentinel5p online")
            #print(sample.get("s5p"))
            s5p = sample.get("s5p").copy()
            s5p = np.array((s5p - self.statistics.s5p_mean) / self.statistics.s5p_std)

        if sample.get("Altitude") is not None:
            alt = sample.get("Altitude")#.copy()
            alt = np.array((alt - self.statistics.alt_mean) / self.statistics.alt_std)

        if sample.get("PopulationDensity") is not None:
            PopulationDensity = sample.get("PopulationDensity")#.copy()
            PopulationDensity = np.array((PopulationDensity - self.statistics.popdense_mean) / self.statistics.popdense_std)

        out = {}
        for k,v in sample.items():
            if k == "img":
                out[k] = img
            elif k == "no2":
                out[k] = no2
            elif k == "o3":
                out[k] = o3
            elif k == "co":
                out[k] = co
            elif k == "so2":
                out[k] = so2
            elif k == "pm10":
                out[k] = pm10
            elif k == "s5p":
                out[k] = s5p
            elif k == "Altitude":
                out[k] = alt
            elif k == "PopulationDensity":
                out[k] = PopulationDensity
            else:
                out[k] = v

        return out

    @staticmethod
    def undo_no2_standardization(statistics, no2):
        return (no2 * statistics.no2_std) + statistics.no2_mean
    @staticmethod
    def undo_o3_standardization(statistics, o3):
        return (o3 * statistics.o3_std) + statistics.o3_mean
    @staticmethod
    def undo_co_standardization(statistics, co):
        return (co * statistics.co_std) + statistics.co_mean
    @staticmethod
    def undo_so2_standardization(statistics, so2):
        return (so2 * statistics.so2_std) + statistics.so2_mean
    @staticmethod
    def undo_pm10_standardization(statistics, pm10):
        return (pm10 * statistics.pm10_std) + statistics.pm10_mean


class RandomRotate:
    """Implementation of random rotation.

    Randomly rotates an input image by a fixed angle. By default, we rotate
    the image by 90 degrees with a probability of 50%.

    This augmentation can be very useful for rotation invariant images such as
    in medical imaging or satellite imaginary.

    Attributes:
        prob:
            Probability with which image is rotated.
        angle:
            Angle by which the image is rotated. We recommend multiples of 90
            to prevent rasterization artifacts. If you pick numbers like
            90, 180, 270 the tensor will be rotated without introducing
            any artifacts.

    """

    def __init__(self, prob: float = 0.5, angle: int = 90):
        self.prob = prob
        self.angle = angle

    def __call__(self, sample):
        """Rotates the image with a given probability.

        Args:
            image:
                PIL image or tensor which will be rotated.

        Returns:
            Rotated image or original image.

        """
        img = sample.get("img").clone()
        prob = np.random.random_sample()
        if prob < self.prob:
            rot_img = TF.rotate(img, self.angle)
        else:
            rot_img = img
        if sample.get("s5p") is not None:
            s5p_available = True
            if len(sample["s5p"].shape) == 2:
                s5p = sample["s5p"].clone().unsqueeze(0)
            else:
                s5p = sample["s5p"].clone()
            if prob < self.prob:
                rot_s5p = TF.rotate(s5p, self.angle)
            else:
                rot_s5p = s5p
        
        out = {}
        for k, v in sample.items():
            if k == "img":
                out[k] = rot_img
            elif k == "s5p":
                out[k] = rot_s5p
            elif k == "no2":
                out[k] = sample.get("no2")
            else:
                out[k] = v
        return out



class RandomRotateDegrees(object):
    def __init__(self, prob: float, degrees: Union[float, Tuple[float, float]]):
        self.transform = T.RandomApply([T.RandomRotation(degrees=degrees)], p=prob)

    def __call__(self, sample):
        img = sample.get("img").clone()
        transformed_img = self.transform(img)

        s5p_available = False
        if sample.get("s5p") is not None:
            s5p_available = True
            if len(sample["s5p"].shape) == 2:
                s5p = sample["s5p"].clone().unsqueeze(0)
            else:
                s5p = sample["s5p"].clone()
            transformed_s5p = self.transform(s5p)

        out = {}
        for k, v in sample.items():
            if k == "img":
                out[k] = transformed_img
            elif k == "s5p":
                out[k] = transformed_s5p
            elif k == "no2":
                out[k] = sample.get("no2")
            else:
                out[k] = v
        return out


def random_rotation_transform(
    rr_prob: float,
    rr_degrees: Union[None, float, Tuple[float, float]],
) -> Union[RandomRotate, T.RandomApply]:
    if rr_degrees is None:
        # Random rotation by 90 degrees.
        return RandomRotate(prob=rr_prob, angle=90)
    else:
        # Random rotation with random angle defined by rr_degrees.
        return RandomRotateDegrees(prob=rr_prob, degrees=rr_degrees)


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample.get("img").clone()

        if sample.get("s5p") is not None:
            if len(sample["s5p"].shape) == 2:
                s5p = sample["s5p"].clone().unsqueeze(0)
            else:
                s5p = sample["s5p"].clone()

        if sample.get("fpt") is not None:
            if not isinstance(sample.get("fpt"), torch.Tensor):
                fpt = torch.from_numpy(sample.get("fpt")).clone().unsqueeze(0)
            else:
                fpt = sample.get("fpt").clone().unsqueeze(0)
            resized_fpt = T.functional.resize(fpt, size=(self.size, self.size),
                                              interpolation=T.InterpolationMode.NEAREST)


        resized_img = T.functional.resize(img, size=(self.size, self.size),
                                          interpolation=T.InterpolationMode.BICUBIC)

        if sample.get("s5p") is not None:
            resized_s5p = T.functional.resize(s5p, size=(self.size, self.size),
                                              interpolation=T.InterpolationMode.BICUBIC)

        out = {}
        for k, v in sample.items():
            if k == "img":
                out[k] = resized_img
            elif k == "fpt":
                out[k] = resized_fpt
            elif k == "no2":
                out[k] = sample.get("no2")
            elif k == "s5p":
                out[k] = resized_s5p
            else:
                out[k] = v
        return out


class RandomHorizontalFlip(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self, sample):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        img = sample.get("img").clone()
        s5p_available = False
        if sample.get("s5p") is not None:
            s5p_available = True
            s5p = sample["s5p"].clone()
            if torch.rand(1) < self.p:
                s5p = torchvision.transforms.functional.hflip(s5p)
        if torch.rand(1) < self.p:
            img = torchvision.transforms.functional.hflip(img)
        out = {}
        for k, v in sample.items():
            if k == "img":
                out[k] = img
            elif k == "s5p":
                out[k] = s5p
            else:
                out[k] = v

        return out


class RandomVerticalFlip(torch.nn.Module):
    """Vertically flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self, sample):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        img = sample.get("img").clone()
        if sample.get("s5p") is not None:
            s5p = sample["s5p"].clone()
            if torch.rand(1) < self.p:
                s5p = torchvision.transforms.functional.vflip(s5p)
        if torch.rand(1) < self.p:
            img = torchvision.transforms.functional.vflip(img)
        out = {}
        for k, v in sample.items():
            if k == "img":
                out[k] = img
            elif k == "s5p":
                out[k] = s5p
            else:
                out[k] = v

        return out
    

class JigsawAugmentation:
    def __init__(self, num_patches=4):
        self.num_patches = num_patches
        self.patch_size = 12 // num_patches  # Assuming 12 bands

    def perform_jigsaw(self, x):
        patches = []
        if x.dim() == 3:
            for i in range(self.num_patches):
                for j in range(self.num_patches):
                    patch = x[:, i * self.patch_size:(i + 1) * self.patch_size, j * self.patch_size:(j + 1) * self.patch_size]
                    patches.append(patch)
        elif x.dim() == 2:
            for i in range(self.num_patches):
                for j in range(self.num_patches):
                    patch = x[i * self.patch_size:(i + 1) * self.patch_size, j * self.patch_size:(j + 1) * self.patch_size]
                    patches.append(patch)

        # Shuffle the patches
        permuted_order = torch.randperm(self.num_patches**2)
        shuffled_patches = [patches[i] for i in permuted_order]

        # Reassemble the image
        augmented_image = torch.cat(shuffled_patches, dim=1)

        return augmented_image

    def __call__(self, sample):
        # x is assumed to be a tensor with shape [C, H, W], where C is the number of bands

        img = sample.get("img").clone()

        jig_img = self.perform_jigsaw(img)
        if sample.get("s5p") is not None:
            s5p = sample["s5p"].clone()
            s5p = self.perform_jigsaw(s5p)
        out = {}
        for k, v in sample.items():
            if k == "img":
                out[k] = jig_img
            elif k == "s5p":
                out[k] = s5p
            else:
                out[k] = v
        return out

        



class Randomize():
    def __call__(self, sample):
        img = sample.get("img").copy()

        s5p_available = False
        if sample.get("s5p") is not None:
            s5p_available = True
            s5p = sample["s5p"].copy()

        if random.random() > 0.5:
            img = np.flip(img, 1)
            if s5p_available: s5p = np.flip(s5p, 0)
        if random.random() > 0.5:
            img = np.flip(img, 2)
            if s5p_available: s5p = np.flip(s5p, 1)
        if random.random() > 0.5:
            img = np.rot90(img, np.random.randint(0, 4), axes=(1, 2))
            if s5p_available: s5p = np.rot90(s5p, np.random.randint(0, 4), axes=(0, 1))

        out = {}
        for k, v in sample.items():
            if k == "img":
                out[k] = img
            elif k == "s5p":
                out[k] = s5p
            else:
                out[k] = v

        return out


class ToGray(object):
    def __init__(self, out_channels):
        self.out_channels = out_channels

    def gray_img(self, sample):
        gray_img = np.mean(sample, axis=-1)
        gray_img = np.tile(gray_img, (self.out_channels, 1, 1))
        gray_img = np.transpose(gray_img, [1, 2, 0])
        return gray_img

    def __call__(self,sample):
        img = sample["img"].copy()
        gray_img = self.gray_img(img)
        out = {}
        for k, v in sample.items():
            if k == "img":
                out[k] = gray_img
            else:
                out[k] = v
        return out


class LightenTransform(object):
    """
    Transformation which adds a mean band average from respective bands,
    scaled by a scaling factor.
    """
    def __init__(self, scaling: float=0.1):
        self.scaling = scaling
        

    def bnd_avrg(self, data: np.ndarray):
        self.per_band_average = np.average(data, axis=0)

    def __call__(self, sample):
        img = sample["img"].copy()
        self.bnd_avrg(img)
        transformed_img = np.zeros(img.shape)
        transformed_img = img + (self.per_band_average * self.scaling)

        s5p_available = False
        if sample.get("s5p") is not None:
            s5p_available = True
            s5p = sample["s5p"].copy()
            self.bnd_avrg(s5p)
            transformed_s5p = np.zeros(s5p.shape)
            transformed_s5p = s5p + (self.per_band_average * self.scaling)

        out = {}
        for k, v in sample.items():
            if k == "img":
                out[k] = transformed_img
            elif k == "s5p":
                out[k] = transformed_s5p
            else:
                out[k] = v
        return out


class DarkenTransform(object):
    """
    Transformation which subtracts a mean band average from respective bands,
    scaled by a scaling factor.
    """
    def __init__(self, scaling: float=0.1):
        self.scaling = scaling
        

    def bnd_avrg(self, data: np.ndarray):
        self.per_band_average = np.average(data, axis=0)

    def __call__(self, sample):
        img = sample["img"].copy()
        self.bnd_avrg(img)
        transformed_img = np.zeros(img.shape)
        transformed_img = img - (self.per_band_average * self.scaling)

        s5p_available = False
        if sample.get("s5p") is not None:
            s5p_available = True
            s5p = sample["s5p"].copy()
            self.bnd_avrg(s5p)
            transformed_s5p = np.zeros(s5p.shape)
            transformed_s5p = s5p - (self.per_band_average * self.scaling)

        out = {}
        for k, v in sample.items():
            if k == "img":
                out[k] = transformed_img
            elif k == "s5p":
                out[k] = transformed_s5p
            else:
                out[k] = v
        return out
    

class RandomChannelDrop(object):
    """ Random Channel Drop """
    
    def __init__(self, min_n_drop=1, max_n_drop=8):
        self.min_n_drop = min_n_drop
        self.max_n_drop = max_n_drop

    def drop_channels(self, sample):
        n_channels = random.randint(self.min_n_drop, self.max_n_drop)
        channels = np.random.choice(range(sample.shape[0]), size=n_channels, replace=False)

        for c in channels:
            sample[c, :, :] = 0        
        return sample  

    def __call__(self, sample):
        img = sample["img"].copy()
        dropped_img = self.drop_channels(img)

        out = {}
        for k, v in sample.items():
            if k == "img":
                out[k] = dropped_img
            else:
                out[k] = v
        return out
          


class RandomFlipTransform(object):
    def __init__(self, flip_orientation: int = None):
        if flip_orientation is None:
            self.flip_orientation = random.randint(1, 2)

    def __call__(self, sample):
        img = sample["img"].copy()
        flipped_img = np.zeros(img.shape)
        flipped_img = np.flip(img, self.flip_orientation)


        s5p_available = False
        if sample.get("s5p") is not None:
            s5p_available = True
            s5p = sample["s5p"].copy()
            flipped_s5p = np.zeros(s5p.shape)
            flipped_s5p = np.flip(s5p, int(self.flip_orientation-1))
        
        out = {}
        for k, v in sample.items():
            if k == "img":
                out[k] = flipped_img
            elif k == "s5p":
                out[k] = flipped_s5p
            else:
                out[k] = v
        return out


    
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, sample):
        img = sample["img"].copy()
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        gaus_img = gaussian_filter(img,sigma)

        s5p_available = False
        if sample.get("s5p") is not None:
            s5p_available = True
            s5p = sample["s5p"].copy()
            gaus_s5p = gaussian_filter(s5p,sigma)
        out = {}
        for k, v in sample.items():
            if k == "img":
                out[k] = gaus_img
            elif k == "s5p":
                out[k] = gaus_s5p
            else:
                out[k] = v
        return out



class RandomBrightness(object):
    """ Random Brightness """

    def __init__(self, brightness=0.4):
        self.brightness = brightness

    def __call__(self, sample):
        img = sample["img"].copy()
        s = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        bright_img = img * s
        s5p_available = False
        if sample.get("s5p") is not None:
            s5p_available = True
            s5p = sample["s5p"].copy()
            bright_s5p = s5p * s
        
        out = {}
        for k, v in sample.items():
            if k == "img":
                out[k] = bright_img
            elif k == "s5p":
                out[k] = bright_s5p
            else:
                out[k] = v
        return out


class RandomResizeCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)):
        self.random_crop = T.RandomResizedCrop(size, scale, ratio)
    
    def __call__(self, sample):
        img = sample.get("img").clone()
        cropped_img = self.random_crop(img)

        if sample.get("s5p") is not None:
            s5p_available = True
            if len(sample["s5p"].shape) == 2:
                s5p = sample["s5p"].clone().unsqueeze(0)
            else:
                s5p = sample["s5p"].clone()
            cropped_s5p = self.random_crop(s5p)

        out = {}
        for k, v in sample.items():
            if k == "img":
                out[k] = cropped_img
            elif k == "s5p":
                out[k] = cropped_s5p
            else:
                out[k] = v
        return out
        


class RandomContrast(object):
    """ Random Contrast """

    def __init__(self, contrast=0.4):
        self.contrast = contrast

    def __call__(self, sample):
        img = sample["img"].copy()
        s = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        mean = np.mean(img, axis=(0, 1))
        contrast_img = ((img - mean) * s + mean)

        s5p_available = False
        if sample.get("s5p") is not None:
            s5p_available = True
            s5p = sample["s5p"].copy()
            mean = np.mean(s5p, axis=(0, 1))
            contrast_s5p = ((s5p - mean) * s + mean)
        
        out = {}
        for k, v in sample.items():
            if k == "img":
                out[k] = contrast_img
            elif k == "s5p":
                out[k] = contrast_s5p
            else:
                out[k] = v
        return out


class UpScaleTransform(object):
    """
    Transformation which upscales a given sample, and then crops it to the
    original size.
    """
    def __init__(self, scale: float=1.25):
        self.scale = scale

    @staticmethod
    def _crop_center(image: np.ndarray, x_size, y_size):
        y, x = image.shape[1:3]
        startx = x // 2 - (x_size // 2)
        starty = y // 2 - (y_size // 2)
        return image[:, starty:starty + y_size, startx:startx + x_size]

    def upscale_sample(self, data, size_x, size_y):
        scaled = np.zeros(data.shape)
        scaled_data = transform.rescale(data, scale=self.scale, channel_axis=0)
        scaled_data = self._crop_center(scaled_data, size_x,
                                               size_y)
        return scaled_data


    def __call__(self, sample):
        img = sample["img"].copy()
        original_img_size_x, original_img_size_y = img.shape[1:3]
        scaled_img = self.upscale_sample(img, original_img_size_x, original_img_size_y)
        s5p_available = False
        if sample.get("s5p") is not None:
            s5p_available = True
            s5p = sample["s5p"].copy()
            original_s5p_x, original_s5p_y = s5p.shape
            s5p = np.expand_dims(s5p, axis=0)
            scaled_s5p = self.upscale_sample(s5p, original_s5p_x, original_s5p_y)
            scaled_s5p = np.squeeze(scaled_s5p, axis=0)
        out = {}
        for k, v in sample.items():
            if k == "img":
                out[k] = scaled_img
            elif k == "s5p":
                out[k] = scaled_s5p
            else:
                out[k] = v
        return out


class SwapNoiseCorrupter(object):
    """Apply swap noise on the input data.

    Each data point has specified chance be replaced by a random value from the same column.
    """

    def __init__(self, probas):
        super().__init__()
        self.probas = torch.from_numpy(np.array(probas))

    def forward(self, x):
        should_swap = torch.bernoulli(self.probas.to(x.device) * torch.ones(x.shape).to(x.device))
        corrupted_x = torch.where(should_swap == 1, x[torch.randperm(x.shape[0])], x)
        mask = (corrupted_x != x).float()
        return corrupted_x