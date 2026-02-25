from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import os
import xarray as xr
from torchvision.transforms import transforms
from PIL import Image
from rasterio.plot import reshape_as_image


class UniModalRGBDataset(Dataset):
    def __init__(self, args, data_tabular, augmentation: transforms.Compose, target_augmentation=None, station_imgs=None):
        self.args = args
        self.data_tabular = data_tabular
        self.samples, self.stations = self.load_data(self.data_tabular)
        self.target_augmentation = target_augmentation
        self.augmentation = augmentation

        self.resize_transform = transforms.Resize((256, 256))


    def load_data(self, sample_file: pd.DataFrame):
        print("Available columns from samples_file :")
        print(sample_file.columns)
        samples = []
        stations = {}
        for station in tqdm(sample_file.AirQualityStation.unique()):
            station_obs = sample_file[sample_file.AirQualityStation == station]
            s5p_path = station_obs.s5p_path.unique().item()
            s5p_data = xr.open_dataset(os.path.join(self.args.dataroot, "sentinel-5p", s5p_path)).rio.write_crs(4326)

            for idx in station_obs.index.values:
                sample = sample_file.loc[idx].to_dict() # select by index value, not position
                sample["idx"] = idx
                sample["s5p"] = s5p_data.tropospheric_NO2_column_number_density.values.squeeze()
                samples.append(sample)
                stations[sample["AirQualityStation"]] = np.float32(np.load(os.path.join(self.args.dataroot, "sentinel-2", sample["img_path"])))
            
            s5p_data.close()
        return samples, stations

    def _normalize_for_display(self, band_data):
        band_data = reshape_as_image(np.array(band_data))
        lower_perc = np.percentile(band_data, 2, axis=(0,1))
        upper_perc = np.percentile(band_data, 98, axis=(0,1))
        return (band_data - lower_perc) / (upper_perc - lower_perc)
        

    def __getitem__(self, index: int):
        sample = self.samples[index]
        if self.stations is not None:
            sample["img"] = self.stations.get(sample["AirQualityStation"])
            sample["img"] = self._normalize_for_display(sample["img"]).transpose(0, 2, 1)

            #extract RGB bands
            rgb_image = sample["img"][:, :, [0, 1, 2]]
            rgb_image = (rgb_image * 255).astype(np.uint8)
            rgb_image = Image.fromarray(rgb_image).resize((256, 256))
            # rgb_image = torch.tensor(rgb_image).permute(2, 0, 1)
            # rgb_image = self.resize_transform(rgb_image)

            rgb_samples = self.augmentation(rgb_image)

            if self.target_augmentation is not None:
                sample = self.target_augmentation(sample)
            target = sample[self.args.measurements]
            return rgb_samples, target
        

    def __len__(self) -> int:
        return len(self.data_tabular)