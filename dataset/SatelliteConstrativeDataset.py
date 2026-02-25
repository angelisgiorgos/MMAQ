from typing import List, Tuple
import random
import numpy as np
import os
import copy
from torchvision.transforms import transforms
import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import xarray as xr
from dataset.transforms import SwapNoiseCorrupter



class SatelliteContrastiveDataset(Dataset):
    def __init__(self, args, data_tabular, augmentation: transforms.Compose, transforms=None, station_imgs=None):
        self.args = args
        self.data_tabular = data_tabular
        self.default_transform = transforms
        self.c = args.corruption_rate
        self.augmentation_rate = args.augmentation_rate
        self.augmentation = augmentation

        self.generate_marginal_distributions(self.data_tabular)
        self.samples, self.stations = self.load_data(self.data_tabular)
        self.noise_curropter =  SwapNoiseCorrupter(self.args.corruption_rate)
    

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

    def get_input_size(self) -> int:
        """
        Returns the number of fields in the table. 
        Used to set the input number of nodes in the MLP
        """
        if self.one_hot_tabular: 
            return int(sum(self.field_lengths_tabular))
        else:
            return len(self.data[0])

    def tabular_data(self, sample, norm=False):
        if norm:
            sample = self.augmentation(sample)
        one_hot_list = ["rural", "suburban", "urban", "traffic", "industrial", "background"]
        tabular = [sample["Altitude"],sample["PopulationDensity"]]
        for one_hot in one_hot_list:
            tabular.append(sample[one_hot])
        return np.array(tabular, dtype='float32')

    def corrupt(self, subject: List[float]) -> List[float]:
        """
        Creates a copy of a subject, selects the indices 
        to be corrupted (determined by hyperparam corruption_rate)
        and replaces their values with ones sampled from marginal distribution
        """
        subject = copy.deepcopy(subject)
        # print(subject)

        indices = random.sample(list(range(len(subject))), int(len(subject) * self.c))
        # print(indices)
        for i in indices:
            subject[i] = random.sample(self.marginal_distributions[i],k=1)[0] 
        return subject


    def generate_marginal_distributions(self, data_df: pd.DataFrame) -> None:
        """
        Generates empirical marginal distribution by transposing data
        """
        data_df = data_df[['Altitude',
                            'PopulationDensity', 'rural', 'suburban', 'urban', 'traffic',
                            'industrial', 'background']]
        self.marginal_distributions = data_df.transpose().values.tolist()

    def generate_imaging_views(self, sample) -> List[torch.Tensor]:
        """
        Generates two views of a subjects image. Also returns original image resized to required dimensions.
        The first is always augmented. The second has {augmentation_rate} chance to be augmented.
        """
        ims = []
        if random.random() < self.augmentation_rate:
            ims.append(self.augmentation(sample))
        else:
            ims.append(self.default_transform(sample))
        ims.append(self.augmentation(sample))

        orig_im = self.default_transform(sample)
        
        return ims, orig_im


    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
        sample = self.samples[index]

        if not self.args.linear_eval:
            if self.stations is not None:
                sample["img"] = self.stations.get(sample["AirQualityStation"])
                imaging_views, unaugmented_image = self.generate_imaging_views(sample)
            norm_tabular = self.tabular_data(sample, norm=True)
            tabular = self.tabular_data(sample, norm=False)
            normal_tabular = torch.from_numpy(norm_tabular).float()
            if self.args.tabular_transform == "marginal":
                tabular_views = [normal_tabular, torch.from_numpy(self.corrupt(norm_tabular)).float()]
            elif self.args.tabular_transform == "proposed":
                temp = norm_tabular[0].item()  # Save the value of the first element in a temporary variable
                norm_tabular[0] = norm_tabular[1]  # Set the first element to the value of the second element
                norm_tabular[1] = temp  # Set the second element to the value stored in the temporary variable
                remaining_values = norm_tabular[2:]
                flipped_remaining_values = np.flip(remaining_values)
                norm_tabular[2:] = flipped_remaining_values

                tabular_views = [torch.from_numpy(tabular).float(), torch.from_numpy(self.corrupt(norm_tabular)).float()]
            else:
                tabular_views = [normal_tabular, self.noise_curropter(normal_tabular).float()]
            sample = self.augmentation(sample)
            target = sample[self.args.measurements]
            return imaging_views, tabular_views, target, unaugmented_image
        else:
            if self.stations is not None:
                sample["img"] = self.stations.get(sample["AirQualityStation"])
            tabular = self.tabular_data(sample)
            normal_tabular = torch.from_numpy(tabular).float()
            sample = self.augmentation(sample)
            target = sample[self.args.measurements]
        return sample, normal_tabular, target


    def __len__(self) -> int:
        return len(self.data_tabular)

