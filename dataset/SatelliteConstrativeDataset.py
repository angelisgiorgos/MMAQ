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
from transformers import AutoTokenizer


class SatelliteContrastiveDataset(Dataset):
    def __init__(self, args, data_tabular, augmentation: transforms.Compose,
                 default_transform=None, station_imgs=None):
        self.args = args
        self.data_tabular = data_tabular
        self.default_transform = default_transform
        self.augmentation = augmentation
        self.c = args.corruption_rate
        self.augmentation_rate = args.augmentation_rate
        self.noise_curropter = SwapNoiseCorrupter(self.c)
        self.generate_marginal_distributions(self.data_tabular)
        self.samples, self.stations = self.load_data(self.data_tabular)

    # ============================================================
    # ---------------------- DATA LOADING -------------------------
    # ============================================================
    def load_data(self, sample_file: pd.DataFrame):
        print("Available columns from samples_file:")
        print(sample_file.columns)
        samples = []
        stations = {}
        for station in tqdm(sample_file.AirQualityStation.unique()):
            station_obs = sample_file[sample_file.AirQualityStation == station]
            s5p_path = station_obs.s5p_path.unique().item()
            s5p_data = xr.open_dataset(
                os.path.join(self.args.dataroot, "sentinel-5p", s5p_path)
            ).rio.write_crs(4326)
            for idx in station_obs.index.values:
                sample = sample_file.loc[idx].to_dict()
                sample["idx"] = idx
                sample["s5p"] = (
                    s5p_data.tropospheric_NO2_column_number_density
                    .values.squeeze()
                )
                samples.append(sample)
                stations[sample["AirQualityStation"]] = np.float32(
                    np.load(
                        os.path.join(
                            self.args.dataroot,
                            "sentinel-2",
                            sample["img_path"]
                        )
                    )
                )
            s5p_data.close()
        return samples, stations

    # ============================================================
    # ---------------------- TABULAR LOGIC ------------------------
    # ============================================================
    def tabular_to_numeric(self, sample, norm=False):
        """
        Returns numeric tabular vector.
        """
        if norm:
            sample = self.augmentation(sample)

        one_hot_list = [
            "rural", "suburban", "urban",
            "traffic", "industrial", "background"
        ]
        tabular = [
            sample["Altitude"],
            sample["PopulationDensity"]
        ]
        for key in one_hot_list:
            tabular.append(sample[key])
        return np.array(tabular, dtype="float32")

    def convert_tabular_to_text(self, sample: dict) -> str:
        """
        Converts structured tabular metadata into natural language.
        Activated via --tabular_as_text.
        """

        # Area type
        if sample["rural"] == 1:
            area_type = "rural"
        elif sample["suburban"] == 1:
            area_type = "suburban"
        elif sample["urban"] == 1:
            area_type = "urban"
        else:
            area_type = "unknown"

        # Station type
        if sample["traffic"] == 1:
            station_type = "traffic monitoring"
        elif sample["industrial"] == 1:
            station_type = "industrial monitoring"
        elif sample["background"] == 1:
            station_type = "background monitoring"
        else:
            station_type = "unknown monitoring"

        altitude = float(sample["Altitude"])
        pop_density = float(sample["PopulationDensity"])
        country = sample.get("Countrycode", "unknown country")

        # Semantic population density bucket
        if pop_density < 100:
            density_level = "low population density"
        elif pop_density < 1000:
            density_level = "medium population density"
        else:
            density_level = "high population density"

        text = (
            f"{area_type.capitalize()} area in {country}. "
            f"{station_type.capitalize()} station located at "
            f"{altitude:.1f} meters altitude with {density_level} "
            f"({pop_density:.1f} inhabitants per km²)."
        )

        return text

    def get_tabular_views(self, sample):
        """
        Returns either:
        - two numeric tabular tensors
        - or two text descriptions
        depending on argparse flag.
        """

        # ================= TEXT MODE =================
        if self.args.tabular_as_text:
            text = self.convert_tabular_to_text(sample)
            return [text, text]

        # ================= NUMERIC MODE ==============
        norm_tabular = self.tabular_to_numeric(sample, norm=True)
        tabular = self.tabular_to_numeric(sample, norm=False)

        normal_tabular = torch.from_numpy(norm_tabular).float()

        if self.args.tabular_transform == "marginal":
            return [
                normal_tabular,
                torch.from_numpy(self.corrupt(norm_tabular)).float()
            ]

        elif self.args.tabular_transform == "proposed":

            # Swap first two
            temp = norm_tabular[0].item()
            norm_tabular[0] = norm_tabular[1]
            norm_tabular[1] = temp

            # Flip rest
            norm_tabular[2:] = np.flip(norm_tabular[2:])

            return [
                torch.from_numpy(tabular).float(),
                torch.from_numpy(self.corrupt(norm_tabular)).float()
            ]

        else:
            return [
                normal_tabular,
                self.noise_curropter(normal_tabular).float()
            ]

    # ============================================================
    # ---------------------- CORRUPTION ---------------------------
    # ============================================================

    def corrupt(self, subject: List[float]) -> List[float]:
        subject = copy.deepcopy(subject)
        indices = random.sample(
            list(range(len(subject))),
            int(len(subject) * self.c)
        )
        for i in indices:
            subject[i] = random.sample(
                self.marginal_distributions[i], k=1
            )[0]

        return subject

    def generate_marginal_distributions(self, data_df: pd.DataFrame):

        data_df = data_df[
            [
                "Altitude",
                "PopulationDensity",
                "rural", "suburban", "urban",
                "traffic", "industrial", "background"
            ]
        ]

        self.marginal_distributions = (
            data_df.transpose().values.tolist()
        )

    # ============================================================
    # ---------------------- IMAGING ------------------------------
    # ============================================================
    def generate_imaging_views(self, sample):
        ims = []
        if random.random() < self.augmentation_rate:
            ims.append(self.augmentation(sample))
        else:
            ims.append(self.default_transform(sample))

        ims.append(self.augmentation(sample))
        orig_im = self.default_transform(sample)
        return ims, orig_im

    # ============================================================
    # ---------------------- GET ITEM -----------------------------
    # ============================================================
    def __getitem__(self, index):
        sample = self.samples[index]
        # Attach image
        if self.stations is not None:
            sample["img"] = self.stations.get(
                sample["AirQualityStation"]
            )

        # ================== SSL TRAINING ==================
        if self.args.task == "pretrain":
            imaging_views, unaugmented_image = (
                self.generate_imaging_views(sample)
            )
            tabular_views = self.get_tabular_views(sample)
            sample = self.augmentation(sample)
            target = sample[self.args.measurements]
            return imaging_views, tabular_views, target, unaugmented_image

        # ================== LINEAR EVAL ==================
        else:
            if self.args.tabular_as_text:
                tabular = self.convert_tabular_to_text(sample)
            else:
                tabular = torch.from_numpy(
                    self.tabular_to_numeric(sample)
                ).float()

            sample = self.augmentation(sample)
            target = sample[self.args.measurements]

            return sample, tabular, target

    # ============================================================
    def __len__(self):
        return len(self.samples)


class DatasetTextCollate:
    def __init__(self, tokenizer_name="bert-base-uncased", max_length=64):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __call__(self, batch):

        # PRETRAIN MODE
        if isinstance(batch[0][0], list):

            imaging_views, tab_views, targets, orig_imgs = zip(*batch)

            # ---------- Imaging ----------
            im_view1 = torch.stack([v[0]["img"] for v in imaging_views])
            im_view2 = torch.stack([v[1]["img"] for v in imaging_views])

            s5_view1 = torch.stack([v[0]["s5p"] for v in imaging_views])
            s5_view2 = torch.stack([v[1]["s5p"] for v in imaging_views])

            # ---------- Tabular ----------
            if isinstance(tab_views[0][0], str):
                texts1 = [v[0] for v in tab_views]
                texts2 = [v[1] for v in tab_views]

                enc1 = self.tokenizer(
                    texts1,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )

                enc2 = self.tokenizer(
                    texts2,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )

                tab1 = enc1
                tab2 = enc2

            else:
                tab1 = torch.stack([v[0] for v in tab_views])
                tab2 = torch.stack([v[1] for v in tab_views])

            targets = torch.tensor(targets).float()

            return (
                im_view1, im_view2,
                s5_view1, s5_view2,
                tab1, tab2,
                targets
            )

        # LINEAR EVAL MODE
        else:
            samples, tabular, targets = zip(*batch)

            imgs = torch.stack([s["img"] for s in samples])
            s5 = torch.stack([s["s5p"] for s in samples])

            if isinstance(tabular[0], str):
                enc = self.tokenizer(
                    list(tabular),
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                tab = enc
            else:
                tab = torch.stack(tabular)

            targets = torch.tensor(targets).float()

            return imgs, s5, tab, targets