import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def split_samples(samples, stations, test_size=0.2, val_size=0.2):
    stations_train, stations_test = train_test_split(stations, test_size=test_size)
    real_val_size = val_size / (1 - test_size)
    stations_train, stations_val = map(set, train_test_split(stations_train, test_size=real_val_size))
    stations_test = set(stations_test)

    samples_train = [s for s in samples if s["AirQualityStation"] in stations_train]
    samples_test = [s for s in samples if s["AirQualityStation"] in stations_test]
    samples_val = [s for s in samples if s["AirQualityStation"] in stations_val]
    
    final_dict = {
        'samples':{'train':samples_train,
                   'val': samples_val,
                   'test':samples_test
                   },
        'stations':{'train':stations_train,
                   'val': stations_val,
                   'test':samples_val
                   }
        }
    return final_dict


def split_samples_df(samples, test_size=0.2):
    stations = samples["AirQualityStation"].drop_duplicates().to_numpy()

    stations_train, stations_test = train_test_split(
        stations,
        test_size=test_size,
        random_state=42,
        shuffle=True
    )

    samples_train = samples[samples["AirQualityStation"].isin(stations_train)]
    samples_test = samples[samples["AirQualityStation"].isin(stations_test)]

    return samples_train, samples_test


def normalize_for_display(band_data):
    from rasterio.plot import reshape_as_image
    import numpy as np
    
    band_data = reshape_as_image(np.array(band_data))
    lower_perc = np.percentile(band_data, 2, axis=(0, 1))
    upper_perc = np.percentile(band_data, 98, axis=(0, 1))
    return (band_data - lower_perc) / (upper_perc - lower_perc)


def normalize_to_uint8(data, min_percentile=2, max_percentile=98):
    
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
