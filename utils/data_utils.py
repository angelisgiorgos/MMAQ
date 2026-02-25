import pandas as pd
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
    """split pd.DF s.t. all samples of a given station
    are either in the train or test set """
    stations = samples.AirQualityStation.unique()
    stations_train, stations_test = train_test_split(stations, test_size=test_size)

    samples_train = samples[samples.AirQualityStation.isin(stations_train)]
    samples_test = samples[samples.AirQualityStation.isin(stations_test)]

    return samples_train, samples_test
