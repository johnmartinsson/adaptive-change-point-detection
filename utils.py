import models
import oracles
import datasets

def shift_bit_length(x):
    x = int(x)
    return 1<<(x-1).bit_length()

def next_power_of_2(x):
    return shift_bit_length(x)

def get_active_dataset_by_config(config, data_dir):
    print("Getting active dataset ...")
    return datasets.ActiveSoundEventDetectionDataset(config, data_dir)

def get_active_learner_by_config(config):
    print("Getting active learner ...")
    return models.BaseActiveLearner()

def get_oracle_by_config(config):
    print("Getting oracle by config ...")
    return oracles.BaseOracle()

def get_test_dataset_by_config(config, data_dir):
    print("Getting test dataset ...")
    return datasets.SoundEventDetectionDataset(config, data_dir)
