import matplotlib.pyplot as plt
from utils import read_config, Config, save_training_plots, save_confusion_matrix
from dataset import prepare_and_split_data, download_and_extract_dataset
from mfcc_light import MFCC
import tensorflow as tf
import numpy as np

config = Config()
config = read_config()

download_and_extract_dataset(config.dataset_url, config.download_dir, config.extract_dir)

train_ds, val_ds, test_ds, all_labels = prepare_and_split_data(config)

batch_audio, batch_labels = next(iter(train_ds.take(3)))

mfcc = MFCC()

# Select the first sample in the batch
audio = batch_audio[15]
print(len(audio))

a = audio.numpy()
a = (a * 32767.0).astype(np.int16)
mfcc.set_signal(a)
mfcc.compute_coefficient()
features = mfcc.get_coefficient()

print(features)