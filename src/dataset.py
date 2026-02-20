import os
import re
import tarfile
import hashlib
import random
import urllib.request
from pathlib import Path
import numpy as np
import soundfile as sf
import tensorflow as tf
from mfcc_light import MFCC

mfcc = MFCC()

# Constants 
MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
SAMPLING_FREQUENCY = 16000 # Hz

# Save to a local directory
dataset_path = "./my_saved_dataset"

def download_and_extract_dataset(url, download_path, extract_path):
    """
    Downloads and extracts a dataset from a specified URL.

    Args:
        url (str): URL pointing to the dataset archive (e.g., a .zip or .tar.gz file).
        download_path (str): Directory path where the downloaded archive will be saved.
        extract_path (str): Directory path where the contents of the archive will be extracted.

    Returns:
        None
    """
    
    # Create directories if they don't exist
    os.makedirs(download_path, exist_ok=True)
    os.makedirs(extract_path, exist_ok=True)

    filename = url.split('/')[-1]
    file_path = os.path.join(download_path, filename)
    folder_name = os.path.basename(os.path.normpath(extract_path))

    # Download the file if it doesn't exist
    if not os.path.exists(file_path):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, file_path)
        print("Download complete.")
    else:
        print(f"{filename} already downloaded.")

    # Extract the dataset
    print(f"Extracting {filename}...")
    if Path(folder_name).is_dir():
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
        print(f"Extraction completed to {extract_path}")
    else:
        print(f"{folder_name} already present.")

def prepare_and_split_data(config):
    """
    Return a training, validation and test dataset
    
    :param config: structure of parameters
    """
    # Check if config updated
    if(has_toml_changed() or not datasets_exist(dataset_path)):
        # Find all available labels
        all_folders = [f for f in os.listdir(config.extract_dir) if os.path.isdir(os.path.join(config.extract_dir, f))]
        all_labels = [label for label in all_folders if label != "_background_noise_"]

        known_labels = config.subset_labels
        unknown_labels = [label for label in all_labels if label not in config.subset_labels]
        known_files = get_files_and_labels(known_labels, config.extract_dir)
        unknown_files = get_files_and_labels(unknown_labels, config.extract_dir, unknown=True)

        # Balance unknown as any other label 
        unknown_files = sample_unknown(unknown_files, round(len(known_files) / len(known_labels)))

        # Split 
        known_files.extend(unknown_files)
        known_labels.extend(["unknown"])
        labels = sorted(known_labels)
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = split_dataset(known_files, config)

        # Quantify labels
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        print(f"Labels used for training: {label_to_index}")
        train_labels_int = [label_to_index[label] for label in train_labels]
        val_labels_int = [label_to_index[label] for label in val_labels]
        test_labels_int = [label_to_index[label] for label in test_labels]

        train_paths, train_labels, train_labels_int = augment_data(config, train_paths, train_labels, label_to_index)

        # Read audio files and organise them in a tf.data.Dataset
        train_ds = prepare_dataset(train_paths, train_labels_int, is_training=True, batch_size=config.batch)
        val_ds = prepare_dataset(val_paths, val_labels_int, is_training=False, batch_size=config.batch)
        test_ds = prepare_dataset(test_paths, test_labels_int, is_training=False, batch_size=config.batch)

        save_all_datasets(dataset_path, train_ds, val_ds, test_ds)
        
    else:
        train_ds, val_ds, test_ds = load_all_datasets(dataset_path)
        labels = config.subset_labels + ["unknown"]

    return train_ds, val_ds, test_ds, labels

def prepare_dataset(filenames, labels, 
                    is_training=True, 
                    batch_size=32):

    ds = tf.data.Dataset.from_tensor_slices((filenames, labels))
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds


def preprocess(filename, label):
    """
    Converts a 16000-sample audio into `num_shifts`
    downsampled signals of length 160 using phase shifts.

    Output shape:
        frames -> (num_shifts, 160)
        labels -> (num_shifts,)
    """

    def _py_fn(a):
        a = a.numpy()
        #a = (a * 32767.0).astype(np.int16) # Beause we load raw 
        mfcc.set_signal(a)
        mfcc.compute_coefficient()
        features = mfcc.get_coefficient()
        return features.astype(np.int16)
    
    audio = decode_audio(filename)
    # Compute MFCC without breaking tensorflow graph
    features = tf.py_function(_py_fn, [audio], np.int16)

    # Normalization
    features = tf.cast(features, tf.float32)
    #mean = tf.reduce_mean(features, axis=0, keepdims=True)
    #std  = tf.math.reduce_std(features, axis=0, keepdims=True)# + 1e-6
    #features = (features - mean) / std
    #Mean: -2337.468
    #Std: 8648.558
    #features = (features + 2337.468) / 8648.558

    # Restore shape 
    features.set_shape([50, 40])
    
    return features, label

def decode_audio(filename):
    #audio_binary = tf.io.read_file(filename)
    #audio, _ = tf.audio.decode_wav(audio_binary) # No more float [-1;1]
    # filename is a tf.Tensor of type string
    def _read_file(f):
        audio, sr = sf.read(f.numpy().decode("utf-8"), dtype='int16')
        return audio

    audio = tf.py_function(_read_file, [filename], tf.int16)
    #audio = tf.squeeze(audio, axis=-1)

    # Get current length
    audio_len = tf.shape(audio)[0]

    # If too short → pad with zeros
    audio = tf.cond(
        audio_len < SAMPLING_FREQUENCY,
        lambda: tf.pad(audio, [[0, SAMPLING_FREQUENCY - audio_len]]),
        lambda: audio[:SAMPLING_FREQUENCY]  # If too long → trim
    )

    return audio

# Inspired from the README of the Speech Commands Data Set v0.01
def which_set_wrapper(filename, validation_percentage, testing_percentage):
    """
    Determines the dataset split (training, validation, or testing) for a given audio filename,
    based on a deterministic hash of the filename. This ensures consistent data splits.

    Args:
        filename (str): Path to the audio file.
        validation_percentage (float): Percentage of data to use for validation (0–100).
        testing_percentage (float): Percentage of data to use for testing (0–100).

    Returns:
        str: One of 'training', 'validation', or 'testing', indicating the split assignment.
    """
    base_name = os.path.basename(filename)
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    # Encode the string to bytes here before hashing:
    hash_name_bytes = hash_name.encode('utf-8')
    hash_name_hashed = hashlib.sha1(hash_name_bytes).hexdigest()
    
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (MAX_NUM_WAVS_PER_CLASS + 1)) *
                       (100.0 / MAX_NUM_WAVS_PER_CLASS))
    
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result

def get_files_and_labels(labels_subset, extract_dir, unknown=False):
    """
    Docstring pour get_files_and_labels
    
    :param labels_subset: List of labels to extract
    :param extract_dir: Path of data
    :param unknown: force unknown labels or not
    """
    subset_files = []
    for label in labels_subset:
        folder_path = os.path.join(extract_dir, label)
        if os.path.exists(folder_path):
            files = tf.io.gfile.glob(os.path.join(folder_path, '*.wav'))
            if(unknown is True):
                subset_files.extend([(f, "unknown") for f in files])
            else:
                subset_files.extend([(f, label) for f in files])
        else:
            print(f"Warning: Label folder '{label}' not found!")
    print(f"Total files in subset: {len(subset_files)}")

    return subset_files

def sample_unknown(unknown_list, target_count):
    """
    Return a random subset of the unkown to match known counts per split
    
    :param unknown_list: list of files and labels
    :param target_count: number of files per labels
    """
    # Shuffle unknown files once for all splits
    random.shuffle(unknown_list)
     # If unknown_list is smaller, just use all
    return unknown_list[:target_count] if len(unknown_list) >= target_count else unknown_list

def split_dataset(files, config):
    # Split known files by set type
    train_paths = []
    train_labels = []
    val_paths = []
    val_labels = []
    test_paths = []
    test_labels = []

    for file_path, label in files:
        set_type = which_set_wrapper(file_path, config.val_ratio, config.test_ratio)
        if set_type == 'training':
            train_paths.append(file_path)
            train_labels.append(label)
        elif set_type == 'validation':
            val_paths.append(file_path)
            val_labels.append(label)
        else:
            test_paths.append(file_path)
            test_labels.append(label)
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels

def augment_data(config, train_paths, train_labels, label_to_index):

    # TODO: Add to config
    augment_factor = 1
    snr_range = (15, 25)  # SNR in dB
    aug_ratio = 0.1       # 10% augmentation

    # Load background noises for augmentation
    bg_noise_dir = os.path.join(config.extract_dir, "_background_noise_")
    bg_noises = load_bg_noises(bg_noise_dir)

    # Prepare augmentation output directory
    augmented_dir = os.path.join(config.extract_dir, "augmented_train")

    if config.augmented:
        print("Starting data augmentation...")
        aug_files, aug_labels = augment_dataset_with_noise(
            train_paths, train_labels, bg_noises, augmented_dir, augment_factor, snr_range
        )
        print(f"Generated {len(aug_files)} augmented samples")

        # Sample 10% of original training size from augmented data
        num_aug_samples = int(len(train_paths) * aug_ratio)

        combined = list(zip(aug_files, aug_labels))
        random.shuffle(combined)
        aug_files_subset, aug_labels_subset = zip(*combined[:num_aug_samples])

        train_paths += list(aug_files_subset)
        train_labels += list(aug_labels_subset)

        # Final shuffle
        combined_final = list(zip(train_paths, train_labels))
        random.shuffle(combined_final)
        train_paths, train_labels = zip(*combined_final)
        train_paths, train_labels = list(train_paths), list(train_labels)

        # Update integer labels after augmentation
        train_labels_int = [label_to_index[label] for label in train_labels]

        return train_paths, train_labels, train_labels_int

def load_bg_noises(bg_noise_folder):
    """
    Loads background noise WAV files from a folder.

    Args:
        bg_noise_folder (str or Path): Path to the folder containing background noise .wav files.

    Returns:
        list of tuples: Each tuple contains (audio_samples, sample_rate) as (np.ndarray, int).
    """
    noises = []
    for f in Path(bg_noise_folder).glob("*.wav"):
        samples, sr = sf.read(f)
        if samples.ndim > 1:
            samples = np.mean(samples, axis=1)  # convert to mono
        samples = samples.astype(np.float32)
        noises.append((samples, sr))
    return noises

def get_random_noise_clip(noises, length):
    """
    Extracts a random noise segment of the specified length.

    Args:
        noises (list): A list of tuples (audio_samples, sample_rate) from load_bg_noises().
        length (int): Desired length of the noise clip in samples.

    Returns:
        np.ndarray: A 1D array containing a noise segment of the given length.
    """
    noise, sr = random.choice(noises)
    if len(noise) < length:
        # loop noise to make sure it's long enough
        repeats = int(np.ceil(length / len(noise)))
        noise = np.tile(noise, repeats)
    start_idx = random.randint(0, len(noise) - length)
    return noise[start_idx:start_idx + length]

def mix_with_noise(speech, noise, snr_db):
    """
    Mixes a speech signal with background noise at a specified SNR.

    Normalizes the noise, adjusts its power to achieve the desired signal-to-noise ratio, 
    and mixes it with the speech signal.

    Args:
        speech (np.ndarray): The original speech signal.
        noise (np.ndarray): The background noise signal.
        snr_db (float): Desired Signal-to-Noise Ratio in decibels.

    Returns:
        np.ndarray: The mixed audio signal, clipped between -1.0 and 1.0.
    """
    if np.max(np.abs(noise)) > 0:
        noise = noise / np.max(np.abs(noise))  # normalize

    if len(noise) != len(speech):
        noise = noise[:len(speech)] if len(noise) > len(speech) else np.pad(noise, (0, len(speech) - len(noise)))

    speech_power = np.mean(speech ** 2)
    noise_power = np.mean(noise ** 2)
    desired_noise_power = speech_power / (10 ** (snr_db / 10))
    scale_factor = np.sqrt(desired_noise_power / (noise_power + 1e-10))
    
    augmented = speech + scale_factor * noise
    return np.clip(augmented, -1.0, 1.0)

def augment_dataset_with_noise(train_files, train_labels, bg_noises, augmented_dir,
                               augment_factor=2, snr_range=(15, 25)):
    """
    Augments a dataset by adding background noise at varying SNR levels.

    For each audio file, generates multiple noisy versions using random background 
    noises and signal-to-noise ratios, then saves them to disk.

    Args:
        train_files (list of str): List of paths to clean audio files.
        train_labels (list): Corresponding labels for the audio files.
        bg_noises (list): List of background noises from load_bg_noises().
        augmented_dir (str or Path): Directory where augmented audio will be saved.
        augment_factor (int): Number of noisy samples to generate per original file.
        snr_range (tuple): Range of SNR values (min, max) in dB for augmentation.

    Returns:
        tuple: (augmented_files, augmented_labels)
            - augmented_files (list of str): Paths to the augmented audio files.
            - augmented_labels (list): Corresponding labels for the augmented files.
    """
    os.makedirs(augmented_dir, exist_ok=True)
    
    augmented_files = []
    augmented_labels = []
    
    for filepath, label in zip(train_files, train_labels):
        audio, sr = sf.read(filepath)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)  # mono
        audio = audio.astype(np.float32)
        
        for i in range(augment_factor):
            shifted_audio = time_shift(audio, shift_max=0.1)
            noise_clip = get_random_noise_clip(bg_noises, len(audio))
            snr_db = random.uniform(*snr_range)
            augmented_audio = mix_with_noise(shifted_audio, noise_clip, snr_db)
            
            orig_stem = Path(filepath).stem
            new_filename = f"{orig_stem}_aug{i}.wav"
            save_path = Path(augmented_dir) / new_filename
            
            sf.write(save_path, augmented_audio, sr)
            
            augmented_files.append(str(save_path))
            augmented_labels.append(label)
    
    return augmented_files, augmented_labels

def time_shift(audio, shift_max=0.1):
    """
    Shifts the audio randomly in time.

    Args:
        audio (np.ndarray): 1D audio signal.
        shift_max (float): Maximum fraction of total length to shift (+/-).

    Returns:
        np.ndarray: Time-shifted audio.
    """
    shift_amount = int(random.uniform(-shift_max, shift_max) * len(audio))
    if shift_amount > 0:
        audio_shifted = np.pad(audio, (shift_amount, 0), mode='constant')[:len(audio)]
    elif shift_amount < 0:
        audio_shifted = np.pad(audio, (0, -shift_amount), mode='constant')[-shift_amount:len(audio)-shift_amount]
    else:
        audio_shifted = audio
    return audio_shifted


def datasets_exist(base_path):
    """
    Returns True only if all three splits exist and appear valid.
    """
    subsets = ["train", "validation", "test"]
    
    for s in subsets:
        subdir = os.path.join(base_path, s)
        # Check if directory exists AND if the TF metadata file is present
        metadata_file = os.path.join(subdir, "dataset_spec.pb")
        if not os.path.exists(metadata_file):
            return False
            
    return True

def save_all_datasets(base_path, train_ds, val_ds, test_ds):
    """
    Saves the three dataset splits into the specified base directory.
    """
    # Ensure the base directory exists
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        
    paths = {
        "train": os.path.join(base_path, "train"),
        "validation": os.path.join(base_path, "validation"),
        "test": os.path.join(base_path, "test")
    }

    print(f"💾 Saving datasets to {base_path}...")
    tf.data.Dataset.save(train_ds, paths["train"])
    tf.data.Dataset.save(val_ds, paths["validation"])
    tf.data.Dataset.save(test_ds, paths["test"])
    print("✅ Save complete.")

def load_all_datasets(base_path):
    """
    Loads the train, validation, and test datasets from the base directory.
    Returns: (train_ds, val_ds, test_ds)
    """
    paths = {
        "train": os.path.join(base_path, "train"),
        "validation": os.path.join(base_path, "validation"),
        "test": os.path.join(base_path, "test")
    }

    print(f"📂 Loading datasets from {base_path}...")
    
    # Load each subset
    train_ds = tf.data.Dataset.load(paths["train"])
    val_ds   = tf.data.Dataset.load(paths["validation"])
    test_ds  = tf.data.Dataset.load(paths["test"])
    
    print("✅ Datasets loaded successfully.")
    return train_ds, val_ds, test_ds

def has_toml_changed():
    TOML_FILE = "run_config.toml"
    CACHE_FILE = ".last_update_check"
    # 1. Get current modification time
    current_mtime = os.path.getmtime(TOML_FILE)

    # 2. Check for the cache file
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            last_mtime = float(f.read())
        
        if current_mtime > last_mtime:
            changed = True
        else:
            changed = False
    else:
        # First time running the script
        changed = True

    # 3. Update the cache file for next time
    with open(CACHE_FILE, "w") as f:
        f.write(str(current_mtime))
        
    return changed