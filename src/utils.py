import os
import toml
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

""" Read training paramters """
@dataclass
class Config:
    dataset_url: str = ""
    download_dir: str = ""
    extract_dir: str = ""
    augmented: bool = False
    subset_labels: list[str] = field(default_factory=list)
    epoch: int = 0
    batch: int = 0
    val_ratio: float = 0.0
    test_ratio: float = 0.0

def read_config(config_file):
    config = Config()

    with open(config_file, 'r') as f:
        file = toml.load(f)
    
    # Copy configuration param from TOML
    config.dataset_url = file['DATA']['dataset_url']
    config.download_dir = file['DATA']['download_dir']
    config.extract_dir = config.download_dir + "/speech_commands"
    config.augmented = file['DATA']['augmented']
    config.subset_labels = file['DATA']['subset_labels']
    config.epoch = file['LEARNING']['epoch']
    config.batch = file['LEARNING']['batch']
    config.val_ratio = file['LEARNING']['val_ratio']
    config.test_ratio = file['LEARNING']['test_ratio']

    return config

# Plotting functions

def save_training_plots(history, output_path_prefix):
    """
    Saves training and validation accuracy/loss plots as a PNG image.

    Args:
        history (dict): Dictionary containing training history.
        output_path_prefix (str): Path prefix for saving the image (without extension).
    """

    # Ensure the directory exists
    output_dir = os.path.dirname(output_path_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Save image
    plt.tight_layout()
    plt.savefig(f"{output_path_prefix}_training_plot.png")
    plt.close()

def save_confusion_matrix(model, test_ds, output_path_prefix, class_names=None):
    """
    Computes and saves the confusion matrix as a PNG image.

    Args:
        model (tf.keras.Model): Trained model.
        test_ds (tf.data.Dataset): Batched test dataset.
        output_path_prefix (str): Path prefix for saving the image.
        class_names (list, optional): List of class names for labels.
    """
    y_true, y_pred = [], []

    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch, verbose=0)
        y_pred_batch = np.argmax(preds, axis=1)
        y_true.extend(y_batch.numpy())
        y_pred.extend(y_pred_batch)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(f"{output_path_prefix}_confusion_matrix.png")
    plt.close()