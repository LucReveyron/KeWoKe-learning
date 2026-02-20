import os
import json
import datetime
import tensorflow as tf
from tensorflow.keras import callbacks

import models
from utils import read_config, Config, save_training_plots, save_confusion_matrix
from dataset import prepare_and_split_data, download_and_extract_dataset

config = Config()
config = read_config()

download_and_extract_dataset(config.dataset_url, config.download_dir, config.extract_dir)

train_ds, val_ds, test_ds, all_labels = prepare_and_split_data(config)

# Prepare model
num_classes = len(all_labels)


model = models.kws_cnn_microspeech()

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-3, clipnorm=1.0) # Other model work with 1e-4 (1e-4 for featuremodel)
model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

lr_scheduler = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-6
)

early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    train_ds,
    epochs=config.epoch,
    validation_data=val_ds,
    callbacks=[lr_scheduler, early_stop],
    verbose=1   
)   

test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.3f}")

# Save model and training steps
run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_dir = f"saved_models"
history_json_path = f"training_history/history_{run_name}.json"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(os.path.dirname(history_json_path), exist_ok=True)

model.save(f"{model_dir + f'/model_{run_name}'}.keras")

with open(history_json_path, "w") as f:
    json.dump(history.history, f)

output_prefix = f"saved_images/metrics_{run_name}"

try:
    save_training_plots(history.history, output_prefix)
except Exception as e:
    print(f"Warning: Failed to save training plots: {e}")

try:
    save_confusion_matrix(model, test_ds, output_prefix, class_names=all_labels)
except Exception as e:
    print(f"Warning: Failed to save confusion matrix: {e}")
