import tensorflow as tf
import keras_tuner as kt

import utils
import models
from utils import read_config, Config
from dataset import prepare_and_split_data, download_and_extract_dataset

config = Config()
config = read_config()

download_and_extract_dataset(config.dataset_url, 
                                    config.download_dir, 
                                    config.extract_dir)

train_ds, val_ds, test_ds, all_labels = prepare_and_split_data(config)

# Prepare model
num_classes = len(all_labels)

tuner = kt.Hyperband(
models.build_tunable_model,
objective='val_accuracy',
max_epochs=50,
max_model_size=100000,
factor=3,  # Reduction factor
directory='tuner_results',
project_name='kws_hyperband',
max_consecutive_failed_trials=100, # Allow many skips
max_retries_per_trial=0,           # Don't waste time retrying an oversized model    
)

# View search space summary
tuner.search_space_summary()

# Start the search
tuner.search(
    train_ds,
    epochs=30,
    validation_data=val_ds,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ],
    verbose=1
)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
Best hyperparameters found:
- Conv1 filters: {best_hps.get('conv1_filters')}
- Conv2 filters: {best_hps.get('conv2_filters')}
- Conv3 filters: {best_hps.get('conv3_filters')}
- Dense units: {best_hps.get('dense_units')}
- Learning rate: {best_hps.get('learning_rate')}
- Use dropout: {best_hps.get('use_dropout')}
""")