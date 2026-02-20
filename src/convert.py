import os
import numpy as np
import tensorflow as tf
from dataset import prepare_and_split_data, download_and_extract_dataset
from utils import read_config, Config

def test_tflite_model(tflite_model_path, test_data, test_labels):
    """
    Evaluates a quantized TensorFlow Lite model on test data.

    Args:
        tflite_model_path (str): Path to the .tflite model file.
        test_data (np.ndarray): Numpy array of test input data (e.g., MFCC features). Shape: [num_samples, frames, coeffs]
        test_labels (np.ndarray): Numpy array of true labels corresponding to test_data.

    Returns:
        None. Prints the accuracy of the model on the test set.
    """
    import numpy as np
    import tensorflow as tf

    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Quantization parameters
    input_scale, input_zero_point = input_details[0]['quantization']

    # Expected input shape (including batch)
    expected_shape = input_details[0]['shape']  # e.g., [1, 50, 40]

    correct = 0
    total = len(test_data)

    for i in range(total):
        input_data = test_data[i]

        # Add batch dimension
        input_data = np.expand_dims(input_data, axis=0)  # shape becomes [1, 50, 40]

        # Add channel dimension if needed
        if input_data.shape != tuple(expected_shape):
            # For models expecting a channel, e.g., [1, 50, 40, 1]
            if len(expected_shape) == 4 and expected_shape[-1] == 1:
                input_data = np.expand_dims(input_data, axis=-1)

        # Final check
        if input_data.shape != tuple(expected_shape):
            raise ValueError(f"Input shape mismatch. Got {input_data.shape}, expected {expected_shape}")

        # Quantize
        input_int8 = np.round(input_data / input_scale + input_zero_point)
        input_int8 = np.clip(input_int8, -128, 127).astype(np.int8)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_int8)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])
        pred_label = np.argmax(output, axis=1)[0]

        if pred_label == test_labels[i]:
            correct += 1

    accuracy = correct / total
    print(f"TFLite model accuracy: {accuracy:.4f}")

config = Config()
config = read_config()

download_and_extract_dataset(config.dataset_url, 
                                    config.download_dir, 
                                    config.extract_dir)

train_ds, _, test_ds, all_labels = prepare_and_split_data(config)

test_data = []
test_labels = []

for x, y in test_ds.as_numpy_iterator():  # convert batch tensors to NumPy arrays
    test_data.append(x)
    test_labels.append(y)

# Concatenate all batches into single arrays
test_data = np.concatenate(test_data, axis=0)
test_labels = np.concatenate(test_labels, axis=0)

def representative_dataset():
    # Take 100-500 samples; 1000 is often more than needed but safe
    for sample, _ in train_ds.unbatch().take(500):
        # 1. Ensure float32 for the quantization process
        sample = tf.cast(sample, tf.float32)
        
        # 2. Handle the specific shaping logic
        # Assuming your input needs to be (Batch, Height, Width, Channels)
        # or (1, 49, 40, 1) based on your comments.
        
        if sample.shape == (99, 50, 40):
            # Example slice: adjustments to get it toward (49, 40)
            sample = sample[0, :50, :40] 

        # 3. Ensure it has the Channel dimension (Height, Width, 1)
        if len(sample.shape) == 2:
            sample = tf.expand_dims(sample, axis=-1)
            
        # 4. Ensure it has the Batch dimension (1, Height, Width, 1)
        if len(sample.shape) == 3:
            sample = tf.expand_dims(sample, axis=0)
            
        yield [sample]

# Test the representative_dataset function
for input_value in representative_dataset():
    print("Input shape:", input_value[0].shape) 
    break  # Only check the first sample

model_path = "saved_models/model_20260219-094550.keras"

model = tf.keras.models.load_model(model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

# 🔑 THIS IS THE KEY PART
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_quant_model)

# Compare size 
original_size =  os.path.getsize(model_path)
quantized_size = os.path.getsize('model.tflite')

print(f"Original model size: {original_size / 1e6:.2f} MB")
print(f"Quantized model size: {quantized_size / 1e6:.2f} MB")
print(f"Size reduction: {(original_size - quantized_size) / original_size * 100:.1f}%")

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()

print(input_details[0]['dtype'])  # FLOAT32 or INT8?

loss, accuracy = model.evaluate(test_data, test_labels, batch_size=32)
print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")

# Test converted model
test_tflite_model('model.tflite', test_data, test_labels)