import tensorflow as tf

def get_model_ops(model_path):
    # Load the model and get details
    interpreter = tf.lite.Interpreter(model_path='model.tflite')
    interpreter.allocate_tensors()
    
    # Get all operators in the model
    nodes = interpreter._get_ops_details()
    
    # Extract unique op names
    unique_ops = set()
    for node in nodes:
        unique_ops.add(node['op_name'])
        
    print("--- Required Kernels for ESP32 ---")
    for op in sorted(unique_ops):
        print(f"- {op}")

get_model_ops("model.tflite")