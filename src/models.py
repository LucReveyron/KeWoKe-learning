import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def kws_cnn_microspeech(input_shape=(50, 40), num_classes=12):
    inputs = layers.Input(shape=input_shape)
    x = layers.Reshape((input_shape[0], input_shape[1], 1))(inputs)

    # ---- Conv block 1 ----
    x = layers.Conv2D(16, (8, 4), strides=(2, 2), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # ---- Block 2 ----
    x = layers.Conv2D(32, (5, 5), strides=(2, 2), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    
    # ---- Block 3 ----
    x = layers.Conv2D(16, (3, 3), padding="same", use_bias=False)(x)  # x = 96 channels
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # ---- Global pooling (TFLM friendly) ----
    x = layers.GlobalAveragePooling2D()(x)

    # ---- Classifier ----
    x = layers.Dense(16, activation="relu")(x)
    #x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="kws_microspeech_cnn")

def build_tunable_model(hp):
    inputs = layers.Input(shape=(50, 40))
    x = layers.Reshape((50, 40, 1))(inputs)
    
    # ---- Conv block 1 ----
    filters_1 = hp.Int('conv1_filters', min_value=8, max_value=32, step=8)
    kernel_1_h = hp.Choice('conv1_kernel_height', values=[8, 10]) # Reduced max height to save params
    kernel_1_w = hp.Choice('conv1_kernel_width', values=[3, 4, 5])
    
    # Use use_bias=False because BatchNormalization follows
    x = layers.Conv2D(
        filters=filters_1,
        kernel_size=(kernel_1_h, kernel_1_w),
        strides=(2, 2),
        padding="same",
        use_bias=False 
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu6")(x) # Aligned with your target model
    
    # ---- Conv block 2 ----
    filters_2 = hp.Int('conv2_filters', min_value=16, max_value=32, step=16)
    kernel_2_h = hp.Choice('conv2_kernel_height', values=[3, 5])
    kernel_2_w = hp.Choice('conv2_kernel_width', values=[3, 4, 5])
    
    x = layers.Conv2D(
        filters=filters_2,
        kernel_size=(kernel_2_h, kernel_2_w),
        strides=(2, 2),
        padding="same",
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu6")(x)
    
    # ---- Conv block 3 ----
    filters_3 = hp.Int('conv3_filters', min_value=16, max_value=48, step=16) # Capped at 48
    kernel_3_size = hp.Choice('conv3_kernel_size', values=[3, 5])
    
    x = layers.Conv2D(
        filters=filters_3,
        kernel_size=(kernel_3_size, kernel_3_size),
        strides=(1, 1),
        padding="same",
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu6")(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    
    # ---- Classifier ----
    dense_units = hp.Int('dense_units', min_value=16, max_value=48, step=16)
    x = layers.Dense(dense_units)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu6")(x)
    
    if hp.Boolean('use_dropout'):
        x = layers.Dropout(hp.Float('dropout_rate', 0.1, 0.5, step=0.1))(x)
    
    outputs = layers.Dense(12, activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # ---- 🔑 SIZE CONSTRAINT CHECK ----
    # This prevents the "3 consecutive failures" crash by checking size BEFORE compiling
    if model.count_params() > 50000:
        raise kt.FailedTrialError(f"Model too large: {model.count_params()} params")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model