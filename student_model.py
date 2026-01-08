# ============================================================
# Student Model: Wavelet-Guided FPN-HydraNet with EfficientNetB0
# ============================================================

# ----------------------------
# Install required package
# ----------------------------
!pip install PyWavelets

# ----------------------------
# Import libraries
# ----------------------------
import os
import time
import json
import numpy as np
import pywt
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from google.colab import drive

# ----------------------------
# Mount Google Drive
# ----------------------------
drive.mount('/content/drive')

# ----------------------------
# Configuration
# ----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_directory = ''
test_directory  = ''

# ============================================================
# Dataset Loading
# ============================================================
train_dataset = image_dataset_from_directory(
    train_directory,
    validation_split=0.2,
    subset='training',
    seed=42,
    shuffle=True,
    labels='inferred',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    color_mode='rgb'
)

validation_dataset = image_dataset_from_directory(
    train_directory,
    validation_split=0.2,
    subset='validation',
    seed=42,
    shuffle=False,
    labels='inferred',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    color_mode='rgb'
)

test_dataset = image_dataset_from_directory(
    test_directory,
    shuffle=False,
    labels='inferred',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    color_mode='rgb',
    seed=42
)

# ----------------------------
# Dataset metadata
# ----------------------------
input_shape = (224, 224, 3)
num_classes = len(train_dataset.class_names)
class_namess = train_dataset.class_names

# ============================================================
# Learnable Wavelet Layer
# ============================================================
class LearnableWaveletLayer(tf.keras.layers.Layer):
    """
    Learnable Haar wavelet transform layer with
    trainable scaling parameters (alpha, beta).
    """
    def __init__(self):
        super(LearnableWaveletLayer, self).__init__()
        self.alpha = tf.Variable(1.0, trainable=True, dtype=tf.float32)
        self.beta  = tf.Variable(1.0, trainable=True, dtype=tf.float32)

    def wavelet_transform_per_channel(self, channel):
        LL, (LH, HL, HH) = pywt.dwt2(channel, 'haar')

        LL *= self.alpha
        LH *= self.beta
        HL *= self.beta
        HH *= self.beta

        return np.stack([LL, LH, HL, HH], axis=-1)

    def wavelet_transform_per_image(self, image):
        batch_size, height, width, _ = image.shape

        channels = [
            np.stack(
                [self.wavelet_transform_per_channel(image[j, :, :, i])
                 for j in range(batch_size)],
                axis=0
            )
            for i in range(3)
        ]

        return np.concatenate(channels, axis=-1)

    def call(self, inputs):
        wavelet_transformed = tf.py_function(
            func=self.wavelet_transform_per_image,
            inp=[inputs],
            Tout=tf.float32
        )
        wavelet_transformed.set_shape([None, 112, 112, 12])
        return wavelet_transformed


# Instantiate wavelet layer
wavelet_layer = LearnableWaveletLayer()

# ============================================================
# Dataset Preprocessing with Wavelet Features
# ============================================================
def apply_wavelet_transform(image):
    return wavelet_layer(image)

def preprocess_with_wavelet(image, label):
    wavelet_features = apply_wavelet_transform(image)
    return (image, wavelet_features), label

# Apply preprocessing
train_dataset      = train_dataset.map(preprocess_with_wavelet)
validation_dataset = validation_dataset.map(preprocess_with_wavelet)
test_dataset       = test_dataset.map(preprocess_with_wavelet)

# ============================================================
# Standalone Wavelet-Based Feature Encoder
# ============================================================
def build_wavelet_model():
    inputs = layers.Input(shape=(112, 112, 12))

    x = layers.Conv2D(24, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(48, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    return models.Model(inputs, x)

# ============================================================
# Depthwise Separable Convolution Block
# ============================================================
def conv_block(input_tensor, filters, kernel_size=3, block_name="conv_block"):
    x = layers.SeparableConv2D(
        filters,
        kernel_size,
        padding='same',
        name=f"{block_name}_conv"
    )(input_tensor)

    x = layers.BatchNormalization(name=f"{block_name}_bn")(x)
    x = layers.ReLU(name=f"{block_name}_relu")(x)

    return x

# ============================================================
# FPN-HydraNet Model Definition
# ============================================================
def build_fpn_hydranet_model(
    input_shape=(224, 224, 3),
    wavelet_shape=(112, 112, 12),
    num_classes=num_classes
):
    base_model = EfficientNetB0(
        include_top=False,
        input_shape=input_shape,
        weights='imagenet'
    )

    # Freeze backbone layers except last few
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[-15:]:
        layer.trainable = True

    # Backbone feature maps
    C2 = base_model.get_layer('block1a_project_bn').output
    C3 = base_model.get_layer('block2b_add').output
    C4 = base_model.get_layer('block3b_add').output
    C5 = base_model.get_layer('block4c_add').output
    C6 = base_model.get_layer('block5c_add').output
    C7 = base_model.get_layer('block6d_add').output
    C8 = base_model.get_layer('block7a_project_bn').output

    # FPN construction
    P8_a = layers.Conv2D(512, (1, 1), padding='same')(C8)
    P8   = conv_block(C8, 512, block_name="blocka")

    p7 = layers.Conv2D(512, (1, 1), padding='same')(C7)
    P8_p7_a   = layers.Add()([P8_a, p7])
    P8_p7_con = conv_block(P8_p7_a, 512, block_name="blockcn")

    P8_p7_up = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(P8_p7_a)
    P6 = layers.Conv2D(512, (1, 1), padding='same')(C6)
    P6_add = layers.Add()([P6, P8_p7_up])
    P6_add_con = conv_block(P6_add, 512, block_name="blocken")

    C5_1 = layers.Conv2D(512, (1, 1), padding='same')(C5)
    P5_add = layers.Add()([P6_add, C5_1])
    P5_con = conv_block(P5_add, 512, block_name="blockfn")

    P5_up = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(P5_add)
    C4_1 = layers.Conv2D(512, (1, 1), padding='same')(C4)
    P4_add = layers.Add()([P5_up, C4_1])
    P4_con = conv_block(P4_add, 512, block_name="blockgn")

    P4_up = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(P4_add)
    C3_1 = layers.Conv2D(512, (1, 1), padding='same')(C3)
    P3_add = layers.Add()([P4_up, C3_1])
    P3_con = conv_block(P3_add, 512, block_name="blockhn")

    P3_up = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(P3_add)
    C2_1 = layers.Conv2D(512, (1, 1), padding='same')(C2)
    P2 = layers.Add()([P3_up, C2_1])
    P2_con = conv_block(P2, 512, block_name="blockin")

    # Wavelet branch
    wavelet_inputs = layers.Input(shape=wavelet_shape)
    wavelet_features = build_wavelet_model()(wavelet_inputs)
    gate_wavelet_features = layers.Conv2D(512, (1, 1), padding='same')(wavelet_features)

    # HydraNet branch selection
    branches = [
        P8, P8_p7_con, P6_add_con,
        P5_con, P4_con, P3_con, P2_con
    ]

    branch_outputs = [
        layers.GlobalAveragePooling2D()(branch)
        for branch in branches
    ]

    branch_outputs_stack = layers.Lambda(
        lambda x: tf.stack(x, axis=1)
    )(branch_outputs)

    wavelet_vector = layers.GlobalAveragePooling2D()(gate_wavelet_features)

    importance_scores = layers.Dense(
        len(branches),
        activation='softmax'
    )(wavelet_vector)

    k = 3
    top_k_scores, top_k_indices = layers.Lambda(
        lambda x: tf.nn.top_k(x, k=k)
    )(importance_scores)

    top_k_branches = layers.Lambda(
        lambda inputs: tf.gather(inputs[0], inputs[1], batch_dims=1),
        output_shape=(k, 512)
    )([branch_outputs_stack, top_k_indices])

    weighted_top_k_branches = layers.Lambda(
        lambda x: tf.reduce_sum(
            x[0] * tf.expand_dims(x[1], axis=-1),
            axis=1
        )
    )([top_k_branches, top_k_scores])

    # Classification head
    x = layers.Dense(256, activation='relu')(weighted_top_k_branches)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(
        inputs=[base_model.input, wavelet_inputs],
        outputs=outputs
    )

    return model

# ============================================================
# Model Training
# ============================================================
student_model = build_fpn_hydranet_model()

student_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
