#teacher model with gating with channel attenion in lateral connection
!pip install PyWavelets
import numpy as np
import pywt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from google.colab import drive
import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
from tensorflow.keras import backend as K

# Mount Google Drive
drive.mount('/content/drive')

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
train_directory =  '/content/drive/MyDrive/RiceLeafsv3/train'
test_directory = '/content/drive/MyDrive/RiceLeafsv3/validation'

# ── Load datasets (NO wavelet preprocessing) ────────────────────────────────
train_dataset = image_dataset_from_directory(
    train_directory,
    validation_split=0.2, subset='training',
    seed=42, shuffle=True, labels='inferred',
    batch_size=BATCH_SIZE, image_size=IMG_SIZE, color_mode='rgb'
)
validation_dataset = image_dataset_from_directory(
    train_directory,
    validation_split=0.2, subset='validation',
    seed=42, shuffle=False, labels='inferred',
    batch_size=BATCH_SIZE, image_size=IMG_SIZE, color_mode='rgb'
)
test_dataset = image_dataset_from_directory(
    test_directory,
    shuffle=False, labels='inferred',
    batch_size=BATCH_SIZE, image_size=IMG_SIZE,
    color_mode='rgb', seed=42
)

input_shape = (224, 224, 3)
num_classes  = len(train_dataset.class_names)
class_names  = train_dataset.class_names
print(class_names)

def channel_attention(x, ratio=8, name="ca"):
    channels = x.shape[-1]

    avg_pool = layers.GlobalAveragePooling2D()(x)
    fc1 = layers.Dense(channels // ratio, activation='relu')(avg_pool)
    fc2 = layers.Dense(channels, activation='sigmoid')(fc1)

    scale = layers.Reshape((1,1,channels))(fc2)
    return layers.Multiply(name=name)([x, scale])


class LearnableWaveletLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LearnableWaveletLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name="alpha", shape=(), initializer="ones", trainable=True
        )
        self.beta = self.add_weight(
            name="beta", shape=(), initializer="ones", trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        ll = tf.constant([[ 0.5,  0.5], [ 0.5,  0.5]], dtype=tf.float32)
        lh = tf.constant([[-0.5, -0.5], [ 0.5,  0.5]], dtype=tf.float32)
        hl = tf.constant([[-0.5,  0.5], [-0.5,  0.5]], dtype=tf.float32)
        hh = tf.constant([[ 0.5, -0.5], [-0.5,  0.5]], dtype=tf.float32)

        filters = tf.stack([ll, lh, hl, hh], axis=-1)  # (2,2,4)
        filters = tf.expand_dims(filters, axis=-2)      # (2,2,1,4)

        outputs = []
        for i in range(3):
            channel = inputs[..., i:i+1]
            # Reflect padding
            channel = tf.pad(channel, [[0,0], [1,1], [1,1], [0,0]], mode='REFLECT')

            # Convolution with VALID
            conv = tf.nn.conv2d(channel, filters, strides=2, padding='VALID')
            LL      = conv[..., 0:1] * self.alpha
            others  = conv[..., 1:]  * self.beta
            outputs.append(tf.concat([LL, others], axis=-1))

        return tf.concat(outputs, axis=-1)  # (B, 112, 112, 12)


# ── Helper blocks ────────────────────────────────────────────────────────────
def conv_block(input_tensor, filters, kernel_size=3, block_name="conv_block"):
    x = layers.SeparableConv2D(filters, kernel_size, padding='same',
                               name=f"{block_name}_conv")(input_tensor)
    x = layers.BatchNormalization(name=f"{block_name}_bn")(x)
    x = layers.ReLU(name=f"{block_name}_relu")(x)
    return x


def build_wavelet_subgraph(wavelet_input):
    """Applies wavelet CNN layers to a tensor; returns (B,14,14,96)."""
    x = layers.Conv2D(24, (3,3), padding='same')(wavelet_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)   # 56x56

    x = layers.Conv2D(48, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)   # 28x28

    x = layers.Conv2D(96, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)   # 14x14
    return x


# ── Main model ───────────────────────────────────────────────────────────────
def build_fpn_hydranet_model(input_shape=(224, 224, 3), num_classes=num_classes):

    # ── 1. Backbone: EfficientNetB0 ─────────────────────────
    backbone = EfficientNetB0(
        include_top=False, input_shape=input_shape, weights='imagenet'
    )

    for layer in backbone.layers:
        layer.trainable = False
    for layer in backbone.layers[-15:]:
        layer.trainable = True

    # ── 2. Feature extractor (MATCH B0 STRUCTURE) ───────────
    feature_extractor = models.Model(
        inputs=backbone.input,
        outputs={
            'C2': backbone.get_layer('block1a_project_bn').output,  # 112
            'C3': backbone.get_layer('block2b_add').output,         # 56
            'C4': backbone.get_layer('block3b_add').output,         # 28
            'C5': backbone.get_layer('block4c_add').output,         # 14
            'C6': backbone.get_layer('block5c_add').output,         # 14
            'C7': backbone.get_layer('block6d_add').output,         # 7
            'C8': backbone.get_layer('block7a_project_bn').output,  # 7
        },
        name='feature_extractor'
    )

    # ── 3. Input ────────────────────────────────────────────
    image_input = layers.Input(shape=input_shape, name="image_input")

    # ── 4. 🔥 Wavelet (UNCHANGED FROM V2M) ───────────────────
    wavelet_layer = LearnableWaveletLayer(name="learnable_wavelet")
    wavelet_features = wavelet_layer(image_input)              # (B,112,112,12)
    wavelet_out = build_wavelet_subgraph(wavelet_features)     # (B,14,14,96)
    gate_wavelet_features = layers.Conv2D(512, (1,1), padding='same')(wavelet_out)

    # ── 5. Extract CNN features ─────────────────────────────
    feats = feature_extractor(image_input)

    C2, C3, C4 = feats['C2'], feats['C3'], feats['C4']
    C5, C6 = feats['C5'], feats['C6']
    C7, C8 = feats['C7'], feats['C8']

    # ── 6. Top layers (7x7) ────────────────────────────────
    P8_a = layers.Conv2D(512, (1,1), padding='same')(C8)
    P8 = conv_block(channel_attention(C8, name="C8_ca"), filters=512, block_name="blocka")
    P7_a = layers.Conv2D(512, (1,1), padding='same')(C7)
    P7 = conv_block(channel_attention(C7, name="C7_ca"), filters=512, block_name="blockb")
    P8_ac = channel_attention(P8_a, name="P8aa_ca")
    P7_ac = channel_attention(P7_a, name="P7aa_ca")
    p7_p8 = layers.Add()([P8_ac, P7_ac])
    p7_p8_con = conv_block(p7_p8, filters=512, block_name="blockc")
    # ── 7. 7x7 → 14x14 ─────────────────────────────────────
    p7_p8_up = layers.UpSampling2D((2,2), interpolation='bilinear')(p7_p8)

    P6 = layers.Conv2D(512, (1,1), padding='same')(C6)
    P6 = channel_attention(P6, name="P6_ca")

    p7_p8_up = channel_attention(p7_p8_up, name="P8up_ca")

    P6_add = layers.Add()([P6, p7_p8_up])
    P6_add_con = conv_block(P6_add, filters=512, block_name="blocken")

    # ── 8. 14x14 fusion ────────────────────────────────────
    C5_1 = layers.Conv2D(512, (1,1), padding='same')(C5)

    C5_1 = channel_attention(C5_1, name="C5_ca")
    P6_add = channel_attention(P6_add, name="P6_ca2")

    P5_add = layers.Add()([C5_1, P6_add])
    P5_con = conv_block(P5_add, filters=512, block_name="blockfn")

    # ── 9. 28x28 ───────────────────────────────────────────
    P5_up = layers.UpSampling2D((2,2), interpolation='bilinear')(P5_add)

    C4_1 = layers.Conv2D(512, (1,1), padding='same')(C4)

    C4_1 = channel_attention(C4_1, name="C4_ca")
    P5_up = channel_attention(P5_up, name="P5up_ca")

    P4_add = layers.Add()([C4_1, P5_up])
    P4_con = conv_block(P4_add, filters=512, block_name="blockgn")

    # ── 10. 56x56 ──────────────────────────────────────────
    P4_up = layers.UpSampling2D((2,2), interpolation='bilinear')(P4_add)

    C3_1 = layers.Conv2D(512, (1,1), padding='same')(C3)

    C3_1 = channel_attention(C3_1, name="C3_ca")
    P4_up = channel_attention(P4_up, name="P4up_ca")

    P3_add = layers.Add()([C3_1, P4_up])
    P3_con = conv_block(P3_add, filters=512, block_name="blockhn")

    # ── 11. 112x112 ────────────────────────────────────────
    P3_up = layers.UpSampling2D((2,2), interpolation='bilinear')(P3_add)

    C2_1 = layers.Conv2D(512, (1,1), padding='same')(C2)

    C2_1 = channel_attention(C2_1, name="C2_ca")
    P3_up = channel_attention(P3_up, name="P3up_ca")

    P2 = layers.Add()([C2_1, P3_up])
    P2_con = conv_block(P2, filters=512, block_name="blockin")

    # ── 12. HydraNet (UNCHANGED) ───────────────────────────
    branches = [P8, P7, p7_p8_con, P6_add_con, P5_con, P4_con, P3_con, P2_con]

    branch_outputs = [layers.GlobalAveragePooling2D()(b) for b in branches]
    branch_stack = layers.Lambda(lambda x: tf.stack(x, axis=1))(branch_outputs)

    wavelet_vector = layers.GlobalAveragePooling2D()(gate_wavelet_features)
    importance_scores = layers.Dense(len(branches), activation='softmax')(wavelet_vector)

    k = 3
    top_k_scores, top_k_indices = layers.Lambda(lambda x: tf.nn.top_k(x, k=k))(importance_scores)

    top_k_branches = layers.Lambda(
        lambda x: tf.gather(x[0], tf.cast(x[1], tf.int32), batch_dims=1)
    )([branch_stack, top_k_indices])

    fused = layers.Lambda(
        lambda x: tf.reduce_sum(x[0] * tf.expand_dims(x[1], axis=-1), axis=1)
    )([top_k_branches, top_k_scores])

    # ── 13. Classification ─────────────────────────────────
    x = layers.Dense(256, activation='relu')(fused)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=image_input, outputs=outputs)

    return model, wavelet_layer

# ── Build & compile ──────────────────────────────────────────────────────────
student_model, wavelet_layer = build_fpn_hydranet_model()
