"""
Teacher Model: EfficientNetV2M Backbone + FPN + HydraNet Dynamic Branch
Selection + Learnable Wavelet Gating + Channel Attention on Lateral
Connections.

Architecture summary
---------------------
1. A frozen (mostly) EfficientNetV2M backbone extracts multi-scale features
   (C2-C9) from the input image.
2. A learnable wavelet decomposition layer produces a low/high-frequency
   representation of the raw image, used purely as a *gating signal*.
3. Each lateral/skip connection in the FPN top-down pathway is refined with
   channel attention before fusion (Add) with the upsampled path.
4. A HydraNet-style dynamic router selects the top-k most relevant FPN
   branches per sample (scored by the wavelet gating signal) and fuses them
   with learned, sample-dependent weights before classification.

Usage
-----
    python train_teacher_model.py --train-dir /path/to/train --test-dir /path/to/test

Or configure paths via environment variables / a `.env` file (see Config
section below) and simply run:

    python train_teacher_model.py
"""

import os
import time
import argparse
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.applications import EfficientNetV2M
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
# All filesystem paths are configurable via CLI args or environment
# variables, so no private/local paths are hardcoded in this file.
#
#   TRAIN_DIR        Path to training data (ImageFolder-style directory)
#   TEST_DIR         Path to test data (ImageFolder-style directory)
#   CHECKPOINT_DIR    Directory to save model checkpoints
#
# Example:
#   export TRAIN_DIR=/data/dataset/train_augment
#   export TEST_DIR=/data/dataset/test
#   export CHECKPOINT_DIR=./checkpoints

def parse_args():
    parser = argparse.ArgumentParser(description="Train teacher model (EfficientNetV2M + FPN + HydraNet)")
    parser.add_argument("--train-dir", type=str,
                         default=os.environ.get("TRAIN_DIR", "./data/train"),
                         help="Path to training dataset directory")
    parser.add_argument("--test-dir", type=str,
                         default=os.environ.get("TEST_DIR", "./data/test"),
                         help="Path to test dataset directory")
    parser.add_argument("--checkpoint-dir", type=str,
                         default=os.environ.get("CHECKPOINT_DIR", "./checkpoints"),
                         help="Directory to save model checkpoints")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size (square)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--top-k", type=int, default=3, help="Number of FPN branches selected by the router")
    parser.add_argument("--unfrozen-backbone-layers", type=int, default=15,
                         help="Number of final backbone layers to unfreeze for fine-tuning")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model building blocks
# ─────────────────────────────────────────────────────────────────────────────

def channel_attention(x, ratio=8, name="ca"):
    """Squeeze-and-excitation-style channel attention."""
    channels = x.shape[-1]

    avg_pool = layers.GlobalAveragePooling2D()(x)
    fc1 = layers.Dense(channels // ratio, activation="relu")(avg_pool)
    fc2 = layers.Dense(channels, activation="sigmoid")(fc1)

    scale = layers.Reshape((1, 1, channels))(fc2)
    return layers.Multiply(name=name)([x, scale])


class LearnableWaveletLayer(tf.keras.layers.Layer):
    """
    Learnable Haar-wavelet-style decomposition with trainable scaling
    coefficients (alpha for the low-frequency band, beta for the three
    high-frequency bands). Applied per RGB channel and concatenated.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(name="alpha", shape=(), initializer="ones", trainable=True)
        self.beta = self.add_weight(name="beta", shape=(), initializer="ones", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        ll = tf.constant([[0.5, 0.5], [0.5, 0.5]], dtype=tf.float32)
        lh = tf.constant([[-0.5, -0.5], [0.5, 0.5]], dtype=tf.float32)
        hl = tf.constant([[-0.5, 0.5], [-0.5, 0.5]], dtype=tf.float32)
        hh = tf.constant([[0.5, -0.5], [-0.5, 0.5]], dtype=tf.float32)

        filters = tf.stack([ll, lh, hl, hh], axis=-1)  # (2, 2, 4)
        filters = tf.expand_dims(filters, axis=-2)  # (2, 2, 1, 4)

        outputs = []
        for i in range(3):  # one pass per RGB channel
            channel = inputs[..., i:i + 1]
            channel = tf.pad(channel, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
            conv = tf.nn.conv2d(channel, filters, strides=2, padding="VALID")
            low_freq = conv[..., 0:1] * self.alpha
            high_freq = conv[..., 1:] * self.beta
            outputs.append(tf.concat([low_freq, high_freq], axis=-1))

        return tf.concat(outputs, axis=-1)  # (B, H/2, W/2, 12)


def conv_block(input_tensor, filters, kernel_size=3, block_name="conv_block"):
    x = layers.SeparableConv2D(filters, kernel_size, padding="same", name=f"{block_name}_conv")(input_tensor)
    x = layers.BatchNormalization(name=f"{block_name}_bn")(x)
    x = layers.ReLU(name=f"{block_name}_relu")(x)
    return x


def build_wavelet_subgraph(wavelet_input):
    """Small CNN that turns the wavelet decomposition into a gating feature map."""
    x = layers.Conv2D(24, (3, 3), padding="same")(wavelet_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(48, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(96, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

def build_fpn_hydranet_model(input_shape, num_classes, top_k=3, unfrozen_backbone_layers=15):
    """
    Builds the teacher model: EfficientNetV2M backbone -> multi-scale FPN with
    channel-attention-gated lateral connections -> HydraNet dynamic branch
    router (gated by a learnable wavelet signal) -> classification head.
    """

    # 1. Backbone (mostly frozen, last N layers fine-tuned)
    backbone = EfficientNetV2M(include_top=False, input_shape=input_shape, weights="imagenet")
    for layer in backbone.layers:
        layer.trainable = False
    for layer in backbone.layers[-unfrozen_backbone_layers:]:
        layer.trainable = True

    # 2. Multi-output feature extractor exposing intermediate feature maps
    feature_extractor = models.Model(
        inputs=backbone.input,
        outputs={
            "C2": backbone.get_layer("block1c_add").output,
            "C3": backbone.get_layer("block2e_add").output,
            "C4": backbone.get_layer("block3e_add").output,
            "C5": backbone.get_layer("block4g_add").output,
            "C6": backbone.get_layer("block5n_add").output,
            "C7": backbone.get_layer("block6b_add").output,
            "C8": backbone.get_layer("block6r_add").output,
            "C9": backbone.get_layer("block7e_add").output,
        },
        name="feature_extractor",
    )

    # 3. Single input
    image_input = layers.Input(shape=input_shape, name="image_input")

    # 4. Wavelet gating branch
    wavelet_layer = LearnableWaveletLayer(name="learnable_wavelet")
    wavelet_features = wavelet_layer(image_input)
    wavelet_out = build_wavelet_subgraph(wavelet_features)
    gate_wavelet_features = layers.Conv2D(512, (1, 1), padding="same")(wavelet_out)

    # 5. Backbone features
    feats = feature_extractor(image_input)
    C2, C3, C4 = feats["C2"], feats["C3"], feats["C4"]
    C5, C6 = feats["C5"], feats["C6"]
    C7, C8, C9 = feats["C7"], feats["C8"], feats["C9"]

    # 6. High-level branches with channel attention
    P8 = conv_block(channel_attention(C8, name="C8_ca"), filters=512, block_name="branch_p8")
    P7 = conv_block(channel_attention(C7, name="C7_ca"), filters=512, block_name="branch_p7")
    P9 = conv_block(channel_attention(C9, name="C9_ca"), filters=512, block_name="branch_p9")

    P8_proj = layers.Conv2D(512, (1, 1), padding="same")(C8)
    P9_proj = layers.Conv2D(512, (1, 1), padding="same")(C9)
    P8_proj = channel_attention(P8_proj, name="P8_proj_ca")
    P9_proj = channel_attention(P9_proj, name="P9_proj_ca")
    P9_P8_fused = layers.Add()([P8_proj, P9_proj])
    P9_P8_con = conv_block(P9_P8_fused, filters=512, kernel_size=3, block_name="branch_p9_p8")

    # 7. FPN top-down pathway, channel attention applied to every lateral
    P9_P8_up = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(P9_P8_fused)
    P6_lateral = layers.Conv2D(512, (1, 1), padding="same")(C6)
    P6_lateral = channel_attention(P6_lateral, name="P6_lateral_ca")
    P9_P8_up = channel_attention(P9_P8_up, name="P9_P8_up_ca")
    P6_fused = layers.Add()([P6_lateral, P9_P8_up])
    P6_con = conv_block(P6_fused, filters=512, block_name="branch_p6")

    C5_lateral = layers.Conv2D(512, (1, 1), padding="same")(C5)
    C5_lateral = channel_attention(C5_lateral, name="C5_lateral_ca")
    P6_fused_ca = channel_attention(P6_fused, name="P6_fused_ca")
    P5_fused = layers.Add()([C5_lateral, P6_fused_ca])
    P5_con = conv_block(P5_fused, filters=512, block_name="branch_p5")

    P5_up = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(P5_fused)
    C4_lateral = layers.Conv2D(512, (1, 1), padding="same")(C4)
    C4_lateral = channel_attention(C4_lateral, name="C4_lateral_ca")
    P5_up = channel_attention(P5_up, name="P5_up_ca")
    P4_fused = layers.Add()([C4_lateral, P5_up])
    P4_con = conv_block(P4_fused, filters=512, block_name="branch_p4")

    P4_up = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(P4_fused)
    C3_lateral = layers.Conv2D(512, (1, 1), padding="same")(C3)
    C3_lateral = channel_attention(C3_lateral, name="C3_lateral_ca")
    P4_up = channel_attention(P4_up, name="P4_up_ca")
    P3_fused = layers.Add()([C3_lateral, P4_up])
    P3_con = conv_block(P3_fused, filters=512, block_name="branch_p3")

    P3_up = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(P3_fused)
    C2_lateral = layers.Conv2D(512, (1, 1), padding="same")(C2)
    C2_lateral = channel_attention(C2_lateral, name="C2_lateral_ca")
    P3_up = channel_attention(P3_up, name="P3_up_ca")
    P2_fused = layers.Add()([C2_lateral, P3_up])
    P2_con = conv_block(P2_fused, filters=512, block_name="branch_p2")

    # 8. HydraNet dynamic branch selection, gated by the wavelet signal
    branches = [P8, P7, P9, P9_P8_con, P6_con, P5_con, P4_con, P3_con, P2_con]
    branch_vectors = [layers.GlobalAveragePooling2D()(b) for b in branches]
    branch_stack = layers.Lambda(lambda x: tf.stack(x, axis=1))(branch_vectors)  # (B, num_branches, 512)

    wavelet_vector = layers.GlobalAveragePooling2D()(gate_wavelet_features)
    importance_scores = layers.Dense(len(branches), activation="softmax")(wavelet_vector)

    top_k_scores, top_k_indices = layers.Lambda(lambda x: tf.nn.top_k(x, k=top_k))(importance_scores)

    top_k_branches = layers.Lambda(
        lambda inp: tf.gather(inp[0], tf.cast(inp[1], tf.int32), batch_dims=1),
        output_shape=(top_k, 512),
    )([branch_stack, top_k_indices])

    weighted_branches = layers.Lambda(
        lambda x: tf.reduce_sum(x[0] * tf.expand_dims(x[1], axis=-1), axis=1)
    )([top_k_branches, top_k_scores])

    x = layers.Dense(256, activation="relu")(weighted_branches)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=image_input, outputs=outputs)
    return model, wavelet_layer


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

class TrackWaveletParams(tf.keras.callbacks.Callback):
    """Logs and stores the learnable wavelet alpha/beta coefficients per epoch."""

    def __init__(self, wavelet_layer):
        super().__init__()
        self.wavelet_layer = wavelet_layer
        self.alpha_values = []
        self.beta_values = []

    def on_train_begin(self, logs=None):
        self.alpha_values = []
        self.beta_values = []

    def on_epoch_end(self, epoch, logs=None):
        alpha = self.wavelet_layer.alpha.numpy()
        beta = self.wavelet_layer.beta.numpy()
        self.alpha_values.append(alpha)
        self.beta_values.append(beta)
        print(f"\nEpoch {epoch + 1}: alpha={alpha:.4f}, beta={beta:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(history, save_path=None):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_wavelet_evolution(track_callback, save_path=None):
    plt.figure()
    plt.plot(track_callback.alpha_values, label="alpha")
    plt.plot(track_callback.beta_values, label="beta")
    plt.legend()
    plt.title("Evolution of Wavelet Alpha/Beta Coefficients")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    img_size = (args.img_size, args.img_size)
    input_shape = (args.img_size, args.img_size, 3)

    # ── Datasets ─────────────────────────────────────────────────────────
    train_dataset = image_dataset_from_directory(
        args.train_dir,
        validation_split=0.2, subset="training",
        seed=args.seed, shuffle=True, labels="inferred",
        batch_size=args.batch_size, image_size=img_size, color_mode="rgb",
    )
    validation_dataset = image_dataset_from_directory(
        args.train_dir,
        validation_split=0.2, subset="validation",
        seed=args.seed, shuffle=False, labels="inferred",
        batch_size=args.batch_size, image_size=img_size, color_mode="rgb",
    )
    test_dataset = image_dataset_from_directory(
        args.test_dir,
        shuffle=False, labels="inferred",
        batch_size=args.batch_size, image_size=img_size,
        color_mode="rgb", seed=args.seed,
    )

    class_names = train_dataset.class_names
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # ── Build & compile model ───────────────────────────────────────────
    model, wavelet_layer = build_fpn_hydranet_model(
        input_shape=input_shape,
        num_classes=num_classes,
        top_k=args.top_k,
        unfrozen_backbone_layers=args.unfrozen_backbone_layers,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    model.summary()

    # ── Callbacks ────────────────────────────────────────────────────────
    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        "teacher_model_epoch{epoch:02d}_val_acc{val_accuracy:.4f}.weights.h5",
    )
    early_stopping = EarlyStopping(monitor="val_accuracy", patience=10, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(
        checkpoint_path, monitor="val_accuracy", save_best_only=True,
        save_weights_only=True, mode="max", verbose=1,
    )
    track_callback = TrackWaveletParams(wavelet_layer)

    # ── Train ────────────────────────────────────────────────────────────
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        callbacks=[early_stopping, model_checkpoint, track_callback],
        verbose=1,
    )

    # ── Evaluate ─────────────────────────────────────────────────────────
    start_test_time = time.time()
    test_loss, test_acc = model.evaluate(test_dataset, verbose=1)
    total_test_time = time.time() - start_test_time
    num_test_samples = len(test_dataset) * args.batch_size
    avg_inference_time = total_test_time / num_test_samples

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    print(f"Total Testing Time: {total_test_time:.2f}s ({total_test_time / 60:.2f} min)")
    print(f"Average Inference Time per Image: {avg_inference_time:.6f}s")

    val_loss, val_acc = model.evaluate(validation_dataset, verbose=1)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    # ── Confusion matrix ─────────────────────────────────────────────────
    y_true = np.concatenate([y.numpy() for _, y in test_dataset], axis=0)
    y_pred = np.argmax(model.predict(test_dataset), axis=-1)

    confusion_mtx = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(args.checkpoint_dir, "confusion_matrix.png"))
    plt.show()

    # ── Plots ────────────────────────────────────────────────────────────
    plot_training_history(history, save_path=os.path.join(args.checkpoint_dir, "training_history.png"))
    plot_wavelet_evolution(track_callback, save_path=os.path.join(args.checkpoint_dir, "wavelet_evolution.png"))

    # ── Save summary metrics ────────────────────────────────────────────
    summary = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc),
        "avg_inference_time_sec": float(avg_inference_time),
        "class_names": class_names,
    }
    with open(os.path.join(args.checkpoint_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved run summary to {os.path.join(args.checkpoint_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
