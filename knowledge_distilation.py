"""
Knowledge Distillation: Wavelet-Guided Teacher -> Student

Trains the lightweight student model (EfficientNetB0-based) using soft-label
supervision from the larger, pretrained teacher model (EfficientNetV2M-based),
combined with standard hard-label supervision.

Both teacher and student expect input of the form (image, wavelet_features)
where `wavelet_features` is produced by the shared `LearnableWaveletLayer`
defined in the model-building scripts.

Loss
----
    total_loss = alpha * student_loss
                 + (1 - alpha) * distillation_loss * temperature^2

    student_loss        : SparseCategoricalCrossentropy(y_true, student_pred)
    distillation_loss    : KLDivergence(softmax(teacher/T), softmax(student/T))

Usage
-----
    python distill_student_model.py \
        --teacher-weights /path/to/teacher.weights.h5 \
        --checkpoint-dir ./checkpoints/distilled_student

Assumes `student_model` and `teacher_model` are already built (e.g. via
`build_fpn_hydranet_model(...)` from the student/teacher model scripts) and
that `train_dataset` / `validation_dataset` yield batches shaped as
`((images, wavelet_features), labels)`. Wire up dataset construction and
model instantiation in `main()` to match your project's loaders.
"""

import os
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Knowledge distillation: teacher -> student")
    parser.add_argument("--teacher-weights", type=str,
                         default=os.environ.get("TEACHER_WEIGHTS", ""),
                         help="Path to pretrained teacher model weights (.weights.h5)")
    parser.add_argument("--checkpoint-dir", type=str,
                         default=os.environ.get("CHECKPOINT_DIR", "./checkpoints/distilled_student"),
                         help="Directory to save distilled student weights")
    parser.add_argument("--temperature", type=float, default=4.0,
                         help="Softmax temperature for distillation")
    parser.add_argument("--alpha", type=float, default=0.5,
                         help="Weight on hard-label student loss vs. distillation loss")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10,
                         help="Early stopping patience (epochs)")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Distillation Model
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeDistillation(tf.keras.Model):
    """
    Wraps a frozen teacher and a trainable student. Computes a weighted
    combination of hard-label cross-entropy (student vs ground truth) and
    soft-label KL divergence (student vs temperature-scaled teacher) at
    each training step, and updates only the student's weights.
    """

    def __init__(self, student, teacher, temperature=3.0, alpha=0.5):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha

        self.student_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.distillation_loss_fn = tf.keras.losses.KLDivergence()

        self.student_loss_tracker = tf.keras.metrics.Mean(name="student_loss")
        self.distillation_loss_tracker = tf.keras.metrics.Mean(name="distillation_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    @property
    def metrics(self):
        return [
            self.student_loss_tracker,
            self.distillation_loss_tracker,
            self.total_loss_tracker,
            self.accuracy_tracker,
        ]

    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer

    def call(self, inputs, training=False):
        x, wavelet_features = inputs
        return self.student([x, wavelet_features], training=training)

    def _compute_losses(self, x, wavelet_features, y, student_training):
        teacher_predictions = self.teacher([x, wavelet_features], training=False)
        student_predictions = self.student([x, wavelet_features], training=student_training)

        student_loss = self.student_loss_fn(y, student_predictions)
        distillation_loss = self.distillation_loss_fn(
            tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
            tf.nn.softmax(student_predictions / self.temperature, axis=1),
        )
        total_loss = (
            self.alpha * student_loss
            + (1 - self.alpha) * distillation_loss * (self.temperature ** 2)
        )
        return student_predictions, student_loss, distillation_loss, total_loss

    def train_step(self, data):
        (x, wavelet_features), y = data

        with tf.GradientTape() as tape:
            student_predictions, student_loss, distillation_loss, total_loss = self._compute_losses(
                x, wavelet_features, y, student_training=True
            )

        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.student_loss_tracker.update_state(student_loss)
        self.distillation_loss_tracker.update_state(distillation_loss)
        self.total_loss_tracker.update_state(total_loss)
        self.accuracy_tracker.update_state(y, student_predictions)

        return {
            "student_loss": self.student_loss_tracker.result(),
            "distillation_loss": self.distillation_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result(),
        }

    def test_step(self, data):
        (x, wavelet_features), y = data

        student_predictions, student_loss, distillation_loss, total_loss = self._compute_losses(
            x, wavelet_features, y, student_training=False
        )

        self.student_loss_tracker.update_state(student_loss)
        self.distillation_loss_tracker.update_state(distillation_loss)
        self.total_loss_tracker.update_state(total_loss)
        self.accuracy_tracker.update_state(y, student_predictions)

        return {
            "student_loss": self.student_loss_tracker.result(),
            "distillation_loss": self.distillation_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Callback: save student-only weights
# ─────────────────────────────────────────────────────────────────────────────

class StudentModelCheckpoint(callbacks.Callback):
    """Saves only the student model's weights, tracked by a chosen metric."""

    def __init__(self, student_model, filepath_template, monitor="val_accuracy",
                 mode="max", save_best_only=True):
        super().__init__()
        self.student_model = student_model
        self.filepath_template = filepath_template
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_value = -np.inf if mode == "max" else np.inf

    def on_epoch_end(self, epoch, logs=None):
        current_value = (logs or {}).get(self.monitor)
        if current_value is None:
            return

        should_save = (
            current_value > self.best_value if self.mode == "max"
            else current_value < self.best_value
        )

        if should_save or not self.save_best_only:
            self.best_value = current_value
            filepath = self.filepath_template.format(epoch=epoch + 1, val_accuracy=current_value)
            self.student_model.save_weights(filepath)
            print(f"\nSaved student model weights to {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Build / load models ─────────────────────────────────────────────
    # Replace these with your actual model construction + dataset loading,
    # e.g. by importing `build_fpn_hydranet_model` from the student/teacher
    # model scripts and your own data pipeline that yields
    # ((images, wavelet_features), labels) batches.
    #
    # Example:
    #   from train_student_model import build_fpn_hydranet_model as build_student
    #   from train_teacher_model import build_fpn_hydranet_model as build_teacher
    #   student_model, _ = build_student(input_shape=(224, 224, 3), num_classes=NUM_CLASSES)
    #   teacher_model, _ = build_teacher(input_shape=(224, 224, 3), num_classes=NUM_CLASSES)
    #   teacher_model.load_weights(args.teacher_weights)
    #   train_dataset, validation_dataset = build_datasets(...)
    raise NotImplementedError(
        "Wire up student_model, teacher_model, train_dataset, and "
        "validation_dataset for your project before running."
    )

    # ── Build models on one batch (required before calling .fit on a
    #    subclassed tf.keras.Model with list/tuple inputs) ───────────────
    for (x, wavelet_features), y in train_dataset.take(1):
        _ = student_model([x, wavelet_features])
        _ = teacher_model([x, wavelet_features])
        break

    # ── Knowledge distillation wrapper ──────────────────────────────────
    distillation_model = KnowledgeDistillation(
        student=student_model,
        teacher=teacher_model,
        temperature=args.temperature,
        alpha=args.alpha,
    )
    distillation_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate))

    for (x, wavelet_features), y in train_dataset.take(1):
        _ = distillation_model([x, wavelet_features])
        break

    # ── Callbacks ────────────────────────────────────────────────────────
    checkpoint_template = os.path.join(
        args.checkpoint_dir,
        "student_distilled_epoch{epoch:02d}_val_acc{val_accuracy:.4f}.weights.h5",
    )
    student_checkpoint = StudentModelCheckpoint(
        student_model=student_model,
        filepath_template=checkpoint_template,
        monitor="val_accuracy",
        mode="max",
    )
    early_stopping = callbacks.EarlyStopping(
        monitor="val_accuracy", patience=args.patience,
        restore_best_weights=True, mode="max",
    )

    # ── Train ────────────────────────────────────────────────────────────
    print("Starting knowledge distillation training...")
    distillation_model.fit(
        train_dataset,
        epochs=args.epochs,
        validation_data=validation_dataset,
        callbacks=[student_checkpoint, early_stopping],
    )


if __name__ == "__main__":
    main()
