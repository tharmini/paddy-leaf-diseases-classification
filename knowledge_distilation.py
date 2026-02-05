# ============================================================
# Knowledge Distillation for Wavelet-Guided Student Model
# ============================================================

import tensorflow as tf
from tensorflow.keras import callbacks
import numpy as np

# ----------------------------
# Distillation Hyperparameters
# ----------------------------
TEMPERATURE = 4
ALPHA = 0.5

# ============================================================
# Knowledge Distillation Model Definition
# ============================================================
class KnowledgeDistillation(tf.keras.Model):
    """
    Custom Keras model for knowledge distillation.
    Trains a student model using supervision from a teacher model.
    """

    def __init__(self, student, teacher, temperature=3, alpha=0.5):
        super(KnowledgeDistillation, self).__init__()

        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha

        # Loss functions
        self.student_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False
        )
        self.distillation_loss_fn = tf.keras.losses.KLDivergence()

        # Metrics
        self.student_loss_tracker = tf.keras.metrics.Mean(name="student_loss")
        self.distillation_loss_tracker = tf.keras.metrics.Mean(name="distillation_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        )

    @property
    def metrics(self):
        """
        List of metrics to reset at the start of each epoch.
        """
        return [
            self.student_loss_tracker,
            self.distillation_loss_tracker,
            self.total_loss_tracker,
            self.accuracy_tracker,
        ]

    def compile(self, optimizer, **kwargs):
        """
        Compile method for the distillation model.
        """
        super().compile(**kwargs)
        self.optimizer = optimizer

    def call(self, inputs, training=False):
        """
        Forward pass (delegates to student model).
        """
        x, wavelet_features = inputs
        return self.student([x, wavelet_features], training=training)

    # --------------------------------------------------------
    # Training Step
    # --------------------------------------------------------
    def train_step(self, data):
        (x, wavelet_features), y = data

        # Teacher forward pass (no gradient)
        teacher_predictions = self.teacher(
            [x, wavelet_features], training=False
        )

        with tf.GradientTape() as tape:
            # Student forward pass
            student_predictions = self.student(
                [x, wavelet_features], training=True
            )

            # Supervised student loss
            student_loss = self.student_loss_fn(y, student_predictions)

            # Distillation loss (soft targets)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1)
            )

            # Combined loss
            total_loss = (
                self.alpha * student_loss
                + (1 - self.alpha)
                * distillation_loss
                * (self.temperature ** 2)
            )

        # Compute gradients w.r.t student parameters only
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)

        # Update student weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
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

    # --------------------------------------------------------
    # Validation / Test Step
    # --------------------------------------------------------
    def test_step(self, data):
        (x, wavelet_features), y = data

        # Student forward pass
        student_predictions = self.student(
            [x, wavelet_features], training=False
        )

        # Student loss
        student_loss = self.student_loss_fn(y, student_predictions)

        # Teacher forward pass
        teacher_predictions = self.teacher(
            [x, wavelet_features], training=False
        )

        # Distillation loss
        distillation_loss = self.distillation_loss_fn(
            tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
            tf.nn.softmax(student_predictions / self.temperature, axis=1)
        )

        # Total loss
        total_loss = (
            self.alpha * student_loss
            + (1 - self.alpha)
            * distillation_loss
            * (self.temperature ** 2)
        )

        # Update metrics
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

# ============================================================
# Build Student and Teacher Models
# ============================================================
print("Building student model...")
for (x, wavelet_features), y in train_dataset.take(1):
    _ = student_model([x, wavelet_features])
    break

print("Building teacher model...")
for (x, wavelet_features), y in train_dataset.take(1):
    _ = teacher_model([x, wavelet_features])
    break

# ============================================================
# Initialize Knowledge Distillation Model
# ============================================================
distillation_model = KnowledgeDistillation(
    student=student_model,
    teacher=teacher_model,
    temperature=TEMPERATURE,
    alpha=ALPHA
)

# Compile distillation model
distillation_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
)

# Build distillation model (required for custom Model)
print("Building distillation model...")
for (x, wavelet_features), y in train_dataset.take(1):
    _ = distillation_model([x, wavelet_features])
    break

# ============================================================
# Custom Callback: Save Student Weights Only
# ============================================================
class StudentModelCheckpoint(callbacks.Callback):
    """
    Custom checkpoint to save only student model weights
    based on validation accuracy.
    """

    def __init__(
        self,
        student_model,
        filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    ):
        super().__init__()
        self.student_model = student_model
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_value = -np.inf if mode == 'max' else np.inf

    def on_epoch_end(self, epoch, logs=None):
        current_value = logs.get(self.monitor)
        if current_value is None:
            return

        if self.mode == 'max':
            should_save = current_value > self.best_value
        else:
            should_save = current_value < self.best_value

        if should_save or not self.save_best_only:
            self.best_value = current_value

            formatted_filepath = self.filepath.format(
                epoch=epoch + 1,
                val_accuracy=current_value
            )

            self.student_model.save_weights(formatted_filepath)
            print(f"\nSaved student model weights to {formatted_filepath}")

# ============================================================
# Callbacks
# ============================================================
student_checkpoint = StudentModelCheckpoint(
    student_model=student_model,
    filepath='',
    monitor='val_accuracy',
    mode='max'
)

early_stopping = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    mode='max'
)

# ============================================================
# Knowledge Distillation Training
# ============================================================
print("Starting knowledge distillation training...")
history = distillation_model.fit(
    train_dataset,
    epochs=100,
    validation_data=validation_dataset,
    callbacks=[student_checkpoint, early_stopping]
)
