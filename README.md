# Haar Wavelet-Guided Dynamic Feature Pyramid Network for Paddy Leaf Disease Classification

Official code for:

> Kiriharan Tharmini, Thanikasalam Kokul, Amirthalingam Ramanan, Terensan Suvanthini, Subha Fernando.
> **Haar wavelet-guided dynamic Feature Pyramid Network for an efficient paddy leaf disease classification.**
> *Computers and Electronics in Agriculture*, 2026.
> DOI: [10.1016/j.compag.2026.111960](https://doi.org/10.1016/j.compag.2026.111960)

The dataset used in this work is published alongside the paper (see the DOI above for access details).

## Overview

This repository implements a teacher-student framework for lightweight, efficient paddy leaf disease classification:

- **Teacher model** (`train_teacher_model.py`) — EfficientNetV2M backbone
- **Student model** (`train_student_model.py`) — EfficientNetB0 backbone, distilled from the teacher
- **Knowledge distillation** (`distill_student_model.py`) — trains the student using combined hard-label and soft-label (teacher-guided) supervision

Both teacher and student share the same core architecture pattern:

- An ImageNet-pretrained backbone (last N layers fine-tuned, rest frozen)
- **Haar wavelet-guided dynamic FPN**: a learnable wavelet decomposition layer (trainable low/high-frequency scaling coefficients) produces a gating signal from the raw image
- **Channel attention** applied to every lateral connection in the FPN top-down pathway before fusion
- **Dynamic branch router** (HydraNet-style): selects the top-k most relevant FPN branches per sample, weighted by the wavelet-derived importance scores, before classification

## Setup

```bash
pip install -r requirements.txt
```

## Dataset format

Expects an `ImageFolder`-style directory structure for both train and test sets:

```
train_dir/
  class_a/
    img1.jpg
    ...
  class_b/
    ...

test_dir/
  class_a/
    ...
  class_b/
    ...
```

## Usage

### 1. Train the teacher model (EfficientNetV2M)

```bash
python train_teacher_model.py \
  --train-dir /path/to/train \
  --test-dir /path/to/test \
  --checkpoint-dir ./checkpoints/teacher \
  --epochs 70 \
  --batch-size 32
```

### 2. Train the student model standalone (EfficientNetB0, optional baseline)

```bash
python train_student_model.py \
  --train-dir /path/to/train \
  --test-dir /path/to/validation \
  --checkpoint-dir ./checkpoints/student \
  --epochs 70 \
  --batch-size 32
```

### 3. Distill the student from the trained teacher

```bash
python distill_student_model.py \
  --teacher-weights ./checkpoints/teacher/teacher_model_best.weights.h5 \
  --checkpoint-dir ./checkpoints/distilled_student \
  --temperature 4.0 \
  --alpha 0.5 \
  --epochs 100
```

`distill_student_model.py` is provided as a template: wire up `student_model`, `teacher_model`, `train_dataset`, and `validation_dataset` in `main()` to match your project's model-building and data-loading code (see the inline comments and the example import lines in the script).

Paths can also be set via environment variables instead of CLI flags:

```bash
export TRAIN_DIR=/path/to/train
export TEST_DIR=/path/to/test
export CHECKPOINT_DIR=./checkpoints
export TEACHER_WEIGHTS=./checkpoints/teacher/teacher_model_best.weights.h5
python train_teacher_model.py
```

## Outputs

Saved to `--checkpoint-dir` for the teacher/student training scripts:
- Best model weights (`.weights.h5`, selected by validation accuracy)
- `confusion_matrix.png`
- `training_history.png` (accuracy/loss curves)
- `wavelet_evolution.png` (alpha/beta coefficient evolution)
- `summary.json` (final test/val metrics, inference time, class names)

For `distill_student_model.py`, only the distilled student's weights are checkpointed (`student_distilled_epoch{N}_val_acc{acc}.weights.h5`).

## Notes

- Update `class_names` ordering in the scripts if your dataset's inferred label order differs from the directory listing order reported by `image_dataset_from_directory`.
- Each script's feature extractor hardcodes backbone-specific layer names (e.g. `block1c_add` for EfficientNetV2M, `block1a_project_bn` for EfficientNetB0) for multi-scale feature extraction. These are stable per-backbone across TensorFlow/Keras versions, but verify against `model.summary()` if you swap the backbone or TF version.
- The student model (EfficientNetB0) has one fewer FPN level (C2-C8, 7 levels) than the teacher model (EfficientNetV2M, C2-C9, 8 levels), reflecting the shallower backbone.
- During distillation, gradients are computed and applied to the student's trainable variables only; the teacher is frozen (`training=False`) throughout.

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{THARMINI2026111960,
title = {Haar wavelet-guided dynamic Feature Pyramid Network for an efficient paddy leaf disease classification},
journal = {Computers and Electronics in Agriculture},
volume = {250},
pages = {111960},
year = {2026},
issn = {0168-1699},
doi = {https://doi.org/10.1016/j.compag.2026.111960},
url = {https://www.sciencedirect.com/science/article/pii/S0168169926005557},
author = {Kiriharan Tharmini and Thanikasalam Kokul and Amirthalingam Ramanan and Terensan Suvanthini and Subha Fernando},
keywords = {Paddy leaf disease classification, Feature pyramid network, Knowledge distillation, Haar wavelet-guided deep learning},
abstract = {Early and accurate identification of paddy leaf diseases is crucial for timely intervention and effective crop management. Although deep learning based approaches have demonstrated strong performance in paddy leaf disease classification, they often struggle to capture the diverse multiscale symptom patterns of diseases, resulting in the misclassification of challenging samples. Moreover, the high computational cost of deep learning based models remains a significant limitation. To address these challenges, this study proposes a novel Haar wavelet-guided deep Feature Pyramid Network (FPN) model. In the proposed framework, deep features are extracted using a backbone network, and a FPN is employed to capture multiscale disease symptoms. In parallel, Haar wavelet features are extracted to provide complementary texture cues that are not explicitly modelled by deep features. Guided by these wavelet features, a gating-based dynamic feature selection module is then utilised to identify image-specific FPN features, which are subsequently used for classification. In the second phase of the study, a lightweight variant of the proposed model is developed using a knowledge distillation technique, achieving high classification accuracy while substantially reducing the number of parameters and FLOPs. In addition, a new benchmark dataset, named NP-LankaPaddy, is constructed to address the limited availability of datasets representing Sri Lankan paddy leaf disease conditions. Experimental results demonstrate that both the proposed approach and its lightweight variant outperform state-of-the-art methods across four benchmark datasets, achieving notably high accuracies of 98.04% and 97.84%, respectively, on the well known Paddy Doctor dataset. The source code and dataset supporting this study are openly available at https://doi.org/10.5281/zenodo.18334503.
}
```
