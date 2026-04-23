# ConDetection: Cross-Domain Audio Deepfake Detection

A Hierarchical Multi-Scale Conformer architecture for audio deepfake detection, with cross-scale temporal attention, consistency regularization, Grad-CAM interpretability, and rigorous cross-domain generalization evaluation.

---

## Research Motivation

Existing audio deepfake detectors suffer severe performance degradation when deployed outside their training distribution. Müller et al. (2022) demonstrated up to **1000% EER degradation** when moving from ASVspoof to real-world audio. ConDetection directly addresses this by:

- Training on the **Fake or Real (FoR)** dataset across all four subsets for strong in-domain diversity
- Testing generalization on the **In-the-Wild** dataset as a hard out-of-domain benchmark
- Combining hierarchical Conformer encoding with cross-scale attention, previously never evaluated together under a cross-domain protocol

---

## Architecture

ConDetection processes audio at three spectral resolutions simultaneously:

| Scale  | n_fft | hop_length | n_mels |
|--------|-------|------------|--------|
| Fine   | 400   | 160        | 64     |
| Mid    | 1024  | 256        | 80     |
| Coarse | 2048  | 512        | 128    |

Each scale is processed by a dedicated `ScaleEncoder` (lightweight CNN → linear projection), then passed through shared Conformer blocks with hierarchical temporal pooling. A `CrossScaleAttentionFusion` module fuses the three scale embeddings via multi-head cross-attention, and a final MLP classifier outputs a binary real/fake prediction.

**Key components:**

- **Multi-resolution Log-Mel Spectrograms** — batched on GPU via `torch.stft`
- **MixStyle** — domain generalization via feature statistics mixing during training
- **SpecAugment** — frequency and time masking applied inline during batch processing
- **Consistency Regularization** — cross-scale embedding alignment loss
- **Focal BCE Loss** — handles class imbalance during training
- **Grad-CAM** — saliency map visualization per spectral scale for interpretability
- **Cross-scale attention visualization** — heatmaps showing inter-resolution information flow

---

## Datasets

### Training & Validation — Fake or Real (FoR)

**Download:** [https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)

The FoR dataset contains real speech samples and TTS-generated fakes across four subsets:

| Subset          | Description                              |
|-----------------|------------------------------------------|
| `for-original`  | Original real/fake pairs                 |
| `for-norm`      | Normalized audio version                 |
| `for-2seconds`  | 2-second segments                        |
| `for-rerecorded`| Re-recorded through a physical speaker   |

All four subsets are used for training and validation to maximize domain diversity. Expected directory structure after download:

```
for-original/for-original/
    training/real/
    training/fake/
    validation/real/
    validation/fake/
    testing/real/
    testing/fake/
for-norm/for-norm/...
for-2seconds/for-2seconds/...
for-rerecorded/for-rerecorded/...
```

### Out-of-Domain Test — In-the-Wild Audio Deepfake

**Download:** [https://www.kaggle.com/datasets/abdallamohamed312/in-the-wild-audio-deepfake](https://www.kaggle.com/datasets/abdallamohamed312/in-the-wild-audio-deepfake)

A real-world deepfake detection dataset collected from public sources — significantly more challenging than studio-recorded benchmarks. Used exclusively for out-of-domain evaluation (no samples are ever seen during training).

Expected directory structure:

```
release_in_the_wild/
    real/
    fake/
```

---

## Setup

### Requirements

```bash
pip install torch torchaudio librosa numpy pandas matplotlib seaborn scipy scikit-learn statsmodels
```

Python 3.9+ and PyTorch 2.0+ are recommended. A CUDA-capable GPU is strongly recommended; the notebook runs on Kaggle's free T4/P100 instances.

### Kaggle Input Paths

The notebook expects datasets mounted at:

```python
FOR_BASE = '/kaggle/input/datasets/mohammedabdeldayem/the-fake-or-real-dataset'
ITW_ROOT = '/kaggle/input/datasets/abdallamohamed312/in-the-wild-audio-deepfake/release_in_the_wild'
```

Adjust these paths in the **Configuration** cell if running locally.

---

## Notebook Structure

| Section | Description |
|---------|-------------|
| 1. Setup & Imports | Library imports, reproducibility seeds, device configuration |
| 2. Configuration | All hyperparameters in one place — audio settings, model size, training schedule, dataset caps |
| 3. Dataset Exploration | File collection, split construction, class balance statistics, duration distribution plots |
| 4. Multi-Resolution Spectrogram Visualization | Side-by-side real vs. fake spectrograms at all three scales |
| 5. Data Augmentation | Additive noise, channel simulation, time stretching, SpecAugment |
| 6. Dataset & DataLoaders | `FastAudioDataset`, batched GPU mel extraction, weighted sampler for class balance |
| 7. Model Architecture | `ScaleEncoder`, `ConformerBlock`, `CrossScaleAttentionFusion`, `ConDetection` |
| 8. Training Infrastructure | Focal BCE loss, cosine LR schedule with warmup, AMP scaler, threshold search |
| 9. Training Loop | Per-epoch metrics, early stopping, checkpoint saving |
| 10. Training Curves | Loss, EER, AUC, AP, F1, Accuracy plotted across epochs |
| 11. Final Test Evaluation | In-domain (FoR) and out-of-domain (In-the-Wild) evaluation with threshold calibration |
| 12. Confusion Matrices & ROC Curves | Visual performance analysis for both test sets |
| 13. Classifier Comparison | ConDetection vs. Logistic Regression and Random Forest baselines on same features |
| 14. Statistical Significance Tests | McNemar's test and bootstrap AUC confidence intervals |
| 15. Cross-Validation | 3-fold CV on FoR for robustness estimates |
| 16. Grad-CAM Interpretability | Per-scale saliency maps overlaid on spectrograms |
| 17. Cross-Scale Attention Weights | Heatmaps of inter-resolution attention across samples |
| 18. Generalization Analysis | EER/AUC gap visualization across all models, addressing Müller et al. (2022) |
| 19. Research Summary | Aggregated results and list of all saved output files |

---

## Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `SR` | 16000 | Sample rate (Hz) |
| `DURATION` | 4 | Clip length (seconds) |
| `D_MODEL` | 96 | Conformer hidden dimension |
| `N_LAYERS` | 2 | Number of Conformer blocks |
| `N_HEADS` | 4 | Attention heads |
| `BATCH_SIZE` | 16 | Kaggle-safe value |
| `LR` | 3e-4 | Peak learning rate |
| `EPOCHS` | 20 | Max training epochs |
| `PATIENCE` | 6 | Early stopping patience |
| `FOCAL_GAMMA` | 1.5 | Focal loss gamma |
| `MIXSTYLE_P` | 0.50 | MixStyle application probability |

---

## Outputs

After a complete run, the following files are saved to `/kaggle/working`:

```
dataset_overview.png
multiresolution_spectrograms.png
augmentation_examples.png
training_curves.png
confusion_roc.png
generalization_analysis.png
gradcam_for_test.png
gradcam_in_the_wild.png
attention_weights_in_the_wild.png
classifier_comparison.csv
statistical_tests.csv
cross_validation.csv
checkpoints/
    best_model.pt
    best_model.pth
    best_checkpoint.pth
    best_threshold.txt
```

---

## References

- Müller, N. M. et al. (2022). *Does Audio Deepfake Detection Generalize?* — establishes the cross-domain generalization problem for ADD.
- Shin, Y. et al. (2023). *HM-Conformer* — hierarchical pooling + multi-scale conformer architecture for ASVspoof.
- Shahriar (2026). *Cross-Scale Attention for Audio Deepfake Detection* — multi-resolution fusion with CNN encoder.
- Park, D. S. et al. (2019). *SpecAugment* — frequency and time masking for data augmentation.