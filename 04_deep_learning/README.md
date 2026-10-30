# Part IV — Deep Learning: U-Net for Ultrasound Nerve Segmentation

## Objective

Implement a U-Net **from scratch in PyTorch** to segment brachial plexus nerves in grayscale ultrasound images. This project follows earlier experiments with logistic regression, random forests, and multilayer perceptrons.

**Dataset:** [Kaggle — Ultrasound Nerve Segmentation](https://www.kaggle.com/competitions/ultrasound-nerve-segmentation/data). Download the labeled `train.zip` archive separately; images and annotations are not redistributed here.

## Method

1. Pair `patient_image.tif` files with their `patient_image_mask.tif` masks.
2. Split train / validation / test using `GroupShuffleSplit` and patient identifiers extracted from filenames. No patient appears in multiple splits. The split allocates approximately 64% / 16% / 20% of **patients** (image proportions may differ).
3. Resize to 256 × 256 pixels: bilinear interpolation for ultrasound images, nearest-neighbor for binary masks.
4. Normalize grayscale intensities to [0, 1]; use binary float32 target masks.
5. Train a custom U-Net with encoder channels 32, 64, 128; bottleneck 256; transposed-convolution decoder and skip connections.
6. Address severe class imbalance using **balanced sampling of positive-mask and empty-mask images**, with replacement, and a **BCE + soft Dice loss**.
7. Select the checkpoint with the lowest validation loss. Report mean per-image Dice on all images and separately on positive-mask images, along with positive-image detection and correct empty-mask prediction rates.

The baseline used ordinary BCE and random batches. It mostly predicted empty masks: **58.05%** of test masks were empty, **94.91%** of predictions were empty, and overall test Dice was **0.5624**. This illustrates why a seemingly acceptable Dice can be misleading when many masks are empty.

## Preliminary results: improved model, 3 epochs

The revised model was run locally for **3 epochs**. Values below are approximate, read from the shared training plots:

| Metric | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---:|---:|---:|
| Validation Dice, positive masks | ~0.61 | ~0.67 | ~0.64 |
| Validation Dice, all masks | ~0.31 | ~0.40 | ~0.52 |
| Validation BCE + Dice loss | ~0.79 | ~0.73 | ~0.74 |

Qualitative predictions on three **positive-mask** examples show the model locating the annotated nerve regions, with imperfect boundaries. This is a meaningful improvement over the all-black baseline, but the model is **not yet fully evaluated**. The positive-detection and empty-mask-specific rates from the final test evaluation have not been provided, so no revised test-set Dice is claimed.

### Run

Requirements: Python, PyTorch, Pillow, NumPy, Matplotlib, scikit-learn.

```bash
pip install torch pillow numpy matplotlib scikit-learn
python 04_deep_learning/unet.py --data-dir "C:/path/to/train" --epochs 3
```

The script accepts `--epochs`, `--batch-size`, `--image-size`, `--lr`, `--seed`, and `--output-dir`. It uses CUDA if available, otherwise CPU, and `num_workers=0` for Windows compatibility.

Outputs in `unet_outputs/`: `unet_best.pth`, `loss_curves.png`, `dice_curve.png`, and `positive_predictions.png`. Checkpoint files and dataset images should not be committed to the repository.

## Limitations and next experiments

- **Only 3 epochs** for the revised model; longer training and multiple random seeds are needed.
- The training sampler deliberately changes the positive/empty-mask balance; validation and testing retain their natural distributions.
- A mask-empty/mask-empty pair scores Dice 1, which can inflate overall Dice. Always report positive-mask Dice and empty-mask performance separately.
- Positive-image detection (at least one predicted positive pixel) is not equivalent to correct localization.
- Future work: longer runs on GPU, data augmentation, threshold calibration on validation data, and evaluation of false positives on empty images.
- The official Kaggle test images do not provide public ground-truth masks; the reported local evaluation uses a held-out subset of the labeled training archive.

**Project status:** completed baseline and improved training implementation; further optimization optional.
