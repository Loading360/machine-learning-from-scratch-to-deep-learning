# Part IV — Deep Learning with PyTorch

## U-Net: ultrasound nerve segmentation

This project implements a PyTorch U-Net baseline for binary semantic segmentation of brachial plexus nerves in ultrasound images.

Dataset: Kaggle **Ultrasound Nerve Segmentation** (https://www.kaggle.com/competitions/ultrasound-nerve-segmentation/data). Download and extract the labeled training TIFF files separately. Do not upload the dataset to GitHub.

### Method

- Pair each image with its corresponding `_mask.tif` file.
- Use `GroupShuffleSplit` to keep patients disjoint across training, validation and testing (approximately 64%, 16%, 20% of patients).
- Resize images to 256 x 256 (bilinear), masks to 256 x 256 (nearest neighbor).
- Convert images to grayscale float32 in [0,1] and masks to binary tensors.
- Train a U-Net from scratch with encoder channels 32/64/128, bottleneck 256, and skip connections.
- Optimize BCEWithLogitsLoss using Adam. Select the best checkpoint using validation loss.
- Evaluate using BCE and Dice score on a held-out subset of labeled training data.

### Running

Install `torch`, `numpy`, `pillow`, `matplotlib`, and `scikit-learn`.

```bash
python 04_deep_learning/unet.py --data-dir "C:/path/to/train" --epochs 2
```

Increase to `--epochs 20` for a longer experiment. You can also configure `--batch-size`, `--lr`, and `--output-dir`.

By default, outputs are written to `unet_outputs/`: best checkpoint `unet_best.pth`, training curves, and example segmentation predictions.

### Status

**Implementation committed; training results not yet verified.** The official Kaggle test set has no public masks, so evaluation uses a held-out portion of the labeled training images.

Empty masks require care when interpreting Dice: an empty prediction against an empty mask scores 1. Future work includes separate positive-mask and empty-mask evaluations, augmentation, and BCE + Dice loss.
