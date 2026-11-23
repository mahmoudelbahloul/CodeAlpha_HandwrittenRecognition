# Handwritten Character Recognition (CNN on EMNIST)

CodeAlpha Machine Learning internship, **Task 3**.

A convolutional neural network that reads handwritten **digits and letters**. A first baseline CNN scored **87.2 %** on 18,800 unseen test images. A second round of experiments, adding one improvement at a time and measuring each, brought an ensemble of three CNNs to **90.5 %**. When the kind of character is known (a digit field or a letter field), a simple restriction lifts digit accuracy to **99.6 %**.

## Dataset

[EMNIST](https://arxiv.org/abs/1702.05373) *Balanced* split (Kaggle copy by Chris Crawford): 28x28 grayscale images, **47 classes** = 10 digits + 37 letter classes. Look-alike upper/lower-case pairs (C/c, I/i, J/j, K/k, L/l, M/m, O/o, P/p, S/s, U/u, V/v, W/w, X/x, Y/y, Z/z) are merged into one class, so the model predicts `C`, not "c or C".

- 112,800 training images (101,520 for fitting + 11,280 stratified validation) and 18,800 test images (400 per class).
- EMNIST stores the images transposed, so the notebooks swap the two image axes before use.
- Pixels are scaled to 0-1.
- Every experiment uses the same split (seed 42). Choices are made on the **validation** set; the test set is only reported.

## Round 1: baseline CNN (`notebooks/01_baseline_cnn.ipynb`)

Conv(32)-Pool-Conv(64)-Pool-Dense(128)-Dropout(0.4)-Dense(47): 229,807 parameters, Adam, batch 128, early stopping, 8 epochs on a CPU.

Test accuracy **0.8719** (macro F1 0.870, test loss 0.3728). Validation 0.8757, training 0.8583 (below validation only because dropout is active while training).

Why a CNN and not a plain dense network: flattening an image into a vector throws away which pixels are neighbours. Convolutions keep that spatial structure, so the same stroke pattern is recognised wherever it appears.

## Round 2: improving it (`notebooks/02_improved_cnn.ipynb`)

One change at a time, same split, run on a Kaggle GPU. The baseline scores 0.8751 here instead of 0.8719 because GPU training is not bit-for-bit reproducible; that 0.3-point wobble is a useful reminder of how small differences should be read.

| Run | What changed | Epochs | Val acc | Test acc |
|---|---|---|---|---|
| E0 | Baseline (v1 recipe) | 8 | 0.8778 | 0.8751 |
| E1 | + BatchNorm, 2 conv layers per block (881k params) | 8 | 0.8921 | 0.8914 |
| E1b | E1 trained longer (control) | 20 | 0.9002 | 0.8963 |
| E2 | E1b + strong augmentation | 17 | 0.8875 | 0.8839 |
| E3 | E2 + learning-rate schedule | 29 | 0.8921 | 0.8913 |
| E3b | E1 + learning-rate schedule, **no** augmentation | 26 | 0.9024 | 0.9010 |
| E3c | E1 + schedule + **mild** augmentation | 30 | 0.8999 | 0.8992 |
| E4 | **Ensemble of 3 models** (E3b recipe, seeds 42/43/44) | - | **0.9053** | **0.9047** |

The three ensemble members score 0.9010, 0.9018 and 0.9026 on their own; averaging their probabilities gives 0.9047.

What worked:
- **BatchNorm and a deeper network:** +1.6 points over the baseline at the same 8 epochs.
- **Longer training with a learning-rate schedule:** another +1.0 point (E1 0.8914 -> E3b 0.9010). The control run E1b (more epochs, no schedule, 0.8963) shows that about half of it is simply "more epochs".
- **Ensembling:** about +0.2 to +0.4 points over a single model.

What did **not** work: **data augmentation** (random rotation, shift and zoom) made the model *worse*, both strong (E2, E3) and mild (E3c). A likely reason, not tested here: EMNIST images are already centred and size-normalised, so random shifts and zooms create training images that look different from the test images. Without augmentation the model was also much faster to train (3.6 min vs 8 min).

For context, published CNN results on this split are mostly in the high 80s to about 90 % (for example 88.3 % and 90.5 % in the TextCaps paper's comparison), so 90.5 % is at the upper end of what a plain CNN reaches on this data.

### Where the final ensemble still fails

Hardest characters by F1: **L 0.58, 1 0.66, F 0.67, O 0.68, q 0.69, I 0.70, f 0.70, 0 0.72**.

Most frequent mistakes (true -> predicted, out of 400 test images per class):

| True | Predicted | Times |
|---|---|---|
| F | f | 154 |
| L | 1 | 148 |
| O | 0 | 143 |
| I | 1 | 95 |
| q | 9 | 93 |
| f | F | 80 |
| 0 | O | 75 |
| L | I | 57 |

Most of these are not a weakness of the network: a handwritten `0` and `O`, or `1`, `I` and `l`, are the same shape. The pixels simply do not contain the answer. That is why the extra points from a better network are getting harder to win, and why context (next section) helps more.

## Using context: digits vs letters

Because the image alone cannot separate 0 from O, we add information from outside the image: **which kind of character the field expects**. If a form field is a phone number, only the 10 digit classes are allowed; the probabilities of the other 37 classes are set to zero before taking the best one. No retraining is needed.

| Test samples | Baseline (v1) | Ensemble, no restriction | Ensemble, restricted |
|---|---|---|---|
| Digits (4,000) | 0.875 -> 0.9915 | 0.9152 | **0.9962** |
| Letters (14,800) | 0.871 -> 0.9067 | 0.9018 | **0.9343** |

(For the baseline, the arrow shows unrestricted -> restricted.) Digits gain a lot because the confusions with letters (0 vs O, 1 vs I) disappear once letters are not allowed. Letters gain less, because many of their errors are between two letters (F/f, L/I), which this restriction cannot fix.

## Repository

| File | Purpose |
|---|---|
| `notebooks/01_baseline_cnn.ipynb` | Round 1: data, baseline model, evaluation, error analysis, restriction trick |
| `notebooks/02_improved_cnn.ipynb` | Round 2: ablation study, ensemble, error analysis, restriction trick |
| `models/member_0.keras`, `member_1.keras`, `member_2.keras` | The three ensemble models (3.6 MB each) |
| `models/baseline_cnn.keras` | The single baseline model from round 1 |
| `label_mapping.json` | Class index -> character |
| `predict.py` | Command-line prediction for your own image (averages the three models) |
| `requirements.txt` | Dependencies |

## How to run

```bash
pip install -r requirements.txt

# any character
python predict.py my_character.png

# the field is known to hold a digit (or a letter)
python predict.py my_character.png --only digits
python predict.py my_character.png --only letters

# use the single baseline model instead of the ensemble
python predict.py my_character.png --model models/baseline_cnn.keras
```

`predict.py` resizes the image to 28x28 grayscale and inverts it, so use a dark character on a light background. A hand-drawn ring is a good test: unrestricted the ensemble answers `0` (54 %) or `O` (46 %); with `--only digits` it is `0`, with `--only letters` it is `O`.

To retrain, open a notebook on Kaggle, attach the *EMNIST (Extended MNIST)* dataset by Chris Crawford as an input, and run all cells (round 2 is meant for a GPU).

## Limitations

- **Merged cases:** the Balanced split cannot tell `C` from `c`, so the model cannot either.
- **Single characters only:** it reads one cropped character at a time, not words or lines.
- **Style shift:** EMNIST comes from US census forms. Handwriting that is very different (other scripts, unusual pens, photos with background) will do worse, and the model is untested on it.
- **Ceiling:** the remaining errors are mostly look-alike pairs (F/f, 0/O, 1/I/L). More network capacity alone is unlikely to fix them.
- **Small differences:** the experiments use one run per setting (plus three seeds for the ensemble). Gaps of a few tenths of a point, such as E3b vs E3c, are within run-to-run noise.
- **Digits-vs-letters restriction needs the field type to be known** and is only as reliable as that knowledge.

## Reference

Cohen, G., Afshar, S., Tapson, J., van Schaik, A. (2017). *EMNIST: an extension of MNIST to handwritten letters.* arXiv:1702.05373
