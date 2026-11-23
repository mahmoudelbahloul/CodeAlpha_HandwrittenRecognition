# Handwritten Character Recognition (CNN on EMNIST)

CodeAlpha Machine Learning internship, **Task 3**.

A convolutional neural network that reads handwritten **digits and letters**. It reaches **87.2 % accuracy** on 18,800 test images it never saw, and the repo includes an error analysis and a simple "use the context" trick that lifts digit accuracy to **99.2 %**.

## Dataset

[EMNIST](https://arxiv.org/abs/1702.05373) *Balanced* split (Kaggle copy by Chris Crawford): 28x28 grayscale images, **47 classes** = 10 digits + 37 letter classes. Look-alike upper/lower-case pairs (C/c, I/i, J/j, K/k, L/l, M/m, O/o, P/p, S/s, U/u, V/v, W/w, X/x, Y/y, Z/z) are merged into one class, so the model predicts `C`, not "c or C".

- 112,800 training images (101,520 for fitting + 11,280 stratified validation) and 18,800 test images.
- EMNIST stores the images transposed, so the notebook swaps the two image axes before use.
- Pixels are scaled to 0-1.

## Model

| Layer | Output | Params |
|---|---|---|
| Conv2D(32, 3x3, ReLU) + MaxPool | 13x13x32 | 320 |
| Conv2D(64, 3x3, ReLU) + MaxPool | 5x5x64 | 18,496 |
| Flatten, Dense(128, ReLU) | 128 | 204,928 |
| Dropout(0.4) | 128 | 0 |
| Dense(47, softmax) | 47 | 6,063 |

229,807 parameters (about 0.9 MB). Adam, sparse categorical cross-entropy, batch size 128, early stopping on validation loss. Trained for 8 epochs on a CPU (Kaggle, about 45 s per epoch).

Why a CNN and not a plain dense network: flattening an image into a vector throws away which pixels are neighbours. Convolutions keep that spatial structure, so the same stroke pattern is recognised wherever it appears.

## Results (test set, 18,800 images)

| Metric | Value |
|---|---|
| Accuracy | **0.8719** |
| Macro precision / recall / F1 | 0.874 / 0.872 / 0.870 |
| Test loss | 0.3728 |

Validation accuracy after the last epoch was 0.8757 and training accuracy 0.8583. Training accuracy sits below validation only because dropout is active during training; the gap is not overfitting, and the curves were still improving when the 8 epochs ended.

For context, published CNN results on this split are mostly in the high 80s to about 90 % (for example 88.3 % and 90.5 % in the TextCaps paper's comparison), so this simple model is in the normal range.

### Where the model fails

Hardest characters by F1: **L 0.48, q 0.59, F 0.60, 1 0.62, g 0.65, f 0.66, I 0.66, O 0.67**. Easiest: M, E, 7 (0.97) and W, 3 (0.98).

Most frequent mistakes (true -> predicted, out of 400 test images per class):

| True | Predicted | Times |
|---|---|---|
| F | f | 183 |
| L | 1 | 169 |
| O | 0 | 129 |
| q | 9 | 109 |
| 0 | O | 94 |
| I | 1 | 88 |
| 1 | I | 79 |
| L | I | 78 |

Most of these are not a weakness of the network: a handwritten `0` and `O`, or `1`, `I` and `l`, are the same shape. The pixels simply do not contain the answer.

## Using context: digits vs letters

Because the image alone cannot separate 0 from O, we add information from outside the image: **which kind of character the field expects**. If a form field is a phone number, only the 10 digit classes are allowed; the model's probabilities for the other 37 classes are set to zero before taking the best one. No retraining is needed.

| Test samples | Accuracy, no restriction | Accuracy, restricted |
|---|---|---|
| Digits (4,000) | 0.875 | **0.9915** |
| Letters (14,800) | 0.871 | **0.9067** |

Digits gain a lot because the confusions with letters (0 vs O, 1 vs I) disappear once letters are not allowed. Letters gain less, because many of their errors are between two letters (F/f, L/I), which this restriction cannot fix.

## Repository

| File | Purpose |
|---|---|
| `handwritten_character_recognition.ipynb` | Full notebook: data, model, training, evaluation, error analysis, restriction trick |
| `emnist_cnn.keras` | Trained model |
| `label_mapping.json` | Class index -> character |
| `predict.py` | Command-line prediction for your own image |
| `requirements.txt` | Dependencies |

## How to run

```bash
pip install -r requirements.txt

# any character
python predict.py my_character.png

# the field is known to hold a digit (or a letter)
python predict.py my_character.png --only digits
python predict.py my_character.png --only letters
```

`predict.py` resizes the image to 28x28 grayscale and inverts it, so use a dark character on a light background. A hand-drawn ring is a good test: unrestricted it comes out as `0` (51 %) or `O` (43 %); with `--only digits` it is `0`, with `--only letters` it is `O`.

To retrain, open the notebook on Kaggle, attach the *EMNIST (Extended MNIST)* dataset by Chris Crawford as an input, and run all cells.

## Limitations

- **Merged cases:** the Balanced split cannot tell `C` from `c`, so the model cannot either.
- **Single characters only:** it reads one cropped character at a time, not words or lines.
- **Style shift:** EMNIST comes from US census forms. Handwriting that is very different (other scripts, unusual pens, photos with background) will do worse, and the model is untested on it.
- **Short training:** 8 epochs on a CPU with no data augmentation; accuracy was still rising.
- **Digits-vs-letters restriction needs the field type to be known** and is only as reliable as that knowledge.

## Reference

Cohen, G., Afshar, S., Tapson, J., van Schaik, A. (2017). *EMNIST: an extension of MNIST to handwritten letters.* arXiv:1702.05373
