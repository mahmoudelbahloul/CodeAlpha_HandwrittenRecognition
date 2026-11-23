"""Recognise a handwritten character with the saved CNN ensemble.

Usage:
    python predict.py path/to/image.png                 # any of the 47 classes
    python predict.py path/to/image.png --only digits   # the field is known to hold a digit
    python predict.py path/to/image.png --only letters  # the field is known to hold a letter
    python predict.py path/to/image.png --model models/baseline_cnn.keras   # the single v1 model

By default the three models in models/member_*.keras are loaded and their probabilities are averaged.
The image should show a dark character on a light background (like ink on paper).
It is converted to 28x28 grayscale and inverted, because EMNIST uses white strokes on black.
"""
import argparse
import json

import numpy as np
from PIL import Image
from tensorflow import keras

DEFAULT_MODELS = [f"models/member_{i}.keras" for i in range(3)]


def load_image(path):
    img = Image.open(path).convert("L").resize((28, 28))
    x = 1.0 - np.asarray(img, dtype="float32") / 255.0  # white strokes on black, like EMNIST
    return x[None, :, :, None]


def predict(models, labels, x, only=None, top=3):
    probs = np.mean([m.predict(x, verbose=0)[0] for m in models], axis=0)  # ensemble = average
    if only:
        is_digit = np.array([labels[i].isdigit() for i in range(len(labels))])
        probs = np.where(is_digit if only == "digits" else ~is_digit, probs, 0.0)
        probs = probs / probs.sum()  # re-normalise over the allowed classes
    best = np.argsort(probs)[::-1][:top]
    return [(labels[i], float(probs[i])) for i in best]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--only", choices=["digits", "letters"])
    ap.add_argument("--model", nargs="+", default=DEFAULT_MODELS, help="one or more .keras files (averaged)")
    ap.add_argument("--labels", default="label_mapping.json")
    args = ap.parse_args()

    models = [keras.models.load_model(p) for p in args.model]
    mapping = json.load(open(args.labels))
    labels = [mapping[str(i)] for i in range(len(mapping))]
    for char, p in predict(models, labels, load_image(args.image), args.only):
        print(f"{char!r}: {p:.1%}")
