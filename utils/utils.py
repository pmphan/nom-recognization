import cv2
import json
import numpy as np
from functools import reduce

def get_alphabet(annotation_files):
    ALPHABET : set[str] = set()
    for annotation_file in annotation_files:
        with open(annotation_file) as file:
            label_dict = json.load(file)
            all_labels = reduce(lambda x, y: x+y, (val['hn_text'] for val in label_dict.values()), [])
            ALPHABET.update(all_labels)

    ALPHABET_DICT = {c:i+1 for i, c in enumerate(ALPHABET)}
    assert '@' not in ALPHABET, "Please pick another blank character."
    ALPHABET_DICT['@'] = 0

    return ALPHABET_DICT

def image_padder(im, H, W, value=None):
    """Work in progress."""
    old_H, old_W = im.shape[:2]
    if old_W > W:
        ratio = W / old_W
        im = cv2.resize(im, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    image = np.zeros((H, W, 3), np.uint8)
    if value is None:
        a2D = im.reshape(-1, im.shape[-1])
        col_range = (256, 256, 256)
        a1D = np.ravel_multi_index(a2D.T, col_range)
        value = np.unravel_index(np.bincount(a1D).argmax(), col_range)

    # Randomize value
    lower = np.array([value[0] - 20, value[1] - 20, value[2] - 20])
    upper = np.array([value[0] + 20, value[1] + 20, value[2] + 20])
    mask = cv2.inRange(im, lower, upper)
    masked_image = np.copy(im)
    masked_image[mask != 0] = value
    image[:] = value

    x_offset, y_offset = int((W - old_W) / 2), 10
    image[y_offset:y_offset + old_H, x_offset:x_offset + old_W] = masked_image.copy()
    return image

