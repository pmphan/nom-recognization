import json
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from itertools import chain

from utils.utils import get_alphabet

class NomDataset(Dataset):
    def __init__(self, names, transforms=None, target_transforms=None):
        super().__init__()
        if isinstance(names, str):
            names = [names]
        self.names = names
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.root = Path("./images")
        self.imgs = list(sorted(chain.from_iterable([self.root.glob(f"{n}/*.jpg") for n in names])))

        annotation_files = list(sorted(chain.from_iterable([self.root.glob(f"{n}/annotations.json") for n in names])))
        self.alphabet_dict = get_alphabet(annotation_files)

        self.annotations = {}
        for file in annotation_files:
            with open(file) as annotf:
                self.annotations.update(json.load(annotf))
        for key, vals in self.annotations.items():
            label, length = self.make_label(vals['hn_text'])
            self.annotations[key] = {
                "label": label,
                "length": length
            }

    def make_label(self, l):
        label = list(map(self.alphabet_dict.get, l))
        #TODO: Prevent hardcodiing
        if len(label) < 13:
            label += [0] * (13-len(label))
        return label, len(l)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        annotation = self.annotations[img_path.name]
        label, length = annotation["label"], annotation["length"]
        img = Image.open(img_path)

        if self.transforms:
            img = self.transforms(img)

        if self.target_transforms:
            label = self.target_transforms(label)

        return img, label, length

    def __len__(self):
        return len(self.imgs)

