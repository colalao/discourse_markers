from .Japanese import load_Japanese
from .English import load_English


def load_multiple_datasets(datasets, split):
    dsets = []
    for d in datasets:
        if d == "Japanese":
            dsets.append(load_Japanese(split))
        elif d == "English":
            dsets.append(load_English(split))
    return dsets
