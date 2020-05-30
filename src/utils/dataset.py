from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data.dataset import Dataset


class Images(Dataset):
    def __init__(self, path: Path, transform: Callable = None):
        self.images = [*path.glob('*.jpg'), *path.glob('*.png')]
        self.transform = transform

    def __getitem__(self, item):
        image = Image.open(self.images[item])
        return self.transform(image)

    def __len__(self):
        return len(self.images)
