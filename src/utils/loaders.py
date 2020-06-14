from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose

from config import cfg
from .dataset import Images


def make_loader(stage: str) -> DataLoader:
    """
    Makes DataLoader according to the path specified in cfg for the given stage
    @param stage: can be eiher 'train', 'val' or 'test'
    @return: new DataLoader instance
    """
    if stage not in ('train', 'val', 'test'):
        raise ValueError

    root = getattr(cfg, stage.upper() + '_ROOT')

    transform = cfg.SHARED_TRANSFORM
    shuffle = False

    if stage == 'train':
        transform = [*cfg.TRAIN_TRANSFORM, *transform]
        shuffle = True

    dataset = Images(root, transform=Compose(transform))

    return DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=shuffle, num_workers=1)
