import torch
from torch import nn
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from config import cfg


def train(model: nn.Module, loader: torch.utils.data.dataloader.DataLoader):
    model.train()

    for images in tqdm(loader):
        images.to(cfg.DEVICE)
        pass


def test(model: nn.Module, loader: torch.utils.data.dataloader.DataLoader):
    model.eval()

    for images in tqdm(loader):
        images.to(cfg.DEVICE)
        with torch.no_grad():
            pass


def sample(model: nn.Module, loader: torch.utils.data.dataloader.DataLoader):
    """
    Samples random rotations of the first batch of images from the loader
    """

    images = next(iter(loader))
    images.to(cfg.DEVICE)

    samples = model(images)
    for i, (image, sample) in enumerate(zip(images, samples)):
        # TODO: convert images to pil properly
        image = ToPILImage()(image.cpu())
        image.save(cfg.SAMPLES_ROOT / f'{i}_ref.png')

        sample = ToPILImage()(sample.cpu())
        sample.save(cfg.SAMPLES_ROOT / f'{i}.png')
