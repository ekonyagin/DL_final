import torch
from torch import nn


class HoloStyleGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

    def forward(self, images: torch.Tensor):
        return self.generator(images)

    def compute_loss(self):
        pass


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, images: torch.Tensor):
        return images


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

