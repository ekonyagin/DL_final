import torch
from torch import nn


class HoloStyleGAN(nn.Module):
    def __init__(self, enc_params, stylegan_params):
        super().__init__()
        self.encoder = Encoder(**enc_params)
        self.stylegan = StyleGAN2(**stylegan_params)


    def forward(self, images: torch.Tensor):
        return x

    def compute_loss(self):
        pass