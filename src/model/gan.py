import torch
from torch import nn
from .encoder import HoloEncoderLight as Encoder
from .stylegan_no_gen_training import StyleGAN2


class HoloStyleGAN(nn.Module):
    def __init__(self, enc_params, stylegan_params):
        super().__init__()
        self.encoder = Encoder(**enc_params)
        self.stylegan = StyleGAN2(**stylegan_params)


    def forward(self, images: torch.Tensor, angles=None):
        """
        angles : list of length=batch_size, consisting of angles for img rotation
        """
        if angles==None:
            angles = [0.0 for _ in range(img.shape[0])]
        return stylegan.G(encoder(images, angles))

    def compute_loss(self):
        pass
