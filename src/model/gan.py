import torch
from torch import nn
from .encoder import HoloEncoder, HoloEncoderLight
from .stylegan_no_gen_training import StyleGAN2


class HoloStyleGAN(nn.Module):
    def __init__(self, enc_class, enc_params, stylegan_params, load_path=''):
        super().__init__()
        self.encoder = enc_class(**enc_params)
        self.stylegan = StyleGAN2(**stylegan_params)

        if load_path:
            self.stylegan.load_state_dict(torch.load(load_path)) 
            print('Checkpoints for StyleGAN2 loaded successfully.')

    def forward(self, images: torch.Tensor, angles=None):
        """
        angles : list of length=batch_size, consisting of angles for img rotation
        """
        if angles is None:
            angles = [0.0 for _ in range(images.shape[0])]
        return self.stylegan.G(self.encoder(images, angles))

    def compute_loss(self):
        pass
