from config import cfg
from src.model.utils import prepare
from src.model.gan import HoloStyleGAN
from src.utils.loaders import make_loader
from src.utils.stages import sample


if __name__ == '__main__':
    artifacts = ['gan', 'g_opt', 'd_opt']

    gan = HoloStyleGAN(**cfg.MODEL_PARAMETERS)
    g_opt = cfg.G_OPT(gan.generator.parameters())
    d_opt = cfg.D_OPT(gan.discriminator.parameters())

    for artifact in artifacts:
        prepare(globals()[artifact], artifact)

    val_loader = make_loader('val')
    sample(gan, val_loader)
