from config import cfg
from src.model.utils import save, prepare
from src.model.gan import HoloStyleGAN
from src.utils import experiment
from src.utils.loaders import make_loader
from src.utils.stages import train, test, sample


if __name__ == '__main__':
    experiment.save()
    artifacts = ['gan', 'g_opt', 'd_opt']

    gan = HoloStyleGAN(**cfg.MODEL_PARAMETERS)
    g_opt = cfg.G_OPT(gan.generator.parameters())
    d_opt = cfg.D_OPT(gan.discriminator.parameters())

    for artifact in artifacts:
        prepare(globals()[artifact], artifact)

    train_loader = make_loader('train')
    val_loader = make_loader('val')

    for epoch in range(cfg.N_EPOCHS):
        train(gan, train_loader)
        test(gan, val_loader)

        sample(gan, val_loader)

        for artifact in artifacts:
            save(globals()[artifact], artifact)
