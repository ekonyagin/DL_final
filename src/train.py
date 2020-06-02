from config import cfg
from src.model.gan import HoloStyleGAN
from src.model.utils import save, prepare
from src.model.encoder import HoloEncoder, HoloEncoderLight
from src.model.stylegan_no_gen_training import StyleGAN2
from src.utils import experiment
from src.utils.loaders import make_loader
from src.utils.stages import train, test, sample


if __name__ == '__main__':
    experiment.save()
    artifacts = ['gan', 'g_opt', 'd_opt']

    model = HoloStyleGAN(cfg.ENCODER_PARAMETERS, cfg.MODEL_PARAMETERS)
    
    disc_opt = cfg.D_OPT(model.stylegan.discriminator.parameters())
    enc_opt = cfg.ENC_OPT(model.encoder.parameters())
    for artifact in artifacts:
        prepare(globals()[artifact], artifact)

    train_loader = make_loader('train')
    val_loader = make_loader('val')

    for epoch in range(cfg.N_EPOCHS):
        train(model, enc_opt, disc_opt, train_loader)
        test(model, val_loader)

        sample(model, val_loader)

        for artifact in artifacts:
            save(globals()[artifact], artifact)
