from config import cfg
from src.model.gan import HoloStyleGAN
from src.model.utils import save, prepare
from src.model.encoder import HoloEncoder, HoloEncoderLight
from src.model.stylegan_no_gen_training import StyleGAN2
from src.utils import experiment
from src.utils.loaders import make_loader
from src.utils.stages import train, test, sample
from tensorboardX import SummaryWriter
from tqdm import tqdm


if __name__ == '__main__':
    experiment.save()
    artifacts = ['model', 'disc_opt', 'enc_opt']

    model = HoloStyleGAN(cfg.ENCODER_CLASS,
                         cfg.ENCODER_PARAMETERS, cfg.STYLEGAN_PARAMETERS,
                         load_path=cfg.STYLEGAN_CHECKPOINT_PATH)
    
    #if STYLEGAN_CHECKPOINT_PATH:
        # sg = StyleGAN2(cfg.STYLEGAN_PARAMETERS)
        # sg.load_state_dict
     #   model.G.load_state_dict(torch.load(STYLEGAN_CHECKPOINT_PATH))
      #  model.D.load_state_dice(torch.load(STYLEGAN_CHECKPOINT_PATH))
    disc_opt = cfg.D_OPT(model.stylegan.D.parameters())
    enc_opt = cfg.ENC_OPT(model.encoder.parameters())

    for artifact in artifacts:
        prepare(globals()[artifact], artifact)

    train_loader = make_loader('train')
    train_iter = iter(train_loader)
    val_loader = make_loader('val')

    writer = SummaryWriter(logdir=cfg.LOGDIR)
    for it in tqdm(range(cfg.N_ITERATIONS)):
        try:
            images = next(train_iter).to(cfg.DEVICE)
        except StopIteration:
            train_iter = iter(train_loader)
            images = next(train_iter).to(cfg.DEVICE)

        d_loss, rot0_loss, g_loss = train(model, enc_opt, disc_opt, images, it)
        writer.add_scalar('Discriminator Loss', d_loss, it)
        writer.add_scalar('Rot0 Loss', rot0_loss, it)
        writer.add_scalar('Generator Loss', g_loss, it)
        writer.add_scalars('combined', {
            'd_loss' : d_loss,
            'rot0_loss' :  rot0_loss,
            'g_loss' : g_loss
            }, it)

        # test(model, val_loader)
        
        if (it + 1) % cfg.SAMPLE_EVERY == 0:
            samples = sample(model, val_loader, it, return_img=True)
            writer.add_image('Samples', samples, it)

        if (it + 1) % cfg.SAVE_EVERY == 0:
            for artifact in artifacts:
                save(globals()[artifact], artifact)
    writer.close()

