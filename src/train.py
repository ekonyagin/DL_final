from config import cfg
from src.model.gan import HoloStyleGAN
from src.model.utils import save, prepare
from src.utils import experiment
from src.utils.loaders import make_loader
from src.utils.stages import train_on_batch, test, sample
from itertools import cycle
from tensorboardX import SummaryWriter
from tqdm import tqdm


if __name__ == '__main__':
    experiment.save()
    artifacts = ['model']

    model = HoloStyleGAN(cfg.ENCODER_CLASS,
                         cfg.ENCODER_PARAMETERS, cfg.STYLEGAN_PARAMETERS,
                         ckpt_path=cfg.STYLEGAN_CHECKPOINT_PATH,
                         emb_ckpt_path=cfg.EMB_CHECKPOINT_PATH)

    for artifact in artifacts:
        prepare(globals()[artifact], artifact)

    train_loader = cycle(make_loader('train'))
    val_loader = make_loader('val')

    writer = SummaryWriter(logdir=cfg.LOGDIR)
    for it in tqdm(range(cfg.N_ITERATIONS)):
        d_loss, rot0_loss, g_loss, emb_loss = train_on_batch(model, train_loader, it)
        # writer.add_scalars('All Losses', {
        #     'd_loss': d_loss,
        #     'rot0_loss': rot0_loss,
        #     'g_loss': g_loss,
        #     'emb_loss': emb_loss,
        # }, it)
        writer.add_scalar('Discriminator Loss', d_loss, it)
        writer.add_scalar('Rot0 Loss', rot0_loss, it)
        writer.add_scalar('Generator Loss', g_loss, it)
        writer.add_scalar('Embedding Loss', emb_loss, it)

        # test(model, val_loader)
        
        if (it + 1) % cfg.SAMPLE_EVERY == 0:
            samples = sample(model, val_loader, it, return_img=True)
            writer.add_image('Samples', samples, it)

        if (it + 1) % cfg.SAVE_EVERY == 0:
            for artifact in artifacts:
                save(globals()[artifact], artifact)
    writer.close()

