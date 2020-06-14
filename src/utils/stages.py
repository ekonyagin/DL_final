import torch
from torch import nn
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import torch.nn.functional as F
from config import cfg

#
# def gen_iter():
#     noise = torch.FloatTensor(batch_size, image_size, image_size, 1).uniform_(0., 1.).cuda()
#     thetas = torch.randint(-30,30, size=(batch_size,)).type(torch.float32) #.to(cfg.DEVICE)
#     w_styles = encoder(images, thetas)
#
#     generated_images = stylegan.G(w_styles, noise)
#     fake_output, fake_q_loss = stylegan.D(generated_images.clone().detach())
#     g_loss = fake_output.mean()
#     g_loss.backward()
#     enc_opt.step()
#
# def rot0_iter():
#     noise = torch.FloatTensor(batch_size, image_size, image_size, 1).uniform_(0., 1.).cuda()
#     thetas = torch.zeros(batch_size)
#     w_styles = encoder(images, thetas)
#
#     generated_images = stylegan.G(w_styles, noise)
#     rot0_loss = rot0_loss_fn(generated_images, images)
#
#
# def disc_iter():
#     noise = torch.FloatTensor(batch_size, image_size, image_size, 1).uniform_(0., 1.).cuda()
#     thetas = torch.randint(-30,30, size=(batch_size,)).type(torch.float32) #.to(cfg.DEVICE)
#     w_styles = encoder(images, thetas)
#
#     generated_images = stylegan.G(w_styles, noise)
#     fake_output, fake_q_loss = stylegan.D(generated_images.clone().detach())
#
#     real_output, real_q_loss = stylegan.D(images)
#     disc_loss = (F.relu(1 + real_output) + F.relu(1 - fake_output)).mean()


def train_on_batch(model: nn.Module, loader: torch.utils.data.DataLoader, current_iteration: int):
    """
    Conducts training of HoloEncoder and StyleGAN2 discriminators in three modes:
     - pass real images directly to discriminator
     - pass rotated through HoloEncoder vectors to StyleGAN
     - pass NOT rotated HoloEncoder to StyleGAN and count L1 loss
    
    Inputs: encoder - nn.Module HoloEncoder, stylegan - nn.Module StyleGAN with frozen generator, 
    enc_opt - torch.optim optimizer for encoder, gan_opt - torch.optim optimizer for discriminator, 
    loader - iter(torch.utils.data.dataloader.DataLoader),
    ,
    save_every - int (optional) frequency of checkpoints saving
    for images loading
    """

    encoder = model.encoder
    stylegan = model.stylegan
    STEPS = current_iteration

    encoder.train()

    def make_g_forward(images, rotate=True):
        noise = torch.FloatTensor(cfg.BATCH_SIZE, cfg.IMG_SIZE, cfg.IMG_SIZE, 1).uniform_(0., 1.).to(cfg.DEVICE)
        if rotate:
            thetas = torch.randint(cfg.MIN_ANGLE, cfg.MAX_ANGLE, size=(cfg.BATCH_SIZE,)).to(torch.float32)
        else:
            thetas = torch.zeros(cfg.BATCH_SIZE).to(torch.float32)
        w_styles = encoder(images, thetas)
        return stylegan.G(w_styles, noise)

    similarity = nn.L1Loss()

    images = next(loader).to(cfg.DEVICE)

    ### Train G ###
    # stylegan.G.unfreeze_()
    # stylegan.D.freeze_()

    # GAN loss
    generated_images = make_g_forward(images)
    fake_output, fake_q_loss = stylegan.D(generated_images.clone().detach())
    g_loss = fake_output.mean()
    g_loss.backward()

    # Identity L1
    generated_images = make_g_forward(images, rotate=False)
    rot0_loss = cfg.ROT0_LOSS_COEF * similarity(generated_images, images)
    rot0_loss.backward()

    # Identity InsightFace

    emb_loss = torch.tensor(0)
    if current_iteration > cfg.EMB_LOSS_START_ITER:
        emb_original = model.embedder(images)
        generated_images = make_g_forward(images)
        emb_generated = model.embedder(generated_images)
        emb_loss = cfg.EMB_LOSS_COEF * similarity(emb_generated, emb_original)
        emb_loss.backward()

    # g_loss = g_loss + rot0_loss + emb_loss
    # g_loss.backward()

    # Steps
    encoder.step()
    stylegan.G.step()
    
    ### Train D ###
    # stylegan.G.freeze_()
    # stylegan.D.unfreeze_()

    generated_images = make_g_forward(images)
    fake_output, fake_q_loss = stylegan.D(generated_images.clone().detach())
    real_output, real_q_loss = stylegan.D(images)

    d_loss = (F.relu(1 + real_output) + F.relu(1 - fake_output)).mean()
    d_loss.backward()
    stylegan.D.opt.step()

    # quantize_loss = (fake_q_loss + real_q_loss).mean()
    # q_loss = float(quantize_loss.detach().item())
    # disc_loss = disc_real_loss + quantize_loss + rot0_loss
    # disc_loss.register_hook(raise_if_nan)
    # total_disc_loss = disc_loss.detach().item()
    # disc_loss.backward()
    # enc_opt.step()
    # if not cfg.STYLEGAN_FIXD:
    #    gan_opt.step()
    
    # generated_images = stylegan.G(w_styles, noise)

    if STEPS % 10 == 0 and STEPS > 20000:
        stylegan.EMA()

    if STEPS <= 25000 and STEPS % 1000 == 2:
        stylegan.reset_parameter_averaging()
    
    """
    if torch.isnan(total_disc_loss):
        print(f'NaN detected for discriminator. Loading from checkpoint #{checkpoint_num}')
        self.load(checkpoint_num)
        raise NanException

    periodically save results
    

    if self.steps % 1000 == 0 or (self.steps % 100 == 0 and self.steps < 2500):
        self.evaluate(floor(self.steps / 1000))
    """
    # STEPS += 1
    #self.av = None
    return d_loss.item(), rot0_loss.item(), g_loss.item(), emb_loss.item()

def test(model: nn.Module, loader: torch.utils.data.dataloader.DataLoader):
    model.eval()

    for images in tqdm(loader):
        images.to(cfg.DEVICE)
        with torch.no_grad():
            pass


def sample(model: nn.Module, loader: torch.utils.data.dataloader.DataLoader,
           it, return_img=False):
    """
    Samples random rotations of the first batch of images from the loader
    """

    encoder = model.encoder
    stylegan = model.stylegan
    model.eval()

    angles = torch.linspace(cfg.MIN_ANGLE, cfg.MAX_ANGLE, steps=5, dtype=torch.float32).view(-1, 1)
    images = next(iter(loader))

    # from pdb import set_trace; set_trace()
    samples = []
    for image in images:
        samples.append(image)
        image = image.to(cfg.DEVICE)
        noise = torch.FloatTensor(1, cfg.IMG_SIZE, cfg.IMG_SIZE, 1).uniform_(0., 1.).cuda()
        for angle in angles:
            with torch.no_grad():
                w_style = encoder(image[None, ...], angle)
            samples.append(stylegan.G(w_style, noise).detach().cpu()[0])
    
    samples = torch.stack(samples)
    save_image(samples, fp=cfg.SAMPLES_ROOT / f'{it}.png', nrow=6, padding=1)
    if return_img:
        return make_grid(samples, nrow=6, padding=1)
