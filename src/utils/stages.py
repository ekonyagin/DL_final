import torch
from torch import nn
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image, make_grid
# from tqdm import tqdm
import torch.nn.functional as F
# from tensorboardX import SummaryWriter

from config import cfg


def train(model: nn.Module,
          enc_opt: torch.optim, gan_opt: torch.optim, 
          images,
          current_iteration: int,
          save_every=1000):
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
    encoder.unfreeze_()
    if cfg.STYLEGAN_FIXD:
        stylegan.freeze_()
    else:
        stylegan.unfreeze_()
        
    STEPS = current_iteration
    
    rot0_loss_fn = nn.L1Loss()

    total_disc_loss = torch.tensor(0.).cuda()
    total_gen_loss = torch.tensor(0.).cuda()

    batch_size = images.shape[0]

    image_size = stylegan.G.image_size
    latent_dim = stylegan.G.latent_dim
    num_layers = stylegan.G.num_layers

    gan_opt.zero_grad()
    enc_opt.zero_grad()
    
    # TRAIN ENC+GEN
    stylegan.freeze_()
    encoder.unfreeze_()

    # random rotations
    noise = torch.FloatTensor(batch_size, image_size, image_size, 1).uniform_(0., 1.).cuda()
    thetas = torch.randint(-30,30, size=(batch_size,)).type(torch.float32) #.to(cfg.DEVICE)
    w_styles = encoder(images, thetas)

    generated_images = stylegan.G(w_styles, noise)
    fake_output, fake_q_loss = stylegan.D(generated_images.clone().detach())
    
    # identity transform
    noise = torch.FloatTensor(batch_size, image_size, image_size, 1).uniform_(0., 1.).cuda()
    thetas = torch.zeros(batch_size)
    w_styles = encoder(images, thetas)
    
    rot0_images = stylegan.G(w_styles, noise)
    
    rot0_loss = rot0_loss_fn(rot0_images, images)
    gen_loss = - (fake_output).mean() + rot0_loss.mean()
    gen_loss.backward()
    enc_opt.step()

    if not cfg.STYLEGAN_FIXD:
        # TRAIN DISCRIMINATOR
        stylegan.unfreeze_()
        encoder.freeze_()

        noise = torch.FloatTensor(batch_size, image_size, image_size, 1).uniform_(0., 1.).cuda()
        thetas = torch.randint(-30,30, size=(batch_size,)).type(torch.float32) #.to(cfg.DEVICE)
        w_styles = encoder(images, thetas)
        generated_images = stylegan.G(w_styles, noise)
        fake_output, fake_q_loss = stylegan.D(generated_images.clone().detach())
                         
        real_output, real_q_loss = stylegan.D(images)

        disc_loss = (F.relu(1 + real_output) + F.relu(1 - fake_output)).mean()
        
        # quantize_loss = (fake_q_loss + real_q_loss).mean()
        # q_loss = float(quantize_loss.detach().item())

        # disc_loss = disc_real_loss + quantize_loss + rot0_loss
        # disc_loss.register_hook(raise_if_nan)
        
       #  total_disc_loss = disc_loss.detach().item()
        
        disc_loss.backward()
        gan_opt.step()
    
    # generated_images = stylegan.G(w_styles, noise)

    if STEPS % 10 == 0 and STEPS > 20000:
        stylegan.EMA()

    if STEPS <= 25000 and STEPS % 1000 == 2:
        stylegan.reset_parameter_averaging()
        
    ##########PLACEHOLDER FOR SAVING##########
    #if STEPS % save_every == 0:
    #    torch.save(model.state_dict(), self.model_name(num))
    
    #checkpoint_num = floor(STEPS / self.save_every)
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
    return disc_loss.item(), rot0_loss.item(), gen_loss.item() 

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
    encoder.eval()
    stylegan.D.eval()
    stylegan.G.eval()

    angles = torch.linspace(-30,30, steps=5, dtype=torch.float32).view(-1, 1)
    images = next(iter(loader))
    # images.to(cfg.DEVICE)
    batch_size = images.size(0)
    image_size = images.size(2)

    # from pdb import set_trace; set_trace()
    samples = []
    for image in images:
        samples.append(image)
        image = image.to(cfg.DEVICE)
        noise = torch.FloatTensor(1, image_size, image_size, 1).uniform_(0., 1.).cuda()
        for angle in angles:
            w_style = encoder(image[None, ...], angle)            
            samples.append(stylegan.G(w_style, noise).detach().cpu()[0])
    
    samples = torch.stack(samples)
    save_image(samples, fp=cfg.SAMPLES_ROOT / f'{it}.png', nrow=6, padding=1)
    if return_img:
        return make_grid(samples, nrow=6, padding=1)
     # for i, (image, sample) in enumerate(zip(images, samples)):
     #   # TODO: convert images to pil properly
     #   image = ToPILImage()(image.cpu())
     #   image.save(cfg.SAMPLES_ROOT / f'{i}_ref.png')

     #   sample = ToPILImage()(sample.cpu())
     #   sample.save(cfg.SAMPLES_ROOT / f'{i}.png')
