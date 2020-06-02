import torch
from torch import nn
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from config import cfg


def train(model: nn.Module,
          enc_opt: torch.optim, gan_opt: torch.optim, 
          loader: torch.utils.data.dataloader.DataLoader, save_every=1000):
    """
    Conducts training of HoloEncoder and StyleGAN2 discriminators in three modes:
     - pass real images directly to discriminator
     - pass rotated through HoloEncoder vectors to StyleGAN
     - pass NOT rotated HoloEncoder to StyleGAN and count identity loss (L1 loss)
    
    Inputs: encoder - nn.Module HoloEncoder, stylegan - nn.Module StyleGAN with frozen generator, 
    enc_opt - torch.optim optimizer for encoder, gan_opt - torch.optim optimizer for discriminator, 
    loader - torch.utils.data.dataloader.DataLoader,
    save_every - int (optional) frequency of checkpoints saving
    for images loading
    """
    #def model_name(num):
    #    return str(/ name / f'model_{num}.pt')
    encoder = model.encoder
    stylegan = model.stylegan
    encoder.train()
    stylegan.D.train()
    stylegan.G.eval()
    
    STEPS = 0
    
    id_loss_fn = nn.L1Loss()
    
    for images in tqdm(loader):
        images.to(cfg.DEVICE)
        total_disc_loss = torch.tensor(0.).cuda()
        total_gen_loss = torch.tensor(0.).cuda()

        batch_size = images.shape[0]

        image_size = stylegan.G.image_size
        latent_dim = stylegan.G.latent_dim
        num_layers = stylegan.G.num_layers

        gan_opt.zero_grad()
        enc_opt.zero_grad()
        
        #rotated_images
        
        
        noise = torch.FloatTensor(batch_size, image_size, image_size, 1).uniform_(0., 1.).cuda()
        thetas = torch.randint(-30,30, size=(batch_size,)).type(torch.float32)
        w_styles = encoder(images, thetas)
        
        generated_images = stylegan.G(w_styles, noise)
        fake_output, fake_q_loss = stylegan.D(generated_images.clone().detach())
        
        #identity transform
        noise = torch.FloatTensor(batch_size, image_size, image_size, 1).uniform_(0., 1.).cuda()
        thetas = torch.zeros(batch_size)
        w_styles = encoder(images, thetas)
        
        generated_images = stylegan.G(w_styles, noise)
        
        id_loss = id_loss_fn(generated_images, images)
        
        ######real_images#######
        
        real_output, real_q_loss = stylegan.D(images)

        disc_loss = (F.relu(1 + real_output) + F.relu(1 - fake_output)).mean()
        

        quantize_loss = (fake_q_loss + real_q_loss).mean()
        self.q_loss = float(quantize_loss.detach().item())

        disc_loss = disc_loss + quantize_loss + id_loss
        
        disc_loss.register_hook(raise_if_nan)
        
        total_disc_loss = disc_loss.detach().item()
        
        disc_loss.backward()
        enc_opt.step()
        gan_opt.step()
        
        generated_images = self.GAN.G(w_styles, noise)

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
        STEPS += 1
        #self.av = None


def test(model: nn.Module, loader: torch.utils.data.dataloader.DataLoader):
    model.eval()

    for images in tqdm(loader):
        images.to(cfg.DEVICE)
        with torch.no_grad():
            pass


def sample(model: nn.Module, loader: torch.utils.data.dataloader.DataLoader):
    """
    Samples random rotations of the first batch of images from the loader
    """

    images = next(iter(loader))
    images.to(cfg.DEVICE)

    samples = model(images)
    for i, (image, sample) in enumerate(zip(images, samples)):
        # TODO: convert images to pil properly
        image = ToPILImage()(image.cpu())
        image.save(cfg.SAMPLES_ROOT / f'{i}_ref.png')

        sample = ToPILImage()(sample.cpu())
        sample.save(cfg.SAMPLES_ROOT / f'{i}.png')
