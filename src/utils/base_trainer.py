import os
import sys
import math
import fire
import json
from math import floor, log2
from random import random
from shutil import rmtree
from functools import partial
import multiprocessing

import numpy as np
import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as F

from torch_optimizer import DiffGrad
from torch.autograd import grad as torch_grad

import torchvision
from torchvision import transforms

from contrastive_learner import ContrastiveLearner

from PIL import Image
from pathlib import Path

from holo_encoder import Encoder

class NanException(Exception):
    pass

class Trainer():
    def __init__(self, 
                 name, 
                 results_dir, 
                 models_dir, 
                 image_size, 
                 network_capacity, 
                 transparent = False, 
                 batch_size = 4, 
                 mixed_prob = 0.9, 
                 gradient_accumulate_every=1, 
                 lr = 2e-4, num_workers = None, 
                 save_every = 1000, 
                 trunc_psi = 0.6, 
                 fp16 = False, 
                 cl_reg = False, 
                 fq_layers = [], 
                 fq_dict_size = 256, 
                 *args, **kwargs):
        self.GAN_params = [args, kwargs]
        self.GAN = None
        #self.Encoder = Encoder(*args)
        #self.Encoder_opt = None
        self.name = name
        self.results_dir = Path(results_dir)
        self.models_dir = Path(models_dir)
        self.config_path = self.models_dir / name / '.config.json'

        assert log2(image_size).is_integer(), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        self.image_size = image_size
        self.network_capacity = network_capacity
        self.transparent = transparent
        self.fq_layers = fq_layers if isinstance(fq_layers, list) else [fq_layers]
        self.fq_dict_size = fq_dict_size

        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixed_prob = mixed_prob

        self.save_every = save_every
        self.steps = 0

        self.av = None
        self.trunc_psi = trunc_psi

        self.gradient_accumulate_every = gradient_accumulate_every

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex is not available for you to use mixed precision training'
        self.fp16 = fp16

        self.cl_reg = cl_reg

        self.d_loss = 0
        self.g_loss = 0
        self.last_gp_loss = 0
        self.last_cr_loss = 0
        self.q_loss = 0
        self.pl_loss = 0

        self.pl_mean = torch.empty(1).cuda()
        self.pl_ema_decay = 0.99

        self.init_folders()

        self.loader = None
        self.id_loss = nn.L1Loss()

    def init_GAN(self):
        args, kwargs = self.GAN_params
        self.GAN = StyleGAN2(lr=self.lr, 
                             image_size = self.image_size, 
                             network_capacity = self.network_capacity, 
                             transparent = self.transparent, 
                             fq_layers = self.fq_layers, 
                             fq_dict_size = self.fq_dict_size, 
                             fp16 = self.fp16, 
                             cl_reg = self.cl_reg, *args, **kwargs)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.network_capacity = config['network_capacity']
        self.transparent = config['transparent']
        self.fq_layers = config['fq_layers']
        self.fq_dict_size = config['fq_dict_size']

        del self.GAN
        self.init_GAN()

    def config(self):
        return {'image_size': self.image_size,
                'network_capacity': self.network_capacity, 
                'transparent': self.transparent, 
                'fq_layers': self.fq_layers, 
                'fq_dict_size': self.fq_dict_size}

    def set_data_src(self, folder):
        self.dataset = Dataset(folder, self.image_size, transparent = self.transparent)
        self.loader = cycle(data.DataLoader(self.dataset, 
                                            num_workers = default(self.num_workers, num_cores), 
                                            batch_size = self.batch_size, 
                                            drop_last = True, 
                                            shuffle=True, 
                                            pin_memory=True))

    def train(self):
        assert self.loader is not None, 'You must first initialize the data source with `.set_data_src(<folder of images>)`'

        if self.GAN is None:
            self.init_GAN()

        self.GAN.D.train()
        self.GAN.G.eval()
        
        total_disc_loss = torch.tensor(0.).cuda()
        total_gen_loss = torch.tensor(0.).cuda()

        batch_size = self.batch_size

        image_size = self.GAN.G.image_size
        latent_dim = self.GAN.G.latent_dim
        num_layers = self.GAN.G.num_layers

        apply_gradient_penalty = self.steps % 4 == 0
        apply_path_penalty = self.steps % 32 == 0

        avg_pl_length = self.pl_mean
        self.GAN.D_opt.zero_grad()
        self.Encoder_opt.zero_grad()
        
        #get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
        #style = get_latents_fn(batch_size, num_layers, latent_dim)
        #w_space = latent_to_w(self.GAN.S, style)
        #w_styles = styles_def_to_tensor(w_space)
        
        #rotated_images
        image_batch = next(self.loader).cuda()
        image_batch.requires_grad_()
        noise = image_noise(batch_size, image_size)
        thetas = torch.randint(-30,30, size=(batch_size,)).type(torch.float32)
        w_styles = self.Encoder(image_batch, thetas)
        
        generated_images = self.GAN.G(w_styles, noise)
        fake_output, fake_q_loss = self.GAN.D(generated_images.clone().detach())
        
        #identity transform
        image_batch = next(self.loader).cuda()
        image_batch.requires_grad_()
        noise = image_noise(batch_size, image_size)
        thetas = torch.zeros(10)
        w_styles = self.Encoder(image_batch, thetas)
        
        generated_images = self.GAN.G(w_styles, noise)
        #fake_output, fake_q_loss = self.GAN.D(generated_images.clone().detach())
        id_loss = self.id_loss(generated_images, image_batch)
        
        ######real_images#######
        image_batch = next(self.loader).cuda()
        image_batch.requires_grad_()
        real_output, real_q_loss = self.GAN.D(image_batch)

        disc_loss = (F.relu(1 + real_output) + F.relu(1 - fake_output)).mean()
        

        quantize_loss = (fake_q_loss + real_q_loss).mean()
        self.q_loss = float(quantize_loss.detach().item())

        disc_loss = disc_loss + quantize_loss + id_loss

        if apply_gradient_penalty:
            gp = gradient_penalty(image_batch, real_output)
            self.last_gp_loss = gp.clone().detach().item()
            disc_loss = disc_loss + gp

        
        disc_loss.register_hook(raise_if_nan)
        
        total_disc_loss = disc_loss.detach().item() 

        self.d_loss = float(total_disc_loss)
        
        disc_loss.backwards()
        self.GAN.D_opt.step()
        self.Encoder_opt.step()
        
        generated_images = self.GAN.G(w_styles, noise)

        if self.steps % 10 == 0 and self.steps > 20000:
            self.GAN.EMA()

        if self.steps <= 25000 and self.steps % 1000 == 2:
            self.GAN.reset_parameter_averaging()

        # save from NaN errors

        checkpoint_num = floor(self.steps / self.save_every)

        if torch.isnan(total_disc_loss):
            print(f'NaN detected for discriminator. Loading from checkpoint #{checkpoint_num}')
            self.load(checkpoint_num)
            raise NanException

        # periodically save results

        if self.steps % self.save_every == 0:
            self.save(checkpoint_num)

        if self.steps % 1000 == 0 or (self.steps % 100 == 0 and self.steps < 2500):
            self.evaluate(floor(self.steps / 1000))

        self.steps += 1
        self.av = None

    @torch.no_grad()
    def evaluate(self, num = 0, num_image_tiles = 8, trunc = 1.0):
        self.GAN.eval()
        ext = 'jpg' if not self.transparent else 'png'
        num_rows = num_image_tiles

        def generate_images(stylizer, generator, latents, noise):
            w = latent_to_w(stylizer, latents)
            w_styles = styles_def_to_tensor(w)
            generated_images = evaluate_in_chunks(self.batch_size, generator, w_styles, noise)
            generated_images.clamp_(0., 1.)
            return generated_images
    
        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise

        latents = noise_list(num_rows ** 2, num_layers, latent_dim)
        n = image_noise(num_rows ** 2, image_size)

        # regular

        generated_images = self.generate_truncated(self.GAN.S, 
                                                   self.GAN.G, 
                                                   latents, 
                                                   n, 
                                                   trunc_psi = self.trunc_psi)
        torchvision.utils.save_image(generated_images, 
                                     str(self.results_dir / self.name / f'{str(num)}.{ext}'), nrow=num_rows)
        
        # moving averages

        generated_images = self.generate_truncated(self.GAN.SE, 
                                                   self.GAN.GE, 
                                                   latents, 
                                                   n, 
                                                   trunc_psi = self.trunc_psi)
        torchvision.utils.save_image(generated_images, 
                                     str(self.results_dir / self.name / f'{str(num)}-ema.{ext}'), 
                                     nrow=num_rows)

        # mixing regularities

        def tile(a, dim, n_tile):
            init_dim = a.size(dim)
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*(repeat_idx))
            order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
            return torch.index_select(a, dim, order_index)

        nn = noise(num_rows, latent_dim)
        tmp1 = tile(nn, 0, num_rows)
        tmp2 = nn.repeat(num_rows, 1)

        tt = int(num_layers / 2)
        mixed_latents = [(tmp1, tt), (tmp2, num_layers - tt)]

        generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, mixed_latents, n, trunc_psi = self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-mr.{ext}'), nrow=num_rows)

    @torch.no_grad()
    def generate_truncated(self, S, G, style, noi, trunc_psi = 0.75, num_image_tiles = 8):
        latent_dim = G.latent_dim

        if self.av is None:
            z = noise(2000, latent_dim)
            samples = evaluate_in_chunks(self.batch_size, S, z).cpu().numpy()
            self.av = np.mean(samples, axis = 0)
            self.av = np.expand_dims(self.av, axis = 0)
            
        w_space = []
        for tensor, num_layers in style:
            tmp = S(tensor)
            av_torch = torch.from_numpy(self.av).cuda()
            tmp = trunc_psi * (tmp - av_torch) + av_torch
            w_space.append((tmp, num_layers))

        w_styles = styles_def_to_tensor(w_space)
        generated_images = evaluate_in_chunks(self.batch_size, G, w_styles, noi)
        return generated_images.clamp_(0., 1.)

    def print_log(self):
        print(f'G: {self.g_loss:.2f} | D: {self.d_loss:.2f} | GP: {self.last_gp_loss:.2f} | PL: {self.pl_loss:.2f} | CR: {self.last_cr_loss:.2f} | Q: {self.q_loss:.2f}')

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(f'./models/{self.name}', True)
        rmtree(f'./results/{self.name}', True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    def save(self, num):
        torch.save(self.GAN.state_dict(), self.model_name(num))
        self.write_config()

    def load(self, num = -1):
        self.load_config()

        name = num
        if num == -1:
            file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
            saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')
        self.steps = name * self.save_every
        self.GAN.load_state_dict(torch.load(self.model_name(name)))

