import os
from pathlib import Path

import torch
from torchvision import transforms
from model.encoder import HoloEncoder, HoloEncoderLight 
from torch_optimizer import DiffGrad
#################################### Experiment setup ####################################

# Hyperparams

N_ITERATIONS = 7000
SAMPLE_EVERY = 100
SAVE_EVERY = 1000
BATCH_SIZE = 2

ENC_OPT = lambda parameters: torch.optim.Adam(parameters, 1e-4, weight_decay=0)
D_OPT = lambda parameters: DiffGrad(parameters, 5e-6)

ENCODER_CLASS = HoloEncoder

###log_shape must be equal to log2(img_shape) - 1 !!!#####
ENCODER_PARAMETERS = {
    "nf" : 16,
    "log_shape": 6
}


STYLEGAN_PARAMETERS = {
    "image_size": 128,
    "network_capacity" : 16
}

STYLEGAN_FIXD = False

TRANSFORM = [
    transforms.CenterCrop(STYLEGAN_PARAMETERS['image_size']),
    # transforms.Resize((128, 128)),
    transforms.ToTensor()
]

TRAIN_TRANSFORM = [
    transforms.RandomHorizontalFlip(),        
    transforms.CenterCrop(STYLEGAN_PARAMETERS['image_size']),
            
]
# Losses coefs
ROT0_LOSS_COEF = 10

# Experiment metadata
# nl = new_loss
# introduced iterations

EXPERIMENT_TAG = 'nl_celeba_h16_s16_dg5e6_eopt1e4_frominit' # Tag used for associated files
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Reproducibility

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

################################### Environment & stuff ###################################

ROOT_DIR = Path(os.environ['ROOT_DIR'])
print(ROOT_DIR)
# Input data

DATA_ROOT =  ROOT_DIR / 'data'
TRAIN_ROOT = DATA_ROOT / 'train'
VAL_ROOT = DATA_ROOT / 'val'
TEST_ROOT = DATA_ROOT / 'test'

# Stylegan Checkpoint
# if no checkpoint is present - fill with empty string
STYLEGAN_CHECKPOINT_PATH = '' # ROOT_DIR / 'model_149.pt'

# Results

RESULTS_ROOT = ROOT_DIR / 'results' / EXPERIMENT_TAG
SAMPLES_ROOT = RESULTS_ROOT / 'samples'
SAMPLES_ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_ROOT = RESULTS_ROOT / 'checkpoints'
CHECKPOINTS_ROOT.mkdir(parents=True, exist_ok=True)
LOGDIR = RESULTS_ROOT / 'logdir'


