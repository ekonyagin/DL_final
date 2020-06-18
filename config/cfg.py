import os
import shutil
from pathlib import Path

import torch
from torchvision import transforms
from src.model.encoder import HoloEncoder, HoloEncoderLight
from torch_optimizer import DiffGrad
#################################### Experiment setup ####################################

# Experiment setup

CONTINUE = True
ENCODER_CLASS = HoloEncoder
DATASET = 'ffhq128'
# DATASET = 'celeba'

N_ITERATIONS = 150000
SAMPLE_EVERY = 100
SAVE_EVERY = 1000
BATCH_SIZE = 2
IMG_SIZE = 128

# Hyperparams

APPLY_NOISE = True

LR_ENC = 1e-4
LR_STYLEGAN = 1e-4

ROT0_LOSS_COEF = 0
EMB_LOSS_COEF = 250
EMB_LOSS_START_ITER = 50000

MIN_ANGLE = -30
MAX_ANGLE = 30

###log_shape must be equal to log2(img_shape) - 1 !!!#####
ENCODER_PARAMETERS = {
    "nf": 16,
    "log_shape": 6,
    "lr": LR_ENC * BATCH_SIZE
}

STYLEGAN_PARAMETERS = {
    "image_size": IMG_SIZE,
    "network_capacity": 16,
    "lr": LR_STYLEGAN / 15 * BATCH_SIZE,
}

TRAIN_TRANSFORM = [
    transforms.CenterCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
]

SHARED_TRANSFORM = [
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
]

trackable_params = [
    f'{ENCODER_CLASS.__name__}',
    f'{DATASET}',
    f'LR_E={ENCODER_PARAMETERS["lr"]:.0E}',
    f'LR_S={STYLEGAN_PARAMETERS["lr"]:.0E}',
    f'R={ROT0_LOSS_COEF}',
    f'E={EMB_LOSS_COEF}',
    f'NF_E={ENCODER_PARAMETERS["nf"]}',
    f'E_START={EMB_LOSS_START_ITER}',
]

EXPERIMENT_TAG = "__".join(trackable_params)  # Tag used for associated files
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Reproducibility

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

################################### Environment & stuff ###################################

ROOT_DIR = Path(os.environ['ROOT_DIR'])
# Input data

DATA_ROOT = ROOT_DIR / 'data'
TRAIN_ROOT = DATA_ROOT / ('train_' + DATASET)
VAL_ROOT = DATA_ROOT / 'val'
TEST_ROOT = DATA_ROOT / 'test'

# Checkpoints
STYLEGAN_CHECKPOINT_PATH = None # ROOT_DIR / 'model_149.pt'
EMB_CHECKPOINT_PATH = DATA_ROOT / 'emb_ckpt.pth'

# Results

RESULTS_ROOT = ROOT_DIR / 'results' / EXPERIMENT_TAG
if not CONTINUE and RESULTS_ROOT.exists():
    shutil.rmtree(RESULTS_ROOT)
SAMPLES_ROOT = RESULTS_ROOT / 'samples'
SAMPLES_ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_ROOT = RESULTS_ROOT / 'checkpoints'
CHECKPOINTS_ROOT.mkdir(parents=True, exist_ok=True)
LOGDIR = RESULTS_ROOT / 'logdir'


