import os
from pathlib import Path

import torch
from torchvision import transforms

#################################### Experiment setup ####################################

# Hyperparams

N_ITERATIONS = 10000
SAMPLE_EVERY = 1000
SAVE_EVERY = 1000
BATCH_SIZE = 2

ENC_OPT = lambda parameters: torch.optim.Adam(parameters, 1e-3, weight_decay=0)
D_OPT = lambda parameters: torch.optim.Adam(parameters, 1e-3, weight_decay=0)

###log_shape must be equal to log2(img_shape) - 1 !!!#####
ENCODER_PARAMETERS = {
    "nf" : 16,
    "log_shape": 6
}

STYLEGAN_PARAMETERS = {
    "image_size": 128
}

TRANSFORM = [
    transforms.Resize((128, 128)),
    transforms.ToTensor()
]

TRAIN_TRANSFORM = [
    transforms.RandomHorizontalFlip(),        
    transforms.CenterCrop(STYLEGAN_PARAMETERS['image_size']),
            
]

# Experiment metadata

EXPERIMENT_TAG = 'holoencoder_stylegan_celeba_iterations' # Tag used for associated files
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

# Results

RESULTS_ROOT = ROOT_DIR / 'results' / EXPERIMENT_TAG
SAMPLES_ROOT = RESULTS_ROOT / 'samples'
SAMPLES_ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_ROOT = RESULTS_ROOT / 'checkpoints'
CHECKPOINTS_ROOT.mkdir(parents=True, exist_ok=True)
LOGDIR = RESULTS_ROOT / 'logdir'
