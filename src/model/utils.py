import torch

from config import cfg


def save(obj, checkpoint_name):
    checkpoint_path = get_checkpoint_path(checkpoint_name)
    torch.save(obj.state_dict(), checkpoint_path)


def prepare(obj, checkpoint_name):
    if hasattr(obj, 'to'):
        obj.to(cfg.DEVICE)

    if hasattr(obj, 'state_dict'):
        checkpoint_path = get_checkpoint_path(checkpoint_name)
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            obj.load_state_dict(checkpoint)


def get_checkpoint_path(checkpoint_name):
    return cfg.CHECKPOINTS_ROOT / (checkpoint_name + '.pth')
