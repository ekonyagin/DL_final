import torch

from config import cfg


def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool


class BaseModel:
    def freeze_(self):
        set_requires_grad(self, False)
        # self.eval()

    def unfreeze_(self):
        set_requires_grad(self, True)
        self.train()

    def step(self):
        self.opt.step()
        self.opt.zero_grad()


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
