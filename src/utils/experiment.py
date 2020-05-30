import os

from config import cfg


def save():
    old, new = cfg.RESULTS_ROOT / 'cfg.py', cfg.ROOT_DIR / 'config' / 'cfg.py'
    if old.exists():
        if os.system(f'cmp --silent {old} {new}'):
            raise EnvironmentError('Config file in RESULTS_ROOT already exists and differs from the one in config/')
    os.system(f'cp {new} {old}')
