import os

from config import cfg


def save():
    old, new = cfg.RESULTS_ROOT / 'cfg.py', cfg.ROOT_DIR / 'config' / 'cfg.py'
    if old.exists():
        if os.system(f'cmp --silent {old} {new}'):
            raise EnvironmentError('Config file in RESULTS_ROOT already exists and differs from the one in config/')
    os.system(f'cp {new} {old}')


def gel_last_iter():
    try:
        start = int(os.popen(f"tensorboard --inspect --event_file={str(next(cfg.LOGDIR.glob('*event*')))} | " +
                             "grep num_steps | awk '{print $2}'").read()) + 1
        print(f'Last iteration found: {start}')
    except Exception as exc:
        print(exc)
        start = 0

    return start
