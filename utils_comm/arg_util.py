import argparse
import os
from datetime import datetime
from pathlib import Path

DATE_TIME = "%Y_%m_%d %H:%M:%S"


class ArgparseUtil(object):
    """
    参数解析工具类
    """
    def __init__(self):
        """ Basic args """
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--seed", default=2, type=int)

    def task(self):
        """ task args """
        self.parser.add_argument('--gpu_device_id', default=1, type=int, help='the GPU NO.')
        self.parser.add_argument("--input_root_dir", type=str, default='', help="")
        self.parser.add_argument("--out_root_dir", type=str, default='', help="")
        self.parser.add_argument("--task_name", type=str, default='', help="")
        args = self.parser.parse_args()
        return args


def save_args(*multi_args, output_dir='.', with_time_at_filename=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    t0 = datetime.now().strftime(DATE_TIME)
    if with_time_at_filename:
        out_file = output_dir / f"args-{t0}.log"
    else:
        out_file = output_dir / "args.log"

    with open(out_file, "w", encoding='utf-8') as f:
        f.write(f"save time: {t0}\n")
        for args in multi_args:
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")
            f.write("\n")


def log_args(*multi_args, logger=None):
    if logger is None:
        import logging
        logger = logging.getLogger()
        logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(filename)s %(lineno)d: %(message)s',
            datefmt='%y-%m-%d %H:%M')
    for args in multi_args:
        for arg, value in vars(args).items():
            logger.info(f"{arg}: {value}")
