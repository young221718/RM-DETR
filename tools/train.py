"""by lyuwenyu
"""

import PIL.Image
import os
import sys
import random
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import argparse

import src.misc.dist as dist
from src.core import YAMLConfig
from src.solver import TASKS

PIL.Image.MAX_IMAGE_PIXELS = 933120000


def main(
    args,
) -> None:
    """main"""
    dist.init_distributed()

    assert not all(
        [args.tuning, args.resume]
    ), "Only support from_scrach or resume or tuning at one time"

    cfg = YAMLConfig(
        args.config, resume=args.resume, use_amp=args.amp, tuning=args.tuning
    )

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)

    if args.flops:
        solver.get_flops()
    elif args.test_only:
        if args.compare:
            solver.get_pred()
        else:
            solver.val()
    else:
        solver.fit()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
    )
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
    )
    parser.add_argument(
        "--tuning",
        "-t",
        type=str,
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--flops",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    # 시드 고정
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(42)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    main(args)
