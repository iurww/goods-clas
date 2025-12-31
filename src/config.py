import argparse
import sys
from datetime import datetime
import json
import os

import torch

from typing import Any, Dict
from dataclasses import dataclass, asdict


@dataclass
class _Config:
    # ===== 路径配置 =====
    train_csv: str = 'data/compress_train.csv'
    test_csv: str = 'data/compress_test.csv'
    test_probs: str = None
    train_images_dir: str = 'data/train_images/train_images'
    test_images_dir: str = 'data/test_images/test_images'
    results_dir: str = 'results/'
    checkpoint_dir: str = ''

    # ===== 模型配置 =====
    model_name: str = './models/siglip-so400m-patch14-384'
    num_classes: int = 21
    hidden_dim: int = 1536
    dropout: float = 0.2
    max_text_length: int = 64
    image_size: int = 384

    # ===== 训练配置 =====
    batch_size: int = 8
    eval_batch_size: int = 16
    num_epochs: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.05
    warmup_ratio: float = 0.1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    n_folds: int = 3
    skip_folds: int = 0
    use_class_weight: bool = False
    use_fp16: bool = True
    
    log_interval: int = 2
    
    _timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    @property
    def timestamp(self):
        return self._timestamp
    @property
    def cur_run_dir(self):
        return os.path.join(self.results_dir, self._timestamp)
    

def _merge_args_to_config(args: argparse.Namespace, cfg: _Config) -> None:
    cfg_dict: Dict[str, Any] = _Config.__dict__
    for k, v in vars(args).items():
        if k in cfg_dict and not k.startswith('_'):
            if isinstance(getattr(type(cfg), k, None), property):
                continue
            if isinstance(cfg_dict[k], bool) and not isinstance(v, bool):
                v = v.lower() in ('true', '1', 'yes', 'on')
            setattr(cfg, k, v)
   
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SigLIP 训练推理配置")
    cfg_dict = _Config.__dict__
    for k, v in cfg_dict.items():
        if k.startswith('_'):
            continue
        if isinstance(v, bool):
            parser.add_argument(f'--{k}', action='store_true' if not v else 'store_false',
                                help=f"覆盖默认 {k}={v}")
        else:
            parser.add_argument(f'--{k}', type=type(v), default=v,
                                help=f"覆盖默认 {k}={v}")
    return parser

Config = _Config()

if any(arg.startswith('--') for arg in sys.argv):
    _parser = _build_parser()
    _known, _unknown = _parser.parse_known_args()
    _merge_args_to_config(_known, Config)
    print("Using command-line arguments to override config:")


os.makedirs(Config.results_dir, exist_ok=True)
os.makedirs(Config.cur_run_dir, exist_ok=True)
os.makedirs(os.path.join(Config.cur_run_dir, 'weights'), exist_ok=True)
with open(os.path.join(Config.cur_run_dir, 'config.json'), 'w') as f:   
    json.dump({k: v for k, v in asdict(Config).items() if not k.startswith('_')}, f, indent=4)


def merge_config(args: argparse.Namespace) -> None:
    _merge_args_to_config(args, Config)