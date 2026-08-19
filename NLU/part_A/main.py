import os
import torch
import torch.nn as nn

# Import everything from functions.py file
from utils import get_dataloaders
from functions import run_pipeline

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
RUNS = 3
EXPERIMENTS = [
  {"id": "0_baseline", "name": "0. Baseline", "lr": 1e-3, "d_model": 20, "n_heads": 1, "num_layers": 1, "ff_dim": 20, "out_dropout": 0.0},
  {"id": "1_1_dmodel", "name": "1.1 Inc d_model", "lr": 1e-3, "d_model": 64, "n_heads": 1, "num_layers": 1, "ff_dim": 20, "out_dropout": 0.0},
  {"id": "1_2_nheads", "name": "1.2 Inc n_heads", "lr": 1e-3, "d_model": 64, "n_heads": 4, "num_layers": 1, "ff_dim": 20, "out_dropout": 0.0},
  {"id": "1_3_nlayers", "name": "1.3 Inc num_layers", "lr": 5e-4, "d_model": 64, "n_heads": 4, "num_layers": 3, "ff_dim": 20, "out_dropout": 0.0},
  {"id": "1_4_ffdim", "name": "1.4 Inc ff_dim", "lr": 5e-4, "d_model": 64, "n_heads": 4, "num_layers": 3, "ff_dim": 256, "out_dropout": 0.0},
  {"id": "2_dropout", "name": "2. Add Output Dropout", "lr": 5e-4, "d_model": 64, "n_heads": 4, "num_layers": 3, "ff_dim": 256, "out_dropout": 0.2},
]

if __name__ == "__main__":
  dataset_dir = os.path.join('dataset', 'ATIS')
  train_loader, dev_loader, test_loader, lang = get_dataloaders(dataset_dir, DEVICE)

  for exp in EXPERIMENTS:
        run_pipeline(exp, train_loader, dev_loader, test_loader, lang, DEVICE, RUNS)