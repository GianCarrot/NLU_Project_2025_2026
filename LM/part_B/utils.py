import os
import glob
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

DATASET_DIR = "../dataset/PennTreeBank"
TRAIN_PATH = os.path.join(DATASET_DIR, "ptb.train.txt")
VALID_PATH = os.path.join(DATASET_DIR, "ptb.valid.txt")
TEST_PATH  = os.path.join(DATASET_DIR, "ptb.test.txt")

DATASET_URLS = {
    "ptb.train.txt": "https://raw.githubusercontent.com/massimo-rizzoli/NLU-2026-Labs/main/labs/dataset/PennTreeBank/ptb.train.txt",
    "ptb.valid.txt": "https://raw.githubusercontent.com/massimo-rizzoli/NLU-2026-Labs/main/labs/dataset/PennTreeBank/ptb.valid.txt",
    "ptb.test.txt": "https://raw.githubusercontent.com/massimo-rizzoli/NLU-2026-Labs/main/labs/dataset/PennTreeBank/ptb.test.txt"
}

class PennTreeBank(Dataset):
    def __init__(self, corpus):
        self.sents = list(corpus)

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        return self.sents[idx]

def download_dataset():
    os.makedirs(DATASET_DIR, exist_ok=True)
    for name, url in DATASET_URLS.items():
        path = os.path.join(DATASET_DIR, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path)

def read_file(path):
    output = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line_str = line.strip()
            if line_str:
                output.append(line_str)
    return output

def collate_fn(batch, tokenizer):
    encoded = tokenizer(
        batch, 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    
    n_tokens = attention_mask.sum().item()
    return input_ids, labels, n_tokens

def load_data(tokenizer, batch_size=8):
    download_dataset()
    train_raw = read_file(TRAIN_PATH)
    dev_raw   = read_file(VALID_PATH)
    test_raw  = read_file(TEST_PATH)

    train_loader = DataLoader(
        PennTreeBank(train_raw), batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    dev_loader = DataLoader(
        PennTreeBank(dev_raw), batch_size=batch_size * 2, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    test_loader = DataLoader(
        PennTreeBank(test_raw), batch_size=batch_size * 2, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )

    return train_loader, dev_loader, test_loader

def param_stats(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Frozen params:    {total - trainable:,}\n")

def freeze_non_lora_params(model):
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

def check_bin_has_weights(bin_dir="bin"):
    if not os.path.exists(bin_dir):
        return False
    pt_files = glob.glob(os.path.join(bin_dir, "*.pt"))
    return len(pt_files) > 0