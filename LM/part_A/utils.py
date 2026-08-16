import os
import urllib.request
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from functools import partial
from transformers import AutoTokenizer

from functions import collate_fn

DATASET_URLS = {
    "ptb.train.txt": "https://raw.githubusercontent.com/massimo-rizzoli/NLU-2026-Labs/main/labs/dataset/PennTreeBank/ptb.train.txt",
    "ptb.valid.txt": "https://raw.githubusercontent.com/massimo-rizzoli/NLU-2026-Labs/main/labs/dataset/PennTreeBank/ptb.valid.txt",
    "ptb.test.txt": "https://raw.githubusercontent.com/massimo-rizzoli/NLU-2026-Labs/main/labs/dataset/PennTreeBank/ptb.test.txt"
}

DATASET_DIR = "../dataset/PennTreeBank/"
TRAINING_SET_DIR = os.path.join(DATASET_DIR, "ptb.train.txt")
VALIDATION_SET_DIR = os.path.join(DATASET_DIR, "ptb.valid.txt")
TEST_SET_DIR = os.path.join(DATASET_DIR, "ptb.test.txt")

EOS_TOKEN = "<eos>"

class PennTreeBank(data.Dataset):
    def __init__(self, corpus):
        self.sents = [sent for sent in corpus]

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        return self.sents[idx]

def read_file(path, eos_token=EOS_TOKEN):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output

def download_dataset_if_missing(dataset_dir=DATASET_DIR):
    os.makedirs(dataset_dir, exist_ok=True)
    for filename, url in DATASET_URLS.items():
        filepath = os.path.join(dataset_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_data(tokenizer, batch_size=32, device="cpu"):
    try:       
        train_raw = read_file(TRAINING_SET_DIR)
        dev_raw = read_file(VALIDATION_SET_DIR)
        test_raw = read_file(TEST_SET_DIR)
        print("Successfully dataset loaded...")
    except FileNotFoundError:
        download_dataset_if_missing()
        train_raw = read_file(TRAINING_SET_DIR)
        dev_raw = read_file(VALIDATION_SET_DIR)
        test_raw = read_file(TEST_SET_DIR)
        print("Successfully dataset loaded...")
        
    train_dataset = PennTreeBank(train_raw)
    dev_dataset = PennTreeBank(dev_raw)
    test_dataset = PennTreeBank(test_raw)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        collate_fn=partial(collate_fn, tokenizer=tokenizer, device=device), 
        shuffle=True
    )

    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=batch_size, 
        collate_fn=partial(collate_fn, tokenizer=tokenizer, device=device)
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        collate_fn=partial(collate_fn, tokenizer=tokenizer, device=device)
    )
    return train_loader, dev_loader, test_loader