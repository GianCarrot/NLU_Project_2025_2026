import torch
import torch.utils.data as data
import os
import urllib.request

DATASET_URLS = {
    "ptb.train.txt": "https://raw.githubusercontent.com/massimo-rizzoli/NLU-2026-Labs/main/labs/dataset/PennTreeBank/ptb.train.txt",
    "ptb.valid.txt": "https://raw.githubusercontent.com/massimo-rizzoli/NLU-2026-Labs/main/labs/dataset/PennTreeBank/ptb.valid.txt",
    "ptb.test.txt": "https://raw.githubusercontent.com/massimo-rizzoli/NLU-2026-Labs/main/labs/dataset/PennTreeBank/ptb.test.txt"
}

DATASET_DIR = "dataset/PennTreeBank/"
TRAINING_SET_DIR =  os.path.join(DATASET_DIR, "ptb.train.txt")
VALIDATION_SET_DIR = os.path.join(DATASET_DIR, "ptb.valid.txt")
TEST_SET_DIR = os.path.join(DATASET_DIR, "ptb.test.txt")

EOS_TOKEN = "<eos>"

class PennTreeBank (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
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