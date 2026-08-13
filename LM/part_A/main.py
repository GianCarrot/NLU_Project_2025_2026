# main.py
import torch

from torch.utils.data import DataLoader
from functools import partial
from transformers import AutoTokenizer

from utils import (
    PennTreeBank, 
    read_file, 
    download_dataset_if_missing, 
    TRAINING_SET_DIR, 
    VALIDATION_SET_DIR, 
    TEST_SET_DIR
)

from functions import (
    run_pipeline, 
    collate_fn
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    try:
        print("Loading dataset...")
        train_raw = read_file(TRAINING_SET_DIR)
        dev_raw = read_file(VALIDATION_SET_DIR)
        test_raw = read_file(TEST_SET_DIR)
        print("Successfully dataset loaded...")
    except FileNotFoundError:
        print("Failure on dataset loading...")
        download_dataset_if_missing()

        rain_raw = read_file(TRAINING_SET_DIR)
        dev_raw = read_file(VALIDATION_SET_DIR)
        test_raw = read_file(TEST_SET_DIR)
        print("Successfully dataset loaded...")
    
    train_dataset = PennTreeBank(train_raw)
    dev_dataset = PennTreeBank(dev_raw)
    test_dataset = PennTreeBank(test_raw)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=8, 
        collate_fn=partial(collate_fn, tokenizer=tokenizer, device=DEVICE), 
        shuffle=True
    )
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=16, 
        collate_fn=partial(collate_fn, tokenizer=tokenizer, device=DEVICE)
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=16, 
        collate_fn=partial(collate_fn, tokenizer=tokenizer, device=DEVICE)
    )

    # 2. Baseline configuration
    baseline_config = {
        "d_model": 128,
        "n_heads": 4,
        "num_layers": 2,
        "ff_dim": 512,
        "dropout": 0.0,
        "use_weight_tying": False
    }

    # 3. Execution Grid Search
    learning_rates = [1e-2, 1e-3, 5e-4, 1e-4]
    results = {}

    for lr in learning_rates:
        test_ppl = run_pipeline(
            lr=lr, 
            config=baseline_config,
            train_loader=train_loader,
            dev_loader=dev_loader,
            test_loader=test_loader,
            vocab_len=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            device=DEVICE
        )
        results[lr] = test_ppl

    # 4. Results output
    print()
    print("="*100)
    for lr, ppl in results.items():
        print(f"LR: {lr:<8} | PPL: {ppl:.2f}")

    best_lr = min(results, key=results.get)
    print(f"\nBest Learning Found = {best_lr} with PPL: {results[best_lr]:.2f}")