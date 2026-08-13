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

        train_raw = read_file(TRAINING_SET_DIR)
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

    # =========================================================================
    # 1. STEP 0: Grid Search per il miglior Learning Rate
    # =========================================================================
    baseline_config_step0 = {
        "d_model": 128,
        "n_heads": 4,
        "num_layers": 2,
        "ff_dim": 512,
        "dropout": 0.0,
        "weight_tying": False
    }

    learning_rates = [1e-2, 1e-3, 5e-4, 1e-4]
    step0_results = {}

    for lr in learning_rates:
        test_ppl = run_pipeline(
            lr=lr, 
            config=baseline_config_step0,
            train_loader=train_loader,
            dev_loader=dev_loader,
            test_loader=test_loader,
            vocab_len=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            device=DEVICE,
            number_step="0"
        )
        step0_results[lr] = test_ppl

    print("\n" + "="*100)
    print(" RISULTATI STEP 0 (GRID SEARCH LR)")
    print("="*100)
    for lr, ppl in step0_results.items():
        print(f"LR: {lr:<8} | PPL: {ppl:.2f}")

    best_lr = min(step0_results, key=step0_results.get)
    print(f"\nBest Learning Rate Found = {best_lr} con PPL: {step0_results[best_lr]:.2f}")


    # =========================================================================
    # 2. DEFINIZIONE DEGLI STEP SUCCESSIVI (Usando il best_lr)
    # =========================================================================
    
    # Dizionario finale per raccogliere le PPL di tutti gli step
    all_steps_results = {
        "Step 0 (Baseline)": step0_results[best_lr]
    }

    # Definiamo la lista dei prossimi esperimenti
    next_steps = [
        {
            "step_id": "1",
            "name": "Step 1 (Dropout)",
            "config": {
                "d_model": 128,
                "n_heads": 4,
                "num_layers": 2,
                "ff_dim": 512,
                "dropout": 0.1,        # Aggiunta del dropout
                "weight_tying": False
            }
        },
        {
            "step_id": "2",
            "name": "Step 2 (Weight Tying)",
            "config": {
                "d_model": 128,
                "n_heads": 4,
                "num_layers": 2,
                "ff_dim": 512,
                "dropout": 0.1,
                "weight_tying": True   # Attivazione Weight Tying
            }
        },
        {
            "step_id": "3",
            "name": "Step 3 (Model Scaling)",
            "config": {
                "d_model": 256,        # Raddoppio della dimensione del modello
                "n_heads": 8,
                "num_layers": 4,
                "ff_dim": 1024,
                "dropout": 0.1,
                "weight_tying": True
            }
        }
    ]

    # =========================================================================
    # 3. ESECUZIONE CICLICA DEGLI STEP SUCCESSIVI
    # =========================================================================
    for step in next_steps:
        print("\n" + "="*100)
        print(f" Avvio {step['name']} con LR = {best_lr}")
        print("="*100)

        ppl = run_pipeline(
            lr=best_lr,
            config=step["config"],
            train_loader=train_loader,
            dev_loader=dev_loader,
            test_loader=test_loader,
            vocab_len=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            device=DEVICE,
            number_step=step["step_id"]
        )
        
        all_steps_results[step["name"]] = ppl

    # =========================================================================
    # 4. TABELLA RIASSUNTIVA DI TUTTI GLI STEP
    # =========================================================================
    print("\n" + "="*50)
    print(" RIEPILOGO FINALE DI TUTTI GLI STEP")
    print("="*50)
    for step_name, ppl in all_steps_results.items():
        print(f"{step_name:<25} | PPL Test: {ppl:.2f}")
    print("="*50)