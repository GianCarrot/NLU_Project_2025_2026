import torch

from torch.utils.data import DataLoader
from functools import partial
from transformers import AutoTokenizer

from utils import (
    load_data
)

from functions import (
    run_pipeline
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LEARNING_RATES = [1e-2, 1e-3, 5e-4, 1e-4]
NEXT_STEPS = [
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
      "weight_tying": True
    }
  },
  {
    "step_id": "3",
    "name": "Step 3 (Model Scaling)",
    "config": {
      "d_model": 256,        
      "n_heads": 8,
      "num_layers": 4,
      "ff_dim": 1024,
      "dropout": 0.1,
      "weight_tying": True
    }
  }
]

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, dev_loader, test_loader = load_data(tokenizer, batch_size=32)
    
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

    step0_results = {}

    for lr in LEARNING_RATES:
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


    # =========================================================================
    # 3. CYCLIC EXECUTION OF THE FOLLOWING STEPS
    # =========================================================================
    for step in NEXT_STEPS:
        print("\n" + "="*100)
        print(f"Starting {step['name']} with LR = {best_lr}")
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
    # 4. SUMMARY TABLE OF ALL STEPS
    # =========================================================================
    print("\n" + "="*100)
    print("FINAL SUMMARY OF ALL STEPS")
    print("="*50)
    for step_name, ppl in all_steps_results.items():
        print(f"{step_name:<25} | PPL Test: {ppl:.2f}")
    print("="*50)