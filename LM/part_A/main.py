import torch

from utils import (
    load_data,
    get_tokenizer
)

from functions import (
    run_pipeline
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LEARNING_RATES = [1e-2, 1e-3, 5e-4, 1e-4]
BASELINE_CONFIG_STEP0 = {
        "d_model": 128,
        "n_heads": 4,
        "num_layers": 2,
        "ff_dim": 512,
        "dropout": 0.0,
        "weight_tying": False
    }
NEXT_STEPS = [
  {
    "step_id": "1",
    "name": "Step 1 (Model Scaling)",
    "config": {
      "d_model": 256,        
      "n_heads": 8,
      "num_layers": 4,
      "ff_dim": 1024,
      "dropout": 0.0,
      "weight_tying": False
    }
  },
  {
    "step_id": "2",
    "name": "Step 2 (Dropout)",
    "config": {
      "d_model": 256,        
      "n_heads": 8,
      "num_layers": 4,
      "ff_dim": 1024,
      "dropout": 0.1,        
      "weight_tying": False
    }
  },
  {
    "step_id": "3",
    "name": "Step 3 (Weight Tying)",
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
    tokenizer = get_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token
    train_loader, dev_loader, test_loader = load_data(tokenizer, batch_size=32)
    
    # =========================================================================
    # STEP 0: Best Learning Rate Grid Search 
    # =========================================================================
    step0_results = {}

    for lr in LEARNING_RATES:
        test_ppl = run_pipeline(
            lr=lr, 
            config=BASELINE_CONFIG_STEP0,
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
    print("STEP 0 RESULTS (GRID SEARCH LR)")
    print("="*100)
    for lr, ppl in step0_results.items():
        print(f"LR: {lr:<8} | PPL: {ppl:.2f}")

    best_lr = min(step0_results, key=step0_results.get)
    print(f"\nBest Learning Rate Found = {best_lr} with PPL: {step0_results[best_lr]:.2f}")


    # =========================================================================
    # NEXT STEPS EXECUTION
    # =========================================================================
    
    # Final dictionary to collect the PPLs of all steps
    all_steps_results = {
        "Step 0 (Baseline)": step0_results[best_lr]
    }

    # =========================================================================
    # CYCLIC EXECUTION OF THE FOLLOWING STEPS
    # =========================================================================
    for step in NEXT_STEPS:
        print()
        print("="*100)
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
    print("="*100)
    for step_name, ppl in all_steps_results.items():
        print(f"{step_name:<25} | PPL Test: {ppl:.2f}")
    print("="*100)