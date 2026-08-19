import torch
from transformers import AutoTokenizer

from utils import load_data, param_stats, freeze_non_lora_params
from functions import run_pipeline, get_gpt2_lora_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "openai-community/gpt2"
RANK = 8
ALPHA = 16
BATCH_SIZE = 16
ACCUMULATION_STEPS = 4  # Simula una batch size effettiva di 64 (16 * 4) per accelerare la convergenza

def main():
    print(f"Using device: {DEVICE}")

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. DataLoaders
    train_loader, dev_loader, test_loader = load_data(tokenizer, BATCH_SIZE)

    # 3. Model
    model = get_gpt2_lora_model(MODEL_NAME, RANK, ALPHA)
    freeze_non_lora_params(model)
    model.to(DEVICE)

    param_stats(model)

    # 4. Run Training
    final_ppl = run_pipeline(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        device=DEVICE,
        lr=5e-4,
        n_epochs=100,
        patience=3,
        accumulation_steps=ACCUMULATION_STEPS  # Passiamo il parametro a run_pipeline
    )

    if final_ppl > 250: 
        print("Error: Final Perplexity must be below 250!")

if __name__ == "__main__":
    main()