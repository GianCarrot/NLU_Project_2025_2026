import torch
from transformers import AutoTokenizer

from model import get_gpt2_lora_model
from utils import load_data, param_stats, freeze_non_lora_params
from functions import run_pipeline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"Using device: {DEVICE}")

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 2. DataLoaders
    train_loader, dev_loader, test_loader = load_data(tokenizer, batch_size=8)

    # 3. Model
    model = get_gpt2_lora_model("openai-community/gpt2", rank=8, alpha=16)
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
        lr=3e-4,
        n_epochs=100,
        patience=3
    )

    assert final_ppl < 250, "Error: Final Perplexity must be below 250!"

if __name__ == "__main__":
    main()