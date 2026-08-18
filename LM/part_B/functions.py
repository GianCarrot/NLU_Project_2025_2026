import os
import glob
import math
import copy
import torch
import torch.optim as optim
from transformers import GPT2LMHeadModel
from model import LoRACLinear
from tqdm.notebook import tqdm

BIN_DIR = "bin"
MODEL_SAVE_PATH = os.path.join(BIN_DIR, "best_lora_model.pt")

def train_loop(data, optimizer, model, device):
    model.train()
    loss_array = []
    number_of_tokens = []

    pbar = tqdm(data, desc="Training", unit="batch", total=len(data))

    for i, (input_ids, labels, n_tokens) in enumerate(pbar):
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(input_ids, labels=labels)
        
        loss = output.loss
        loss_array.append(loss.item() * n_tokens)
        number_of_tokens.append(n_tokens)

        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            current_avg_loss = sum(loss_array) / sum(number_of_tokens)
            pbar.set_postfix(loss=float(current_avg_loss))

    total_loss = sum(loss_array) / sum(number_of_tokens)
    return total_loss


def eval_loop(data, model, device):
    model.eval()
    loss_array = []
    number_of_tokens = []

    with torch.no_grad():
        for input_ids, labels, n_tokens in tqdm(data, desc="Evaluating", unit="batch", total=len(data)):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            output = model(input_ids, labels=labels)
            
            loss_array.append(output.loss.item() * n_tokens)
            number_of_tokens.append(n_tokens)

    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    ppl = math.exp(loss_to_return)
    return ppl, loss_to_return

def get_gpt2_lora_model(model_name, rank, alpha):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    hidden_dim = model.config.n_embd

    for i in range(len(model.transformer.h)):
        old_c_attn = model.transformer.h[i].attn.c_attn
        model.transformer.h[i].attn.c_attn = LoRACLinear(old_c_attn, hidden_dim, rank=rank, alpha=alpha)

    return model

def run_pipeline(model, train_loader, dev_loader, test_loader, device, lr=3e-4, n_epochs=100, patience=3):
    """
    Loads pretrained weights if a .pt file exists in bin/, 
    otherwise runs full training with early stopping and evaluates on the test set.
    """
    # 1. Check if a .pt checkpoint exists in bin/
    pt_files = glob.glob(os.path.join(BIN_DIR, "*.pt")) if os.path.exists(BIN_DIR) else []

    if pt_files:
        weights_path = pt_files[0]
        print(f"Existing model detected at '{weights_path}'. Skipping training.")
        print("Loading weights for Evaluation on Test Set...")
        
        model.load_state_dict(torch.load(weights_path, map_location=device))
        final_ppl, _ = eval_loop(test_loader, model, device)
        
        print()
        print("="*100)
        print(f"Test Perplexity: {final_ppl:.2f}")
        print("="*100)
        print()
        return final_ppl

    # 2. Run Training if bin/ is empty
    print("Directory 'bin/' is empty. Starting training loop")
    print()
    os.makedirs(BIN_DIR, exist_ok=True)

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )

    best_ppl = math.inf
    best_model_state = None
    current_patience = patience

    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        
        train_loss = train_loop(train_loader, optimizer, model, device)
        ppl_dev, dev_loss = eval_loop(dev_loader, model, device)

        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f} | Dev PPL: {ppl_dev:.2f}")

        if ppl_dev < best_ppl:
            best_ppl = ppl_dev
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, MODEL_SAVE_PATH)
            current_patience = patience
            print(f"Saved new best model to '{MODEL_SAVE_PATH}'. Current PPL = {best_ppl:.2f}\n")
        else:
            current_patience -= 1
            print(f"Patience = {current_patience}/{patience}")

        if current_patience <= 0:
            print("Early stopping triggered.")
            break

    # 3. Final Test with the best saved checkpoint
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    final_ppl, _ = eval_loop(test_loader, model, device)
    print()
    print("="*100)
    print(f"Final Test Perplexity: {final_ppl:.2f}")
    print()
    print("="*100)

    return final_ppl