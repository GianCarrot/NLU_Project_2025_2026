import math
import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy

from tqdm.notebook import tqdm
from model import GPT2

def collate_fn(batch, tokenizer, device):
    tokenized = tokenizer(batch, padding=True, return_tensors="pt")

    input_ids = tokenized.input_ids[:, :-1].detach().clone().to(device)
    labels = tokenized.input_ids[:, 1:].detach().clone().to(device)
    n_tokens = torch.sum(input_ids != tokenizer.pad_token_id)

    return input_ids, labels, n_tokens


def train_loop(data, optimizer, criterion, model, scaler, device, accum_steps=1):
    model.train()
    loss_array = []
    number_of_tokens = []
    
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(data, desc="Training", unit="batch", leave=False)

    for i, (input_ids, labels, n_tokens) in enumerate(pbar):
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Mixed Precision Forward Pass
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16):
            output = model(input_ids)
            loss = criterion(output.permute(0, 2, 1), labels)
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        loss_array.append(loss.item() * accum_steps * n_tokens)
        number_of_tokens.append(n_tokens)

        if (i + 1) % accum_steps == 0 or (i + 1) == len(data):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if i % 50 == 0:
            avg_loss = sum(loss_array) / sum(number_of_tokens)
            pbar.set_postfix(train_loss=f"{float(avg_loss):.4f}")

    return (sum(loss_array) / sum(number_of_tokens)).item()


def eval_loop(data, eval_criterion, model, device):
    model.eval()
    loss_array = []
    number_of_tokens = []

    pbar = tqdm(data, desc="Evaluating", unit="batch", leave=False)

    with torch.no_grad():
        for input_ids, labels, n_tokens in pbar:
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16):
                output = model(input_ids)
                loss = eval_criterion(output.permute(0, 2, 1), labels)

            loss_array.append(loss.item() * n_tokens)
            number_of_tokens.append(n_tokens)
            
    loss_to_return = (sum(loss_array) / sum(number_of_tokens)).item()
    ppl = math.exp(loss_to_return)
    return ppl, loss_to_return


def run_pipeline(lr, config, train_loader, dev_loader, test_loader, vocab_len, pad_token_id, device, number_step):
    # Assicuriamoci che device sia un oggetto torch.device
    if isinstance(device, str):
        device = torch.device(device)

    save_path = f"bin/model_{number_step}_name_lr_{lr}.pt"

    model = GPT2(
        vocab_len,
        pos_emb_size=1024,
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        num_layers=config["num_layers"],
        ff_dim=config["ff_dim"],
        dropout=config.get("dropout", 0.0),
        weight_tying=config.get("weight_tying", False)
    ).to(device)

    # GradScaler configurato con il tipo del device
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    criterion_eval = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    if os.path.exists(save_path):
        print("\n" + "="*80)
        print(f"Checkpoint found on {save_path}. Loading best weights...")
        print("="*80)
        model.load_state_dict(torch.load(save_path, map_location=device))
        final_ppl, _ = eval_loop(test_loader, criterion_eval, model, device)
        print(f"PPL loaded is {final_ppl:.2f}")
    else: 
        print("\n" + "="*80)
        print(f"Pipeline Step {number_step} | LR: {lr}")
        print("="*80)

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        criterion_train = nn.CrossEntropyLoss(ignore_index=pad_token_id)

        n_epochs = 15
        patience = 3
        best_ppl = math.inf
        best_state_dict = None

        epochs_pbar = tqdm(range(1, n_epochs + 1), desc="Epochs", unit="epoch")

        for epoch in epochs_pbar:
            loss_train = train_loop(train_loader, optimizer, criterion_train, model, scaler, device)
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model, device)
            
            if ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 3
            else:
                patience -= 1
                
            epochs_pbar.set_postfix({
                "Dev PPL": f"{ppl_dev:.2f}",
                "Best PPL": f"{best_ppl:.2f}",
                "Patience": f"{patience}/3"
            })

            if patience <= 0:
                print(f"\nEarly stopping raggiunto all'epoca {epoch}.")
                break 

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        final_ppl, _ = eval_loop(test_loader, criterion_eval, model, device)    
        print(f"\nFinal Test PPL = {final_ppl:.2f}")

        os.makedirs("bin", exist_ok=True)
        torch.save(best_state_dict if best_state_dict else model.state_dict(), save_path)
        print(f"Best Model successfully saved on {save_path}\n")

    return final_ppl