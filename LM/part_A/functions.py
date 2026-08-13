import math
import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy

from tqdm import tqdm
from model import GPT2

def collate_fn(batch, tokenizer, device):
    tokenized = tokenizer(batch, padding=True, return_tensors="pt")

    input_ids = tokenized.input_ids[:, :-1].detach().clone().to(device)
    # labels are the input shifted left -> predict the next token
    labels = tokenized.input_ids[:, 1:].detach().clone().to(device)

    # count non-pad tokens
    n_tokens = torch.sum(input_ids != tokenizer.pad_token_id)

    return input_ids, labels, n_tokens

def train_loop(data, optimizer, criterion, model):
    model.train()
    loss_array = []
    number_of_tokens = []
    
    pbar = tqdm(data, desc="Training:", unit="batch", total=len(data))

    for i, (input_ids, labels, n_tokens) in enumerate(pbar):
        optimizer.zero_grad() # Zeroing the gradient
        output = model(input_ids)
        # need to reshape as (B, vocab, L)
        loss = criterion(output.permute(0,2,1), labels)
        loss_array.append(loss.item() * n_tokens)
        number_of_tokens.append(n_tokens)
        loss.backward() # Compute the gradient, deleting the computational graph
        optimizer.step() # Update the weights

        if i % 100 == 0:
            avg_loss = sum(loss_array) / sum(number_of_tokens)
            pbar.set_postfix(loss=float(avg_loss))

    return (sum(loss_array) / sum(number_of_tokens)).item()

def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_array = []
    number_of_tokens = []

    with torch.no_grad(): # Used to avoid the creation of computational graph
        for input_ids, labels, n_tokens in tqdm(data, 
        desc="Evaluating: ", 
        unit="batch", total=len(data)):
            output = model(input_ids)
            # need to reshape as (B, vocab, L)
            loss = eval_criterion(output.permute(0,2,1), labels)
            loss_array.append(loss.item() * n_tokens)
            number_of_tokens.append(n_tokens)
            
    loss_to_return = (sum(loss_array) / sum(number_of_tokens)).item()
    ppl = math.exp(loss_to_return)
    return ppl, loss_to_return

def init_weights(mat):
    for m in mat.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.data.fill_(0.01)

def run_pipeline(lr, 
                 config, 
                 train_loader, 
                 dev_loader, 
                 test_loader, 
                 vocab_len, 
                 pad_token_id, 
                 device,
                 number_step):
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

    criterion_eval = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    if os.path.exists(save_path):
        print()
        print("="*100)
        print(f"Checkpoint found on {save_path}. Loading best weights...")
        print("="*100)
        model.load_state_dict(torch.load(save_path, map_location=device))
        final_ppl, _ = eval_loop(test_loader, criterion_eval, model)
        print(f"PPL loaded is {final_ppl}")
    else: 
        print()
        print("="*100)
        print(f"Pipeline Execution [Step {number_step}] with Learning Rate: {lr}")
        print("="*100)

        model.apply(init_weights)

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        criterion_train = nn.CrossEntropyLoss(ignore_index=pad_token_id)

        n_epochs = 100
        patience = 3
        losses_train = []
        losses_dev = []
        sampled_epochs = []

        best_ppl = math.inf
        best_model = None

        pbar = tqdm(range(n_epochs))

        for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_train, model)
                
            if epoch % 1 == 0:
                sampled_epochs.append(epoch)
                losses_train.append(loss)
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                losses_dev.append(loss_dev)
                pbar.set_description("PPL: %f" % ppl_dev)
                if ppl_dev < best_ppl: # the lower, the better
                    best_ppl = ppl_dev
                    best_model = copy.deepcopy(model).to('cpu')
                    patience = 3
                else:
                    patience -= 1
                    
                if patience <= 0: # Early stopping with patience
                    print("\nEarly stopping raggiunto.")
                    break 

        best_model.to(device)
        final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)    
        print('Final Test PPL = ', final_ppl)

        os.makedirs("bin", exist_ok=True)
        torch.save(best_model.state_dict(), save_path)
        print(f"Best Model successfully saved on {save_path}")

    return final_ppl