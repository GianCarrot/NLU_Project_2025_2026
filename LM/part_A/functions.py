import math
import torch
import torch.nn as nn
import torch.optim as optim

from copy import copy
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
            pbar.set_postfix(loss=(sum(loss_array)/sum(number_of_tokens)).item())

    return sum(loss_array)/sum(number_of_tokens)

def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []

    with torch.no_grad(): # It used to avoid the creation of computational graph
        for input_ids, labels, n_tokens in tqdm(data, desc="Evaluating: ", unit="batch", total=len(data)):
            output = model(input_ids)
            # need to reshape as (B, vocab, L)
            loss = eval_criterion(output.permute(0,2,1), labels)
            loss_array.append(loss.item() * n_tokens)
            number_of_tokens.append(n_tokens)
            
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    ppl = math.exp(loss_to_return)
    return ppl, loss_to_return

def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.Linear]:
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias != None:
                m.bias.data.fill_(0.01)

def run_pipeline(lr, config, train_loader, dev_loader, test_loader, vocab_len, pad_token_id, device):
    print()
    print(f"="*100)
    print(f"Pipeline Execution with Learning Rate: {lr}")
    print(f"="*100)

    model = GPT2(
        vocab_len,
        pos_emb_size=1024,
        d_model=20,
        n_heads=1,
        num_layers=1,
        ff_dim=20,
    ).to(device)
    model.apply(init_weights)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    criterion_eval = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    n_epochs = 100
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []

    best_ppl = math.inf
    best_model = None

    pbar = tqdm(range(n_epochs))
    #If the PPL is too high try to change the learning rate

    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model)
            
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(loss.item())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(loss_dev.item())
            pbar.set_description("PPL: %f" % ppl_dev)
            if ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1
                
            if patience <= 0: # Early stopping with patience
                break 

    best_model.to(device)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)    
    print('Test ppl: ', final_ppl)