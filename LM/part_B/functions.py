import math
import torch
import numpy as np

SEED = 42

def set_seed(seed=SEED):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)

def train_loop(data, optimizer, model):
    model.train()
    loss_array = []
    number_of_tokens = []
    
    pbar = tqdm(data, desc="Training:", unit="batch", total=len(data))

    for i, (input_ids, labels, n_tokens) in enumerate(pbar):
        optimizer.zero_grad() # Zeroing the gradient
        output = model(input_ids, labels=input_ids)
        loss_array.append(output.loss.item() * n_tokens)
        number_of_tokens.append(n_tokens)
        output.loss.backward() # Compute the gradient, deleting the computational graph
        optimizer.step() # Update the weights

        if i % 100 == 0:
            pbar.set_postfix(loss=(sum(loss_array)/sum(number_of_tokens)).item())

    return sum(loss_array)/sum(number_of_tokens)

def eval_loop(data, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for input_ids, labels, n_tokens in tqdm(data, desc="Evaluating: ", unit="batch", total=len(data)):
            output = model(input_ids, labels=input_ids)
            loss_array.append(output.loss.item() * n_tokens)
            number_of_tokens.append(n_tokens)
            
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    ppl = math.exp(loss_to_return)
    return ppl, loss_to_return

def param_stats(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"total params: {total:,}")
    print(f"trainable params: {trainable:,}")
    print(f"frozen params: {total - trainable:,}")
