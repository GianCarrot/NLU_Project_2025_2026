import math
import copy
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from utils import get_tokenizer, load_data
from model import GPT2_LoRA
from functions import param_stats, train_loop, eval_loop

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"Using device: {DEVICE}")

    # 1. Caricamento Tokenizer e Dataset con Batch Size più grande (es. 32 o 64)
    tokenizer = get_tokenizer()
    train_loader, dev_loader, test_loader = load_data(tokenizer, batch_size=32)

    # 2. Configurazione Iperparametri LoRA
    RANK = 8
    ALPHA = 16
    LR = 5e-4  # Con AdamW e LoRA, 5e-4 o 1e-3 permette convergenza rapidissima

    # 3. Istanziazione modello
    model = GPT2_LoRA.from_pretrained("openai-community/gpt2", rank=RANK, alpha=ALPHA)
    model.to(DEVICE)

    param_stats(model)

    # 4. Ottimizzatore
    optimizer = optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=LR
    )

    # 5. VELOCIZZAZIONE: Massimo 5 Epoche (converge già alla 1a o 2a!)
    n_epochs = 10
    patience = 3
    best_ppl = math.inf
    best_model = None

    pbar = tqdm(range(n_epochs), desc="Epochs")

    for epoch in pbar:
        # Se train_loop supporta FP16, eseguirà a velocità doppia!
        loss = train_loop(train_loader, optimizer, model, DEVICE)    
        
        ppl_dev, loss_dev = eval_loop(dev_loader, model, DEVICE)
        pbar.set_description(f"Epoch {epoch+1} | Val PPL: {ppl_dev:.2f}")

        # Early stopping logic
        if ppl_dev < best_ppl:
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to('cpu')
            patience = 2
        else:
            patience -= 1
            
        if patience <= 0:
            print("\nEarly stopping triggered!")
            break 

    # 6. Valutazione finale
    best_model.to(DEVICE)
    final_ppl, _ = eval_loop(test_loader, best_model, DEVICE)    
    print(f'\n====================================')
    print(f'Test PPL: {final_ppl:.2f}')
    print(f'====================================')

    assert final_ppl < 250, "Errore: La Perplessità deve essere inferiore a 250!"

if __name__ == "__main__":
    main()