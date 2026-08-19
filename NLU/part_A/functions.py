import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from utils import get_dataloaders, PAD_TOKEN
from model import GPT2
from conll import evaluate
from sklearn.metrics import classification_report

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, device):
    model.train()
    loss_array = []
    for batch in data:
        utterances = batch['utterances'].to(device)
        intents_gt = batch['intents'].to(device)
        y_slots = batch['y_slots'].to(device)
        slots_len = batch['slots_len'].to(device)

        optimizer.zero_grad()
        slots, intent = model(utterances, slots_len)
        slots = slots.permute(0, 2, 1)

        loss_intent = criterion_intents(intent, intents_gt)
        loss_slot = criterion_slots(slots, y_slots)
        loss = loss_intent + loss_slot

        loss_array.append(loss.item())
        loss.backward()
        optimizer.step()
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang, device):
    model.eval()
    loss_array = []
    
    all_intents_pred, all_intents_gt = [], []
    all_slots_pred, all_slots_gt = [], []
    all_utt_ids, all_lens = [], []

    use_amp = (device.type == 'cuda')

    with torch.no_grad():
        for batch in data:
            utterances = batch['utterances'].to(device)
            intents_gt = batch['intents'].to(device)
            y_slots = batch['y_slots'].to(device)
            slots_len = batch['slots_len'].to(device)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                slots, intents = model(utterances, slots_len)
                slots = slots.permute(0, 2, 1)
                loss_intent = criterion_intents(intents, intents_gt)
                loss_slot = criterion_slots(slots, y_slots)
                
            loss_array.append((loss_intent + loss_slot).item())

            all_intents_pred.append(torch.argmax(intents, dim=1).cpu())
            all_intents_gt.append(intents_gt.cpu())
            all_slots_pred.append(torch.argmax(slots, dim=1).cpu())
            all_slots_gt.append(y_slots.cpu())
            all_utt_ids.append(utterances.cpu())
            all_lens.append(slots_len.cpu())

    ref_intents = [lang.id2intent[x.item()] for cat in all_intents_gt for x in cat]
    hyp_intents = [lang.id2intent[x.item()] for cat in all_intents_pred for x in cat]

    ref_slots, hyp_slots = [], []
    for slots_pred, slots_gt, utts, lens in zip(all_slots_pred, all_slots_gt, all_utt_ids, all_lens):
        for seq_pred, seq_gt, utt, length in zip(slots_pred, slots_gt, utts, lens):
            # Exclude the trailing CLS token (-1) for slot evaluation
            l = length.item() - 1
            utterance = [lang.id2word[elem.item()] for elem in utt[:l]]
            gt_slots = [lang.id2slot[elem.item()] for elem in seq_gt[:l]]
            to_decode = [lang.id2slot[elem.item()] for elem in seq_pred[:l]]

            ref_slots.append([(utterance[i], gt_slots[i]) for i in range(l)])
            hyp_slots.append([(utterance[i], to_decode[i]) for i in range(l)])

    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception:
        results = {"total": {"f": 0}}

    report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)
    return results, report_intent, loss_array

def save_model(model, optimizer, lang, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "w2id": lang.word2id,
        "slot2id": lang.slot2id,
        "intent2id": lang.intent2id
    }, path)

def load_model(path, vocab_len, slots_len, n_intents, config, device):
    checkpoint = torch.load(path, map_location=device)
    model = GPT2(
        vocab_len, slots_len, n_intents,
        d_model=config.get('d_model', 20),
        n_heads=config.get('n_heads', 1),
        num_layers=config.get('num_layers', 1),
        ff_dim=config.get('ff_dim', 20),
        out_dropout=config.get('out_dropout', 0.0)
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model

def init_weights(mat):
    for m in mat.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

def run_pipeline(config, train_loader, dev_loader, test_loader, lang, device, runs=3):
    os.makedirs("bin", exist_ok=True)
    print()
    print("="*100)
    print(f"Running = {config['name']}")
    print("="*100)
    slot_f1s, intent_accs = [], []

    vocab_len = len(lang.word2id)
    slots_len = len(lang.id2slot)
    n_intents = len(lang.intent2id)

    for r in range(runs):
        model_path = os.path.join("bin", f"{config['id']}_run{r}_best.pt")

        if os.path.exists(model_path):
            print(f"Run {r+1}/{runs}: Model found in '{model_path}'. Skipping training...")
            model = load_model(model_path, vocab_len, slots_len, n_intents, config, device)
        else:
            print(f"Run {r+1}/{runs}: No checkpoint found in '{model_path}'. Starting training...")
            model = GPT2(
                vocab_len, slots_len, n_intents,
                d_model=config.get('d_model', 20),
                n_heads=config.get('n_heads', 1),
                num_layers=config.get('num_layers', 1),
                ff_dim=config.get('ff_dim', 20),
                out_dropout=config.get('out_dropout', 0.0)
            ).to(device)

            model.apply(init_weights)
            optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
            criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
            criterion_intents = nn.CrossEntropyLoss()

            patience = 5
            best_f1 = 0.0

            for epoch in range(100):
                train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, device)
                results_dev, _, _ = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang, device)
                
                f1 = results_dev['total']['f']
                if f1 > best_f1:
                    best_f1 = f1
                    patience = 5
                    save_model(model, optimizer, lang, model_path)
                else:
                    patience -= 1
                    if patience <= 0:
                        break

            model = load_model(model_path, vocab_len, slots_len, n_intents, config, device)

        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang, device)
        
        slot_f1s.append(results_test['total']['f'])
        intent_accs.append(intent_test['accuracy'])

        # Free VRAM to prevent memory leaks during multiple runs
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    f1_m, f1_s = np.mean(slot_f1s), np.std(slot_f1s)
    acc_m, acc_s = np.mean(intent_accs), np.std(intent_accs)

    print(f"Results -> Slot F1: {f1_m:.3f} ± {f1_s:.3f} | Intent Acc: {acc_m:.3f} ± {acc_s:.3f}")