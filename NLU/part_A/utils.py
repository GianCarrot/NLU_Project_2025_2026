import os
import json
from collections import Counter
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

PAD_TOKEN = 0

def load_data(path):
    with open(path) as f:
        return json.loads(f.read())

class Lang():
    def __init__(self, words, intents, slots, cutoff=0, cls=True):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True, cls=cls)
        self.slot2id = self.lab2id(slots, cls=cls)
        self.intent2id = self.lab2id(intents, pad=False, cls=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        # cls will have the same id as the pad token
        self.id2slot = {v:k for k, v in self.slot2id.items() if not cls or k != 'cls'}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=None, unk=True, cls=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        if cls:
            vocab['cls'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    def lab2id(self, elements, pad=True, cls=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
            vocab[elem] = len(vocab)
        if cls:
            # when predicting the slots, we want to ignore the CLS
            # CLS will only be used for intent classification
            vocab['cls'] = PAD_TOKEN
        return vocab

class IntentsAndSlots(data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk='unk', cls='cls', add_cls=True):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        self.cls = cls
        self.add_cls = add_cls
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
    
    # Auxiliary methods
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            if self.add_cls:
                tmp_seq.append(mapper[self.cls])
            res.append(tmp_seq)
        return res

def get_dataloaders(dataset_dir, device, batch_size_train=128, batch_size_eval=64):
    tmp_train_raw = load_data(os.path.join(dataset_dir, 'train.json'))
    test_raw = load_data(os.path.join(dataset_dir, 'test.json'))

    portion = 0.10
    intents = [x['intent'] for x in tmp_train_raw]
    count_y = Counter(intents)

    labels, inputs, mini_train = [], [], []
    for id_y, y in enumerate(intents):
        if count_y[y] > 1:
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])

    X_train, X_dev, _, _ = train_test_split(inputs, labels, test_size=portion, random_state=42, shuffle=True, stratify=labels)
    X_train.extend(mini_train)
    train_raw, dev_raw = X_train, X_dev

    words = sum([x['utterance'].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw 
    slots = set(sum([line['slots'].split() for line in corpus], []))
    intents = set([line['intent'] for line in corpus])

    lang = Lang(words, intents, slots, cutoff=0)

    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    def collate_fn(data_batch):
        def merge(sequences):
            lengths = [len(seq) for seq in sequences]
            max_len = 1 if max(lengths) == 0 else max(lengths)
            padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
            for i, seq in enumerate(sequences):
                padded_seqs[i, :lengths[i]] = seq
            return padded_seqs, lengths

        data_by_key = {k: [d[k] for d in data_batch] for k in data_batch[0].keys()}
        src_utt, _ = merge(data_by_key['utterance'])
        y_slots, y_lengths = merge(data_by_key['slots'])
        intent = torch.LongTensor(data_by_key['intent'])

        return {
            "utterances": src_utt,
            "intents": intent,
            "y_slots": y_slots,
            "slots_len": torch.LongTensor(y_lengths)
        }

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, collate_fn=collate_fn, shuffle=True, num_workers=2, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size_eval, collate_fn=collate_fn, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_eval, collate_fn=collate_fn, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, dev_loader, test_loader, lang