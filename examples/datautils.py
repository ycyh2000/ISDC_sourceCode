import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer
import os
from torch.utils.data import DataLoader, TensorDataset
os.environ['HF_DATASETS_TIMEOUT'] = '6000'

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def get_tokenizer(model):
    if "llama" in model.lower():
        tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False)
        # fix for transformer 4.28.0.dev0 compatibility
        if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
            try:
                tokenizer.bos_token_id = 1
                tokenizer.eos_token_id = 2
            except AttributeError:
                pass
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    return tokenizer

def get_wikitext2(nsamples, seed, seqlen, model, tokenizer, trust_remote_code=True):
    
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', trust_remote_code=trust_remote_code)
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', trust_remote_code=trust_remote_code)

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model, tokenizer, trust_remote_code=True):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', trust_remote_code=trust_remote_code)
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test', trust_remote_code=trust_remote_code)

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model, tokenizer, trust_remote_code=True):
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', trust_remote_code=trust_remote_code
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation', trust_remote_code=trust_remote_code
    )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc

def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=''):
    tokenizer = get_tokenizer(model)
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, tokenizer, trust_remote_code=True)
    if 'ptb' in name:
        return get_ptb(nsamples, seed, seqlen, model, tokenizer, trust_remote_code=True)
    if 'c4' in name:
        return get_c4(nsamples, seed, seqlen, model, tokenizer, trust_remote_code=True)





def get_wikitext2_(nsamples, seed, seqlen, model, tokenizer, trust_remote_code=True, shuffle=False, num_workers=0):
    # Use an independent RNG for deterministic behavior
    rng = random.Random(seed)

    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', trust_remote_code=trust_remote_code)
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', trust_remote_code=trust_remote_code)

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Build list of (input_ids, target_ids) pairs
    samples = []
    for _ in range(nsamples):
        i = rng.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        samples.append((inp, tar))

    # Convert to DataLoader with controlled shuffle and num_workers
    # Since each sample is (1, seqlen), we can stack them into tensors of shape (nsamples, seqlen)
    inputs = torch.cat([s[0] for s in samples], dim=0)   # (nsamples, seqlen)
    targets = torch.cat([s[1] for s in samples], dim=0)  # (nsamples, seqlen)
    dataset = TensorDataset(inputs, targets)
    trainloader = DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers)

    return trainloader, testenc


def get_ptb_(nsamples, seed, seqlen, model, tokenizer, trust_remote_code=True, shuffle=False, num_workers=0):
    rng = random.Random(seed)

    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', trust_remote_code=trust_remote_code)
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test', trust_remote_code=trust_remote_code)

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    samples = []
    for _ in range(nsamples):
        i = rng.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        samples.append((inp, tar))

    inputs = torch.cat([s[0] for s in samples], dim=0)
    targets = torch.cat([s[1] for s in samples], dim=0)
    dataset = TensorDataset(inputs, targets)
    trainloader = DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers)

    return trainloader, testenc


def get_c4_(nsamples, seed, seqlen, model, tokenizer, trust_remote_code=True, shuffle=False, num_workers=0):
    rng = random.Random(seed)

    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train',
        trust_remote_code=trust_remote_code
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation',
        trust_remote_code=trust_remote_code
    )

    samples = []
    for _ in range(nsamples):
        while True:
            idx = rng.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[idx]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = rng.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        samples.append((inp, tar))

    inputs = torch.cat([s[0] for s in samples], dim=0)
    targets = torch.cat([s[1] for s in samples], dim=0)
    dataset = TensorDataset(inputs, targets)
    trainloader = DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers)

    # Validation set processing (remains unchanged; deterministic)
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders_(name, nsamples=128, seed=0, seqlen=2048, model='', shuffle=False, num_workers=0):
    tokenizer = get_tokenizer(model)
    if 'wikitext2' in name:
        return get_wikitext2_(nsamples, seed, seqlen, model, tokenizer,
                             trust_remote_code=True, shuffle=shuffle, num_workers=num_workers)
    if 'ptb' in name:
        return get_ptb_(nsamples, seed, seqlen, model, tokenizer,
                       trust_remote_code=True, shuffle=shuffle, num_workers=num_workers)
    if 'c4' in name:
        return get_c4_(nsamples, seed, seqlen, model, tokenizer,
                      trust_remote_code=True, shuffle=shuffle, num_workers=num_workers)