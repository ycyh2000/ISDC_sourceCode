import sys
import time
import math

import torch
import torch.nn as nn
import json
import cupy as cp

from quant import *
from sparsegpt import *
from modelutils import *
import matplotlib.pyplot as plt
import numpy as np
from isdcLinear import *
import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

try:
    import wandb

    has_wandb = True
except:
    has_wandb = False


def get_opt(model, path=None):
    import torch
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    if path is not None:
        model = OPTForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)  # <-- changed to bfloat16
    else:
        # Load using bfloat16
        model = OPTForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16)
    model.seqlen = model.config.max_position_embeddings
    return model


@torch.no_grad()
def opt_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype  # automatically bfloat16
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        print(f"subset = {subset}")
        gpts = {}
        for name in subset:
            if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
                continue
            gpts[name] = SparseGPT(subset[name])
            if args.wbits < 16:
                gpts[name].quantizer = Quantizer()
                gpts[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')
            sparsity = args.sparsity
            gpts[name].fasterprune(
                sparsity, prunen=args.prunen, prunem=args.prunem, percdamp=args.percdamp, blocksize=args.blocksize
            )
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache


@torch.no_grad()
def opt_eval(model, testenc, dev, dataset: str, log_wandb: bool = False):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype  # automatically bfloat16
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.gmp:
            subset = find_layers(layer)

            print(f"subset = {subset}")

            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * args.sparsity)]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
                       :, (i * model.seqlen):((i + 1) * model.seqlen)
                       ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    if log_wandb:
        wandb.log({f'{dataset}/perplexity': ppl.item()})

    model.config.use_cache = use_cache


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='OPT model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--sparsity', type=float, default=0,
        help='Target sparsity'
    )
    parser.add_argument(
        '--prunen', type=int, default=0,
        help='N for N:M pruning.'
    )
    parser.add_argument(
        '--prunem', type=int, default=0,
        help='M for N:M pruning.'
    )
    parser.add_argument(
        '--blocksize', type=int, default=128,
        help='Blocksize to use for adaptive mask selection.'
    )
    parser.add_argument(
        '--gmp', action='store_true',
        help='Whether to run the GMP baseline.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16,
        help='Whether to quantize as well.'
    )
    parser.add_argument(
        '--minlayer', type=int, default=-1,
        help='Prune all layers with id >= this.'
    )
    parser.add_argument(
        '--maxlayer', type=int, default=1000,
        help='Prune all layers with id < this.'
    )
    parser.add_argument(
        '--prune_only', type=str, default='',
        help='Prune only layers that contain this text.'
    )
    parser.add_argument(
        '--invert', action='store_true',
        help='Invert subset.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Path to saved model.'
    )
    parser.add_argument(
        '--log_wandb', action='store_true',
        help='Whether to log to wandb.'
    )

    print("opt_bfloat16_isdc")

    args = parser.parse_args()

    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    model = get_opt(args.model)
    model.eval()

    print("model.dtype = ", model.dtype)  # will show torch.bfloat16

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    # Fix DEV variable, use defined device
    DEV = device

    if (args.sparsity or args.prunen) and not args.gmp:
        tick = time.time()
        opt_sequential(model, dataloader, DEV)
        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()))
            if 'fc2' in n:
                break
        print(time.time() - tick)

    for dataset in ['wikitext2', 'ptb', 'c4']:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        opt_eval(model, testloader, DEV, dataset, args.log_wandb)

    if args.save:
        model.save_pretrained(args.save)

    print("\nAll tests completed!")

    dtype_ = torch.bfloat16
    mantissa_bitwidth = get_mantissa_bitwidth(dtype_)

    folder_path = "opt1.3b_isdc_compressed_weight_4_8"
    file_names = os.listdir(folder_path)
    layer_name_list = []
    for name in file_names:
        if "_startAddr" in name:
            new_name = name.replace("_startAddr.json", "")
            layer_name_list.append(new_name)
    # print(layer_name_list)

    for name, module in model.named_modules():
        if name.replace('.', '_') in layer_name_list:
            print(f"current linear name = {name}")
            layer_name = f"{folder_path}/{name.replace('.', '_')}"
            with open(f"{layer_name}.json", 'r', encoding='utf-8') as f:
                message = json.load(f)

            ocSize = message["ocSize"]
            icSize = message["icSize"]

            # if ocSize != 768 or icSize != 768:
            #     continue

            minibatch = 1
            data_width_len = math.ceil(math.log2(mantissa_bitwidth - message["base_width"] + 1))
            base_width = message["base_width"]

            # print(f"layer_name = {layer_name}, ocSize = {ocSize}, icSize = {icSize}")

            data_width_str = read_txt_to_string(f"{layer_name}_data_width_str.txt")
            weight_data_str = read_txt_to_string(f"{layer_name}_weight_data_str.txt")
            with open(f"{layer_name}_index_data_str.json", 'r', encoding='utf-8') as f:
                index_data = json.load(f)

            index_data_size = len(index_data) * len(index_data[0])
            index_data = np.array(index_data, dtype=np.uint8)

            with open(f"{layer_name}_startAddr.json", 'r', encoding='utf-8') as f:
                startAddr = json.load(f)
            with open(f"{layer_name}_sign.json", 'r', encoding='utf-8') as f:
                sign_data = json.load(f)
            with open(f"{layer_name}_exp.json", 'r', encoding='utf-8') as f:
                exp_data = json.load(f)

            exp_data_size = len(exp_data) * len(exp_data[0])
            exp_data = np.array(exp_data, dtype=np.uint8)

            sign_data_size = len(sign_data) * len(sign_data[0])
            sign_data = np.array(sign_data, dtype=np.bool)

            compressed_data_width = string_to_uint32_array(data_width_str)
            compressed_data_width_size = compressed_data_width.size
            compressed_weight_data = string_to_uint32_array(weight_data_str)

            compressed_weight_data_size = compressed_weight_data.size
            # compressed_index_data = string_to_uint32_array(index_data_str)
            # compressed_index_data_size = compressed_index_data.size

            compressed = {}
            compressed['compressed_data_width'] = compressed_data_width
            compressed['compressed_data_width_size'] = compressed_data_width_size
            compressed['compressed_weight_data'] = compressed_weight_data
            compressed['compressed_weight_data_size'] = compressed_weight_data_size
            compressed['index_data'] = index_data
            compressed['index_data_size'] = index_data_size
            compressed['sign_data'] = sign_data
            compressed['sign_data_size'] = sign_data_size
            compressed['exp_data'] = exp_data
            compressed['exp_data_size'] = exp_data_size
            compressed['startAddr'] = startAddr
            compressed['data_width_len'] = data_width_len
            compressed['base_width'] = base_width

            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr_name = parts[-1]

            new_layer = replace_linear_layer(module, compressed)
            setattr(parent, attr_name, new_layer)

            pass
            # print(name, module)
        # print(name)

    for name, module in model.named_modules():
        print(name, module)

    torch.save(model, "opt-125m_isdc.pth")

    # Fix DEV variable, use defined device
    DEV = device

    for dataset in ['wikitext2', 'ptb', 'c4']:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        opt_eval(model, testloader, DEV, dataset, args.log_wandb)
import sys
import time
import math

import torch
import torch.nn as nn
import json
import cupy as cp

from quant import *
from sparsegpt import *
from modelutils import *
import matplotlib.pyplot as plt
import numpy as np
from isdcLinear import *
import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

try:
    import wandb

    has_wandb = True
except:
    has_wandb = False


def get_opt(model, path=None):
    import torch
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    if path is not None:
        model = OPTForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)  # <-- changed to bfloat16
    else:
        # Load using bfloat16
        model = OPTForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16)
    model.seqlen = model.config.max_position_embeddings
    return model


@torch.no_grad()
def opt_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype  # automatically bfloat16
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        print(f"subset = {subset}")
        gpts = {}
        for name in subset:
            if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
                continue
            gpts[name] = SparseGPT(subset[name])
            if args.wbits < 16:
                gpts[name].quantizer = Quantizer()
                gpts[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')
            sparsity = args.sparsity
            gpts[name].fasterprune(
                sparsity, prunen=args.prunen, prunem=args.prunem, percdamp=args.percdamp, blocksize=args.blocksize
            )
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache


@torch.no_grad()
def opt_eval(model, testenc, dev, dataset: str, log_wandb: bool = False):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype  # automatically bfloat16
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.gmp:
            subset = find_layers(layer)

            print(f"subset = {subset}")

            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * args.sparsity)]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
                       :, (i * model.seqlen):((i + 1) * model.seqlen)
                       ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    if log_wandb:
        wandb.log({f'{dataset}/perplexity': ppl.item()})

    model.config.use_cache = use_cache


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='OPT model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--sparsity', type=float, default=0,
        help='Target sparsity'
    )
    parser.add_argument(
        '--prunen', type=int, default=0,
        help='N for N:M pruning.'
    )
    parser.add_argument(
        '--prunem', type=int, default=0,
        help='M for N:M pruning.'
    )
    parser.add_argument(
        '--blocksize', type=int, default=128,
        help='Blocksize to use for adaptive mask selection.'
    )
    parser.add_argument(
        '--gmp', action='store_true',
        help='Whether to run the GMP baseline.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16,
        help='Whether to quantize as well.'
    )
    parser.add_argument(
        '--minlayer', type=int, default=-1,
        help='Prune all layers with id >= this.'
    )
    parser.add_argument(
        '--maxlayer', type=int, default=1000,
        help='Prune all layers with id < this.'
    )
    parser.add_argument(
        '--prune_only', type=str, default='',
        help='Prune only layers that contain this text.'
    )
    parser.add_argument(
        '--invert', action='store_true',
        help='Invert subset.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Path to saved model.'
    )
    parser.add_argument(
        '--log_wandb', action='store_true',
        help='Whether to log to wandb.'
    )

    print("opt_bfloat16_isdc")

    args = parser.parse_args()

    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    model = get_opt(args.model)
    model.eval()

    print("model.dtype = ", model.dtype)  # will show torch.bfloat16

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    # Fix DEV variable, use defined device
    DEV = device

    if (args.sparsity or args.prunen) and not args.gmp:
        tick = time.time()
        opt_sequential(model, dataloader, DEV)
        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()))
            if 'fc2' in n:
                break
        print(time.time() - tick)

    for dataset in ['wikitext2', 'ptb', 'c4']:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        opt_eval(model, testloader, DEV, dataset, args.log_wandb)

    if args.save:
        model.save_pretrained(args.save)

    print("\nAll tests completed!")

    dtype_ = torch.bfloat16
    mantissa_bitwidth = get_mantissa_bitwidth(dtype_)

    folder_path = "opt1.3b_isdc_compressed_weight_4_8"
    file_names = os.listdir(folder_path)
    layer_name_list = []
    for name in file_names:
        if "_startAddr" in name:
            new_name = name.replace("_startAddr.json", "")
            layer_name_list.append(new_name)
    # print(layer_name_list)

    for name, module in model.named_modules():
        if name.replace('.', '_') in layer_name_list:
            print(f"current linear name = {name}")
            layer_name = f"{folder_path}/{name.replace('.', '_')}"
            with open(f"{layer_name}.json", 'r', encoding='utf-8') as f:
                message = json.load(f)

            ocSize = message["ocSize"]
            icSize = message["icSize"]

            # if ocSize != 768 or icSize != 768:
            #     continue

            minibatch = 1
            data_width_len = math.ceil(math.log2(mantissa_bitwidth - message["base_width"] + 1))
            base_width = message["base_width"]

            # print(f"layer_name = {layer_name}, ocSize = {ocSize}, icSize = {icSize}")

            data_width_str = read_txt_to_string(f"{layer_name}_data_width_str.txt")
            weight_data_str = read_txt_to_string(f"{layer_name}_weight_data_str.txt")
            with open(f"{layer_name}_index_data_str.json", 'r', encoding='utf-8') as f:
                index_data = json.load(f)

            index_data_size = len(index_data) * len(index_data[0])
            index_data = np.array(index_data, dtype=np.uint8)

            with open(f"{layer_name}_startAddr.json", 'r', encoding='utf-8') as f:
                startAddr = json.load(f)
            with open(f"{layer_name}_sign.json", 'r', encoding='utf-8') as f:
                sign_data = json.load(f)
            with open(f"{layer_name}_exp.json", 'r', encoding='utf-8') as f:
                exp_data = json.load(f)

            exp_data_size = len(exp_data) * len(exp_data[0])
            exp_data = np.array(exp_data, dtype=np.uint8)

            sign_data_size = len(sign_data) * len(sign_data[0])
            sign_data = np.array(sign_data, dtype=np.bool)

            compressed_data_width = string_to_uint32_array(data_width_str)
            compressed_data_width_size = compressed_data_width.size
            compressed_weight_data = string_to_uint32_array(weight_data_str)

            compressed_weight_data_size = compressed_weight_data.size
            # compressed_index_data = string_to_uint32_array(index_data_str)
            # compressed_index_data_size = compressed_index_data.size

            compressed = {}
            compressed['compressed_data_width'] = compressed_data_width
            compressed['compressed_data_width_size'] = compressed_data_width_size
            compressed['compressed_weight_data'] = compressed_weight_data
            compressed['compressed_weight_data_size'] = compressed_weight_data_size
            compressed['index_data'] = index_data
            compressed['index_data_size'] = index_data_size
            compressed['sign_data'] = sign_data
            compressed['sign_data_size'] = sign_data_size
            compressed['exp_data'] = exp_data
            compressed['exp_data_size'] = exp_data_size
            compressed['startAddr'] = startAddr
            compressed['data_width_len'] = data_width_len
            compressed['base_width'] = base_width

            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr_name = parts[-1]

            new_layer = replace_linear_layer(module, compressed)
            setattr(parent, attr_name, new_layer)

            pass
            # print(name, module)
        # print(name)

    for name, module in model.named_modules():
        print(name, module)

    torch.save(model, "opt-125m_isdc.pth")

    # Fix DEV variable, use defined device
    DEV = device

    for dataset in ['wikitext2', 'ptb', 'c4']:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        opt_eval(model, testloader, DEV, dataset, args.log_wandb)
import sys
import time
import math

import torch
import torch.nn as nn
import json
import cupy as cp

from quant import *
from sparsegpt import *
from modelutils import *
import matplotlib.pyplot as plt
import numpy as np
from isdcLinear import *
import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

try:
    import wandb

    has_wandb = True
except:
    has_wandb = False


def get_opt(model, path=None):
    import torch
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    if path is not None:
        model = OPTForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)  # <-- changed to bfloat16
    else:
        # Load using bfloat16
        model = OPTForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16)
    model.seqlen = model.config.max_position_embeddings
    return model


@torch.no_grad()
def opt_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype  # automatically bfloat16
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        print(f"subset = {subset}")
        gpts = {}
        for name in subset:
            if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
                continue
            gpts[name] = SparseGPT(subset[name])
            if args.wbits < 16:
                gpts[name].quantizer = Quantizer()
                gpts[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')
            sparsity = args.sparsity
            gpts[name].fasterprune(
                sparsity, prunen=args.prunen, prunem=args.prunem, percdamp=args.percdamp, blocksize=args.blocksize
            )
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache


@torch.no_grad()
def opt_eval(model, testenc, dev, dataset: str, log_wandb: bool = False):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype  # automatically bfloat16
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.gmp:
            subset = find_layers(layer)

            print(f"subset = {subset}")

            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * args.sparsity)]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
                       :, (i * model.seqlen):((i + 1) * model.seqlen)
                       ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    if log_wandb:
        wandb.log({f'{dataset}/perplexity': ppl.item()})

    model.config.use_cache = use_cache


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='OPT model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--sparsity', type=float, default=0,
        help='Target sparsity'
    )
    parser.add_argument(
        '--prunen', type=int, default=0,
        help='N for N:M pruning.'
    )
    parser.add_argument(
        '--prunem', type=int, default=0,
        help='M for N:M pruning.'
    )
    parser.add_argument(
        '--blocksize', type=int, default=128,
        help='Blocksize to use for adaptive mask selection.'
    )
    parser.add_argument(
        '--gmp', action='store_true',
        help='Whether to run the GMP baseline.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16,
        help='Whether to quantize as well.'
    )
    parser.add_argument(
        '--minlayer', type=int, default=-1,
        help='Prune all layers with id >= this.'
    )
    parser.add_argument(
        '--maxlayer', type=int, default=1000,
        help='Prune all layers with id < this.'
    )
    parser.add_argument(
        '--prune_only', type=str, default='',
        help='Prune only layers that contain this text.'
    )
    parser.add_argument(
        '--invert', action='store_true',
        help='Invert subset.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Path to saved model.'
    )
    parser.add_argument(
        '--log_wandb', action='store_true',
        help='Whether to log to wandb.'
    )

    print("opt_bfloat16_isdc")

    args = parser.parse_args()

    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    model = get_opt(args.model)
    model.eval()

    print("model.dtype = ", model.dtype)  # will show torch.bfloat16

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    # Fix DEV variable, use defined device
    DEV = device

    if (args.sparsity or args.prunen) and not args.gmp:
        tick = time.time()
        opt_sequential(model, dataloader, DEV)
        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()))
            if 'fc2' in n:
                break
        print(time.time() - tick)

    for dataset in ['wikitext2', 'ptb', 'c4']:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        opt_eval(model, testloader, DEV, dataset, args.log_wandb)

    if args.save:
        model.save_pretrained(args.save)

    print("\nAll tests completed!")

    dtype_ = torch.bfloat16
    mantissa_bitwidth = get_mantissa_bitwidth(dtype_)

    folder_path = "opt1.3b_isdc_compressed_weight_4_8"
    file_names = os.listdir(folder_path)
    layer_name_list = []
    for name in file_names:
        if "_startAddr" in name:
            new_name = name.replace("_startAddr.json", "")
            layer_name_list.append(new_name)
    # print(layer_name_list)

    for name, module in model.named_modules():
        if name.replace('.', '_') in layer_name_list:
            print(f"current linear name = {name}")
            layer_name = f"{folder_path}/{name.replace('.', '_')}"
            with open(f"{layer_name}.json", 'r', encoding='utf-8') as f:
                message = json.load(f)

            ocSize = message["ocSize"]
            icSize = message["icSize"]

            # if ocSize != 768 or icSize != 768:
            #     continue

            minibatch = 1
            data_width_len = math.ceil(math.log2(mantissa_bitwidth - message["base_width"] + 1))
            base_width = message["base_width"]

            # print(f"layer_name = {layer_name}, ocSize = {ocSize}, icSize = {icSize}")

            data_width_str = read_txt_to_string(f"{layer_name}_data_width_str.txt")
            weight_data_str = read_txt_to_string(f"{layer_name}_weight_data_str.txt")
            with open(f"{layer_name}_index_data_str.json", 'r', encoding='utf-8') as f:
                index_data = json.load(f)

            index_data_size = len(index_data) * len(index_data[0])
            index_data = np.array(index_data, dtype=np.uint8)

            with open(f"{layer_name}_startAddr.json", 'r', encoding='utf-8') as f:
                startAddr = json.load(f)
            with open(f"{layer_name}_sign.json", 'r', encoding='utf-8') as f:
                sign_data = json.load(f)
            with open(f"{layer_name}_exp.json", 'r', encoding='utf-8') as f:
                exp_data = json.load(f)

            exp_data_size = len(exp_data) * len(exp_data[0])
            exp_data = np.array(exp_data, dtype=np.uint8)

            sign_data_size = len(sign_data) * len(sign_data[0])
            sign_data = np.array(sign_data, dtype=np.bool)

            compressed_data_width = string_to_uint32_array(data_width_str)
            compressed_data_width_size = compressed_data_width.size
            compressed_weight_data = string_to_uint32_array(weight_data_str)

            compressed_weight_data_size = compressed_weight_data.size
            # compressed_index_data = string_to_uint32_array(index_data_str)
            # compressed_index_data_size = compressed_index_data.size

            compressed = {}
            compressed['compressed_data_width'] = compressed_data_width
            compressed['compressed_data_width_size'] = compressed_data_width_size
            compressed['compressed_weight_data'] = compressed_weight_data
            compressed['compressed_weight_data_size'] = compressed_weight_data_size
            compressed['index_data'] = index_data
            compressed['index_data_size'] = index_data_size
            compressed['sign_data'] = sign_data
            compressed['sign_data_size'] = sign_data_size
            compressed['exp_data'] = exp_data
            compressed['exp_data_size'] = exp_data_size
            compressed['startAddr'] = startAddr
            compressed['data_width_len'] = data_width_len
            compressed['base_width'] = base_width

            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr_name = parts[-1]

            new_layer = replace_linear_layer(module, compressed)
            setattr(parent, attr_name, new_layer)

            pass
            # print(name, module)
        # print(name)

    for name, module in model.named_modules():
        print(name, module)

    torch.save(model, "opt-125m_isdc.pth")

    # Fix DEV variable, use defined device
    DEV = device

    for dataset in ['wikitext2', 'ptb', 'c4']:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        opt_eval(model, testloader, DEV, dataset, args.log_wandb)
import sys
import time
import math

import torch
import torch.nn as nn
import json
import cupy as cp

from quant import *
from sparsegpt import *
from modelutils import *
import matplotlib.pyplot as plt
import numpy as np
from isdcLinear import *
import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

try:
    import wandb

    has_wandb = True
except:
    has_wandb = False


def get_opt(model, path=None):
    import torch
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    if path is not None:
        model = OPTForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)  # <-- changed to bfloat16
    else:
        # Load using bfloat16
        model = OPTForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16)
    model.seqlen = model.config.max_position_embeddings
    return model


@torch.no_grad()
def opt_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype  # automatically bfloat16
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        print(f"subset = {subset}")
        gpts = {}
        for name in subset:
            if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
                continue
            gpts[name] = SparseGPT(subset[name])
            if args.wbits < 16:
                gpts[name].quantizer = Quantizer()
                gpts[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')
            sparsity = args.sparsity
            gpts[name].fasterprune(
                sparsity, prunen=args.prunen, prunem=args.prunem, percdamp=args.percdamp, blocksize=args.blocksize
            )
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache


@torch.no_grad()
def opt_eval(model, testenc, dev, dataset: str, log_wandb: bool = False):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype  # automatically bfloat16
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.gmp:
            subset = find_layers(layer)

            print(f"subset = {subset}")

            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * args.sparsity)]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
                       :, (i * model.seqlen):((i + 1) * model.seqlen)
                       ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    if log_wandb:
        wandb.log({f'{dataset}/perplexity': ppl.item()})

    model.config.use_cache = use_cache


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='OPT model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--sparsity', type=float, default=0,
        help='Target sparsity'
    )
    parser.add_argument(
        '--prunen', type=int, default=0,
        help='N for N:M pruning.'
    )
    parser.add_argument(
        '--prunem', type=int, default=0,
        help='M for N:M pruning.'
    )
    parser.add_argument(
        '--blocksize', type=int, default=128,
        help='Blocksize to use for adaptive mask selection.'
    )
    parser.add_argument(
        '--gmp', action='store_true',
        help='Whether to run the GMP baseline.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16,
        help='Whether to quantize as well.'
    )
    parser.add_argument(
        '--minlayer', type=int, default=-1,
        help='Prune all layers with id >= this.'
    )
    parser.add_argument(
        '--maxlayer', type=int, default=1000,
        help='Prune all layers with id < this.'
    )
    parser.add_argument(
        '--prune_only', type=str, default='',
        help='Prune only layers that contain this text.'
    )
    parser.add_argument(
        '--invert', action='store_true',
        help='Invert subset.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Path to saved model.'
    )
    parser.add_argument(
        '--log_wandb', action='store_true',
        help='Whether to log to wandb.'
    )

    print("opt_bfloat16_isdc")

    args = parser.parse_args()

    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    model = get_opt(args.model)
    model.eval()

    print("model.dtype = ", model.dtype)  # will show torch.bfloat16

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    # Fix DEV variable, use defined device
    DEV = device

    if (args.sparsity or args.prunen) and not args.gmp:
        tick = time.time()
        opt_sequential(model, dataloader, DEV)
        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()))
            if 'fc2' in n:
                break
        print(time.time() - tick)

    for dataset in ['wikitext2', 'ptb', 'c4']:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        opt_eval(model, testloader, DEV, dataset, args.log_wandb)

    if args.save:
        model.save_pretrained(args.save)

    print("\nAll tests completed!")

    dtype_ = torch.bfloat16
    mantissa_bitwidth = get_mantissa_bitwidth(dtype_)

    folder_path = "opt1.3b_isdc_compressed_weight_4_8"
    file_names = os.listdir(folder_path)
    layer_name_list = []
    for name in file_names:
        if "_startAddr" in name:
            new_name = name.replace("_startAddr.json", "")
            layer_name_list.append(new_name)
    # print(layer_name_list)

    for name, module in model.named_modules():
        if name.replace('.', '_') in layer_name_list:
            print(f"current linear name = {name}")
            layer_name = f"{folder_path}/{name.replace('.', '_')}"
            with open(f"{layer_name}.json", 'r', encoding='utf-8') as f:
                message = json.load(f)

            ocSize = message["ocSize"]
            icSize = message["icSize"]

            # if ocSize != 768 or icSize != 768:
            #     continue

            minibatch = 1
            data_width_len = math.ceil(math.log2(mantissa_bitwidth - message["base_width"] + 1))
            base_width = message["base_width"]

            # print(f"layer_name = {layer_name}, ocSize = {ocSize}, icSize = {icSize}")

            data_width_str = read_txt_to_string(f"{layer_name}_data_width_str.txt")
            weight_data_str = read_txt_to_string(f"{layer_name}_weight_data_str.txt")
            with open(f"{layer_name}_index_data_str.json", 'r', encoding='utf-8') as f:
                index_data = json.load(f)

            index_data_size = len(index_data) * len(index_data[0])
            index_data = np.array(index_data, dtype=np.uint8)

            with open(f"{layer_name}_startAddr.json", 'r', encoding='utf-8') as f:
                startAddr = json.load(f)
            with open(f"{layer_name}_sign.json", 'r', encoding='utf-8') as f:
                sign_data = json.load(f)
            with open(f"{layer_name}_exp.json", 'r', encoding='utf-8') as f:
                exp_data = json.load(f)

            exp_data_size = len(exp_data) * len(exp_data[0])
            exp_data = np.array(exp_data, dtype=np.uint8)

            sign_data_size = len(sign_data) * len(sign_data[0])
            sign_data = np.array(sign_data, dtype=np.bool)

            compressed_data_width = string_to_uint32_array(data_width_str)
            compressed_data_width_size = compressed_data_width.size
            compressed_weight_data = string_to_uint32_array(weight_data_str)

            compressed_weight_data_size = compressed_weight_data.size
            # compressed_index_data = string_to_uint32_array(index_data_str)
            # compressed_index_data_size = compressed_index_data.size

            compressed = {}
            compressed['compressed_data_width'] = compressed_data_width
            compressed['compressed_data_width_size'] = compressed_data_width_size
            compressed['compressed_weight_data'] = compressed_weight_data
            compressed['compressed_weight_data_size'] = compressed_weight_data_size
            compressed['index_data'] = index_data
            compressed['index_data_size'] = index_data_size
            compressed['sign_data'] = sign_data
            compressed['sign_data_size'] = sign_data_size
            compressed['exp_data'] = exp_data
            compressed['exp_data_size'] = exp_data_size
            compressed['startAddr'] = startAddr
            compressed['data_width_len'] = data_width_len
            compressed['base_width'] = base_width

            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr_name = parts[-1]

            new_layer = replace_linear_layer(module, compressed)
            setattr(parent, attr_name, new_layer)

            pass
            # print(name, module)
        # print(name)

    for name, module in model.named_modules():
        print(name, module)

    torch.save(model, "opt-125m_isdc.pth")

    # Fix DEV variable, use defined device
    DEV = device

    for dataset in ['wikitext2', 'ptb', 'c4']:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        opt_eval(model, testloader, DEV, dataset, args.log_wandb)
import sys
import time
import math

import torch
import torch.nn as nn
import json
import cupy as cp

from quant import *
from sparsegpt import *
from modelutils import *
import matplotlib.pyplot as plt
import numpy as np
from isdcLinear import *
import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

try:
    import wandb

    has_wandb = True
except:
    has_wandb = False


def get_opt(model, path=None):
    import torch
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    if path is not None:
        model = OPTForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)  # <-- changed to bfloat16
    else:
        # Load using bfloat16
        model = OPTForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16)
    model.seqlen = model.config.max_position_embeddings
    return model


@torch.no_grad()
def opt_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype  # automatically bfloat16
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        print(f"subset = {subset}")
        gpts = {}
        for name in subset:
            if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
                continue
            gpts[name] = SparseGPT(subset[name])
            if args.wbits < 16:
                gpts[name].quantizer = Quantizer()
                gpts[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')
            sparsity = args.sparsity
            gpts[name].fasterprune(
                sparsity, prunen=args.prunen, prunem=args.prunem, percdamp=args.percdamp, blocksize=args.blocksize
            )
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache


@torch.no_grad()
def opt_eval(model, testenc, dev, dataset: str, log_wandb: bool = False):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype  # automatically bfloat16
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.gmp:
            subset = find_layers(layer)

            print(f"subset = {subset}")

            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * args.sparsity)]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
                       :, (i * model.seqlen):((i + 1) * model.seqlen)
                       ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    if log_wandb:
        wandb.log({f'{dataset}/perplexity': ppl.item()})

    model.config.use_cache = use_cache


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='OPT model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--sparsity', type=float, default=0,
        help='Target sparsity'
    )
    parser.add_argument(
        '--prunen', type=int, default=0,
        help='N for N:M pruning.'
    )
    parser.add_argument(
        '--prunem', type=int, default=0,
        help='M for N:M pruning.'
    )
    parser.add_argument(
        '--blocksize', type=int, default=128,
        help='Blocksize to use for adaptive mask selection.'
    )
    parser.add_argument(
        '--gmp', action='store_true',
        help='Whether to run the GMP baseline.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16,
        help='Whether to quantize as well.'
    )
    parser.add_argument(
        '--minlayer', type=int, default=-1,
        help='Prune all layers with id >= this.'
    )
    parser.add_argument(
        '--maxlayer', type=int, default=1000,
        help='Prune all layers with id < this.'
    )
    parser.add_argument(
        '--prune_only', type=str, default='',
        help='Prune only layers that contain this text.'
    )
    parser.add_argument(
        '--invert', action='store_true',
        help='Invert subset.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Path to saved model.'
    )
    parser.add_argument(
        '--log_wandb', action='store_true',
        help='Whether to log to wandb.'
    )

    print("opt_bfloat16_isdc")

    args = parser.parse_args()

    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    model = get_opt(args.model)
    model.eval()

    print("model.dtype = ", model.dtype)  # will show torch.bfloat16

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    # Fix DEV variable, use defined device
    DEV = device

    if (args.sparsity or args.prunen) and not args.gmp:
        tick = time.time()
        opt_sequential(model, dataloader, DEV)
        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()))
            if 'fc2' in n:
                break
        print(time.time() - tick)

    for dataset in ['wikitext2', 'ptb', 'c4']:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        opt_eval(model, testloader, DEV, dataset, args.log_wandb)

    if args.save:
        model.save_pretrained(args.save)

    print("\nAll tests completed!")

    dtype_ = torch.bfloat16
    mantissa_bitwidth = get_mantissa_bitwidth(dtype_)

    folder_path = "opt1.3b_isdc_compressed_weight_4_8"
    file_names = os.listdir(folder_path)
    layer_name_list = []
    for name in file_names:
        if "_startAddr" in name:
            new_name = name.replace("_startAddr.json", "")
            layer_name_list.append(new_name)
    # print(layer_name_list)

    for name, module in model.named_modules():
        if name.replace('.', '_') in layer_name_list:
            print(f"current linear name = {name}")
            layer_name = f"{folder_path}/{name.replace('.', '_')}"
            with open(f"{layer_name}.json", 'r', encoding='utf-8') as f:
                message = json.load(f)

            ocSize = message["ocSize"]
            icSize = message["icSize"]

            # if ocSize != 768 or icSize != 768:
            #     continue

            minibatch = 1
            data_width_len = math.ceil(math.log2(mantissa_bitwidth - message["base_width"] + 1))
            base_width = message["base_width"]

            # print(f"layer_name = {layer_name}, ocSize = {ocSize}, icSize = {icSize}")

            data_width_str = read_txt_to_string(f"{layer_name}_data_width_str.txt")
            weight_data_str = read_txt_to_string(f"{layer_name}_weight_data_str.txt")
            with open(f"{layer_name}_index_data_str.json", 'r', encoding='utf-8') as f:
                index_data = json.load(f)

            index_data_size = len(index_data) * len(index_data[0])
            index_data = np.array(index_data, dtype=np.uint8)

            with open(f"{layer_name}_startAddr.json", 'r', encoding='utf-8') as f:
                startAddr = json.load(f)
            with open(f"{layer_name}_sign.json", 'r', encoding='utf-8') as f:
                sign_data = json.load(f)
            with open(f"{layer_name}_exp.json", 'r', encoding='utf-8') as f:
                exp_data = json.load(f)

            exp_data_size = len(exp_data) * len(exp_data[0])
            exp_data = np.array(exp_data, dtype=np.uint8)

            sign_data_size = len(sign_data) * len(sign_data[0])
            sign_data = np.array(sign_data, dtype=np.bool)

            compressed_data_width = string_to_uint32_array(data_width_str)
            compressed_data_width_size = compressed_data_width.size
            compressed_weight_data = string_to_uint32_array(weight_data_str)

            compressed_weight_data_size = compressed_weight_data.size
            # compressed_index_data = string_to_uint32_array(index_data_str)
            # compressed_index_data_size = compressed_index_data.size

            compressed = {}
            compressed['compressed_data_width'] = compressed_data_width
            compressed['compressed_data_width_size'] = compressed_data_width_size
            compressed['compressed_weight_data'] = compressed_weight_data
            compressed['compressed_weight_data_size'] = compressed_weight_data_size
            compressed['index_data'] = index_data
            compressed['index_data_size'] = index_data_size
            compressed['sign_data'] = sign_data
            compressed['sign_data_size'] = sign_data_size
            compressed['exp_data'] = exp_data
            compressed['exp_data_size'] = exp_data_size
            compressed['startAddr'] = startAddr
            compressed['data_width_len'] = data_width_len
            compressed['base_width'] = base_width

            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr_name = parts[-1]

            new_layer = replace_linear_layer(module, compressed)
            setattr(parent, attr_name, new_layer)

            pass
            # print(name, module)
        # print(name)

    for name, module in model.named_modules():
        print(name, module)

    torch.save(model, "opt-125m_isdc.pth")

    # Fix DEV variable, use defined device
    DEV = device

    for dataset in ['wikitext2', 'ptb', 'c4']:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        opt_eval(model, testloader, DEV, dataset, args.log_wandb)
import sys
import time
import math

import torch
import torch.nn as nn
import json
import cupy as cp

from quant import *
from sparsegpt import *
from modelutils import *
import matplotlib.pyplot as plt
import numpy as np
from isdcLinear import *
import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

try:
    import wandb

    has_wandb = True
except:
    has_wandb = False


def get_opt(model, path=None):
    import torch
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    if path is not None:
        model = OPTForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)  # <-- changed to bfloat16
    else:
        # Load using bfloat16
        model = OPTForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16)
    model.seqlen = model.config.max_position_embeddings
    return model


@torch.no_grad()
def opt_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype  # automatically bfloat16
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        print(f"subset = {subset}")
        gpts = {}
        for name in subset:
            if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
                continue
            gpts[name] = SparseGPT(subset[name])
            if args.wbits < 16:
                gpts[name].quantizer = Quantizer()
                gpts[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')
            sparsity = args.sparsity
            gpts[name].fasterprune(
                sparsity, prunen=args.prunen, prunem=args.prunem, percdamp=args.percdamp, blocksize=args.blocksize
            )
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache


@torch.no_grad()
def opt_eval(model, testenc, dev, dataset: str, log_wandb: bool = False):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype  # automatically bfloat16
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.gmp:
            subset = find_layers(layer)

            print(f"subset = {subset}")

            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * args.sparsity)]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
                       :, (i * model.seqlen):((i + 1) * model.seqlen)
                       ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    if log_wandb:
        wandb.log({f'{dataset}/perplexity': ppl.item()})

    model.config.use_cache = use_cache


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='OPT model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--sparsity', type=float, default=0,
        help='Target sparsity'
    )
    parser.add_argument(
        '--prunen', type=int, default=0,
        help='N for N:M pruning.'
    )
    parser.add_argument(
        '--prunem', type=int, default=0,
        help='M for N:M pruning.'
    )
    parser.add_argument(
        '--blocksize', type=int, default=128,
        help='Blocksize to use for adaptive mask selection.'
    )
    parser.add_argument(
        '--gmp', action='store_true',
        help='Whether to run the GMP baseline.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16,
        help='Whether to quantize as well.'
    )
    parser.add_argument(
        '--minlayer', type=int, default=-1,
        help='Prune all layers with id >= this.'
    )
    parser.add_argument(
        '--maxlayer', type=int, default=1000,
        help='Prune all layers with id < this.'
    )
    parser.add_argument(
        '--prune_only', type=str, default='',
        help='Prune only layers that contain this text.'
    )
    parser.add_argument(
        '--invert', action='store_true',
        help='Invert subset.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Path to saved model.'
    )
    parser.add_argument(
        '--log_wandb', action='store_true',
        help='Whether to log to wandb.'
    )

    print("opt_bfloat16_isdc")

    args = parser.parse_args()

    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    model = get_opt(args.model)
    model.eval()

    print("model.dtype = ", model.dtype)  # will show torch.bfloat16

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    # Fix DEV variable, use defined device
    DEV = device

    if (args.sparsity or args.prunen) and not args.gmp:
        tick = time.time()
        opt_sequential(model, dataloader, DEV)
        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()))
            if 'fc2' in n:
                break
        print(time.time() - tick)

    for dataset in ['wikitext2', 'ptb', 'c4']:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        opt_eval(model, testloader, DEV, dataset, args.log_wandb)

    if args.save:
        model.save_pretrained(args.save)

    print("\nAll tests completed!")

    dtype_ = torch.bfloat16
    mantissa_bitwidth = get_mantissa_bitwidth(dtype_)

    folder_path = "opt1.3b_isdc_compressed_weight_4_8"
    file_names = os.listdir(folder_path)
    layer_name_list = []
    for name in file_names:
        if "_startAddr" in name:
            new_name = name.replace("_startAddr.json", "")
            layer_name_list.append(new_name)
    # print(layer_name_list)

    for name, module in model.named_modules():
        if name.replace('.', '_') in layer_name_list:
            print(f"current linear name = {name}")
            layer_name = f"{folder_path}/{name.replace('.', '_')}"
            with open(f"{layer_name}.json", 'r', encoding='utf-8') as f:
                message = json.load(f)

            ocSize = message["ocSize"]
            icSize = message["icSize"]

            # if ocSize != 768 or icSize != 768:
            #     continue

            minibatch = 1
            data_width_len = math.ceil(math.log2(mantissa_bitwidth - message["base_width"] + 1))
            base_width = message["base_width"]

            # print(f"layer_name = {layer_name}, ocSize = {ocSize}, icSize = {icSize}")

            data_width_str = read_txt_to_string(f"{layer_name}_data_width_str.txt")
            weight_data_str = read_txt_to_string(f"{layer_name}_weight_data_str.txt")
            with open(f"{layer_name}_index_data_str.json", 'r', encoding='utf-8') as f:
                index_data = json.load(f)

            index_data_size = len(index_data) * len(index_data[0])
            index_data = np.array(index_data, dtype=np.uint8)

            with open(f"{layer_name}_startAddr.json", 'r', encoding='utf-8') as f:
                startAddr = json.load(f)
            with open(f"{layer_name}_sign.json", 'r', encoding='utf-8') as f:
                sign_data = json.load(f)
            with open(f"{layer_name}_exp.json", 'r', encoding='utf-8') as f:
                exp_data = json.load(f)

            exp_data_size = len(exp_data) * len(exp_data[0])
            exp_data = np.array(exp_data, dtype=np.uint8)

            sign_data_size = len(sign_data) * len(sign_data[0])
            sign_data = np.array(sign_data, dtype=np.bool)

            compressed_data_width = string_to_uint32_array(data_width_str)
            compressed_data_width_size = compressed_data_width.size
            compressed_weight_data = string_to_uint32_array(weight_data_str)

            compressed_weight_data_size = compressed_weight_data.size
            # compressed_index_data = string_to_uint32_array(index_data_str)
            # compressed_index_data_size = compressed_index_data.size

            compressed = {}
            compressed['compressed_data_width'] = compressed_data_width
            compressed['compressed_data_width_size'] = compressed_data_width_size
            compressed['compressed_weight_data'] = compressed_weight_data
            compressed['compressed_weight_data_size'] = compressed_weight_data_size
            compressed['index_data'] = index_data
            compressed['index_data_size'] = index_data_size
            compressed['sign_data'] = sign_data
            compressed['sign_data_size'] = sign_data_size
            compressed['exp_data'] = exp_data
            compressed['exp_data_size'] = exp_data_size
            compressed['startAddr'] = startAddr
            compressed['data_width_len'] = data_width_len
            compressed['base_width'] = base_width

            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr_name = parts[-1]

            new_layer = replace_linear_layer(module, compressed)
            setattr(parent, attr_name, new_layer)

            pass
            # print(name, module)
        # print(name)

    for name, module in model.named_modules():
        print(name, module)

    torch.save(model, "opt-125m_isdc.pth")

    # Fix DEV variable, use defined device
    DEV = device

    for dataset in ['wikitext2', 'ptb', 'c4']:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        opt_eval(model, testloader, DEV, dataset, args.log_wandb)
import sys
import time
import math

import torch
import torch.nn as nn
import json
import cupy as cp

from quant import *
from sparsegpt import *
from modelutils import *
import matplotlib.pyplot as plt
import numpy as np
from isdcLinear import *
import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

try:
    import wandb

    has_wandb = True
except:
    has_wandb = False


def get_opt(model, path=None):
    import torch
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    if path is not None:
        model = OPTForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)  # <-- changed to bfloat16
    else:
        # Load using bfloat16
        model = OPTForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16)
    model.seqlen = model.config.max_position_embeddings
    return model


@torch.no_grad()
def opt_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype  # automatically bfloat16
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        print(f"subset = {subset}")
        gpts = {}
        for name in subset:
            if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
                continue
            gpts[name] = SparseGPT(subset[name])
            if args.wbits < 16:
                gpts[name].quantizer = Quantizer()
                gpts[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')
            sparsity = args.sparsity
            gpts[name].fasterprune(
                sparsity, prunen=args.prunen, prunem=args.prunem, percdamp=args.percdamp, blocksize=args.blocksize
            )
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache


@torch.no_grad()
def opt_eval(model, testenc, dev, dataset: str, log_wandb: bool = False):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype  # automatically bfloat16
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.gmp:
            subset = find_layers(layer)

            print(f"subset = {subset}")

            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * args.sparsity)]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
                       :, (i * model.seqlen):((i + 1) * model.seqlen)
                       ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    if log_wandb:
        wandb.log({f'{dataset}/perplexity': ppl.item()})

    model.config.use_cache = use_cache


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='OPT model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--sparsity', type=float, default=0,
        help='Target sparsity'
    )
    parser.add_argument(
        '--prunen', type=int, default=0,
        help='N for N:M pruning.'
    )
    parser.add_argument(
        '--prunem', type=int, default=0,
        help='M for N:M pruning.'
    )
    parser.add_argument(
        '--blocksize', type=int, default=128,
        help='Blocksize to use for adaptive mask selection.'
    )
    parser.add_argument(
        '--gmp', action='store_true',
        help='Whether to run the GMP baseline.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16,
        help='Whether to quantize as well.'
    )
    parser.add_argument(
        '--minlayer', type=int, default=-1,
        help='Prune all layers with id >= this.'
    )
    parser.add_argument(
        '--maxlayer', type=int, default=1000,
        help='Prune all layers with id < this.'
    )
    parser.add_argument(
        '--prune_only', type=str, default='',
        help='Prune only layers that contain this text.'
    )
    parser.add_argument(
        '--invert', action='store_true',
        help='Invert subset.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Path to saved model.'
    )
    parser.add_argument(
        '--log_wandb', action='store_true',
        help='Whether to log to wandb.'
    )

    print("opt_bfloat16_isdc")

    args = parser.parse_args()

    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    model = get_opt(args.model)
    model.eval()

    print("model.dtype = ", model.dtype)  # will show torch.bfloat16

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    # Fix DEV variable, use defined device
    DEV = device

    if (args.sparsity or args.prunen) and not args.gmp:
        tick = time.time()
        opt_sequential(model, dataloader, DEV)
        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()))
            if 'fc2' in n:
                break
        print(time.time() - tick)

    for dataset in ['wikitext2', 'ptb', 'c4']:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        opt_eval(model, testloader, DEV, dataset, args.log_wandb)

    if args.save:
        model.save_pretrained(args.save)

    print("\nAll tests completed!")

    dtype_ = torch.bfloat16
    mantissa_bitwidth = get_mantissa_bitwidth(dtype_)

    folder_path = "opt1.3b_isdc_compressed_weight_4_8"
    file_names = os.listdir(folder_path)
    layer_name_list = []
    for name in file_names:
        if "_startAddr" in name:
            new_name = name.replace("_startAddr.json", "")
            layer_name_list.append(new_name)
    # print(layer_name_list)

    for name, module in model.named_modules():
        if name.replace('.', '_') in layer_name_list:
            print(f"current linear name = {name}")
            layer_name = f"{folder_path}/{name.replace('.', '_')}"
            with open(f"{layer_name}.json", 'r', encoding='utf-8') as f:
                message = json.load(f)

            ocSize = message["ocSize"]
            icSize = message["icSize"]

            # if ocSize != 768 or icSize != 768:
            #     continue

            minibatch = 1
            data_width_len = math.ceil(math.log2(mantissa_bitwidth - message["base_width"] + 1))
            base_width = message["base_width"]

            # print(f"layer_name = {layer_name}, ocSize = {ocSize}, icSize = {icSize}")

            data_width_str = read_txt_to_string(f"{layer_name}_data_width_str.txt")
            weight_data_str = read_txt_to_string(f"{layer_name}_weight_data_str.txt")
            with open(f"{layer_name}_index_data_str.json", 'r', encoding='utf-8') as f:
                index_data = json.load(f)

            index_data_size = len(index_data) * len(index_data[0])
            index_data = np.array(index_data, dtype=np.uint8)

            with open(f"{layer_name}_startAddr.json", 'r', encoding='utf-8') as f:
                startAddr = json.load(f)
            with open(f"{layer_name}_sign.json", 'r', encoding='utf-8') as f:
                sign_data = json.load(f)
            with open(f"{layer_name}_exp.json", 'r', encoding='utf-8') as f:
                exp_data = json.load(f)

            exp_data_size = len(exp_data) * len(exp_data[0])
            exp_data = np.array(exp_data, dtype=np.uint8)

            sign_data_size = len(sign_data) * len(sign_data[0])
            sign_data = np.array(sign_data, dtype=np.bool)

            compressed_data_width = string_to_uint32_array(data_width_str)
            compressed_data_width_size = compressed_data_width.size
            compressed_weight_data = string_to_uint32_array(weight_data_str)

            compressed_weight_data_size = compressed_weight_data.size
            # compressed_index_data = string_to_uint32_array(index_data_str)
            # compressed_index_data_size = compressed_index_data.size

            compressed = {}
            compressed['compressed_data_width'] = compressed_data_width
            compressed['compressed_data_width_size'] = compressed_data_width_size
            compressed['compressed_weight_data'] = compressed_weight_data
            compressed['compressed_weight_data_size'] = compressed_weight_data_size
            compressed['index_data'] = index_data
            compressed['index_data_size'] = index_data_size
            compressed['sign_data'] = sign_data
            compressed['sign_data_size'] = sign_data_size
            compressed['exp_data'] = exp_data
            compressed['exp_data_size'] = exp_data_size
            compressed['startAddr'] = startAddr
            compressed['data_width_len'] = data_width_len
            compressed['base_width'] = base_width

            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr_name = parts[-1]

            new_layer = replace_linear_layer(module, compressed)
            setattr(parent, attr_name, new_layer)

            pass
            # print(name, module)
        # print(name)

    for name, module in model.named_modules():
        print(name, module)

    torch.save(model, "opt-125m_isdc.pth")

    # Fix DEV variable, use defined device
    DEV = device

    for dataset in ['wikitext2', 'ptb', 'c4']:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        opt_eval(model, testloader, DEV, dataset, args.log_wandb)
