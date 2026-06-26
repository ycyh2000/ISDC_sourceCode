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
from originalCPULinear import *
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



import gc

def cleanup_model(model, dev):
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.reset_peak_memory_stats(dev)
    torch.cuda.synchronize()


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


@torch.no_grad()
def benchmark_model(model, input_ids, dev, warmup=10, iters=50):
    model.eval()
    model.config.use_cache = False

    # Move normal modules to GPU first.
    # CPUOffloadLinear.weight_cpu will NOT move if it is a plain tensor attribute.
    model = model.to(dev)

    input_ids = input_ids.to(dev)

    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    tokens_per_iter = batch_size * seq_len

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)

    for _ in range(warmup):
        _ = model(input_ids)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(dev)

    start = time.perf_counter()
    for _ in range(iters):
        _ = model(input_ids)
    torch.cuda.synchronize()
    end = time.perf_counter()

    total_time = end - start
    latency = total_time / iters
    throughput = tokens_per_iter * iters / total_time
    peak_memory = torch.cuda.max_memory_allocated(dev) / 1024 / 1024

    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Latency: {latency * 1000:.3f} ms/iter")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print(f"Peak memory: {peak_memory:.2f} MB")


if __name__ == '__main__':
    import argparse
    import copy
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
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--percdamp', type=float, default=.01)
    parser.add_argument('--sparsity', type=float, default=0)
    parser.add_argument('--prunen', type=int, default=0)
    parser.add_argument('--prunem', type=int, default=0)
    parser.add_argument('--blocksize', type=int, default=128)
    parser.add_argument('--gmp', action='store_true')
    parser.add_argument('--wbits', type=int, default=16)
    parser.add_argument('--minlayer', type=int, default=-1)
    parser.add_argument('--maxlayer', type=int, default=1000)
    parser.add_argument('--prune_only', type=str, default='')
    parser.add_argument('--invert', action='store_true')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--log_wandb', action='store_true')

    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run latency, throughput, and peak memory benchmark.'
    )
    parser.add_argument(
        '--eval_ppl',
        action='store_true',
        help='Run perplexity evaluation after layer replacement.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for benchmark input.'
    )
    parser.add_argument(
        '--seq_len',
        type=int,
        default=2048,
        help='Sequence length for benchmark input.'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=10,
        help='Warmup iterations for benchmark.'
    )
    parser.add_argument(
        '--iters',
        type=int,
        default=50,
        help='Measured iterations for benchmark.'
    )
    parser.add_argument(
        '--compressed_folder',
        type=str,
        default='opt1.3b_isdc_compressed_weight_4_8',
        help='Folder containing ISDC compressed weight files.'
    )

    print("opt_bfloat16_isdc")

    args = parser.parse_args()

    if args.log_wandb:
        assert has_wandb, "wandb not installed, try `pip install wandb`"
        wandb.init(config=args)

    DEV = device

    model = get_opt(args.model)
    model.eval()

    print("model.dtype = ", model.dtype)

    dataloader, testloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen
    )

    if (args.sparsity or args.prunen) and not args.gmp:
        tick = time.time()
        opt_sequential(model, dataloader, DEV)

        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()))
            if 'fc2' in n:
                break

        print("Pruning time:", time.time() - tick)

    folder_path = args.compressed_folder
    file_names = os.listdir(folder_path)

    layer_name_list = []
    for file_name in file_names:
        if "_startAddr" in file_name:
            new_name = file_name.replace("_startAddr.json", "")
            layer_name_list.append(new_name)

    print(f"Number of selected layers: {len(layer_name_list)}")

    model_original_cpu = copy.deepcopy(model)
    model_isdc = copy.deepcopy(model)

    print("\nReplacing selected Linear layers with CPUOffloadLinear...")

    replaced_original_count = 0

    for name, module in list(model_original_cpu.named_modules()):
        if name.replace('.', '_') in layer_name_list:
            if not isinstance(module, nn.Linear):
                continue

            print(f"Original CPU-offload layer: {name}")

            parts = name.split('.')
            parent = model_original_cpu
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr_name = parts[-1]

            new_layer = replace_linear_layer_cpu(module)
            setattr(parent, attr_name, new_layer)

            replaced_original_count += 1

    print(f"Total CPUOffloadLinear layers: {replaced_original_count}")

    print("\nReplacing selected Linear layers with ISDCLinear...")

    dtype_ = torch.bfloat16
    mantissa_bitwidth = get_mantissa_bitwidth(dtype_)

    replaced_isdc_count = 0

    for name, module in list(model_isdc.named_modules()):
        if name.replace('.', '_') in layer_name_list:
            if not isinstance(module, nn.Linear):
                continue

            print(f"ISDC layer: {name}")

            layer_name = f"{folder_path}/{name.replace('.', '_')}"

            with open(f"{layer_name}.json", 'r', encoding='utf-8') as f:
                message = json.load(f)

            data_width_len = math.ceil(
                math.log2(mantissa_bitwidth - message["base_width"] + 1)
            )
            base_width = message["base_width"]

            data_width_str = read_txt_to_string(
                f"{layer_name}_data_width_str.txt"
            )
            weight_data_str = read_txt_to_string(
                f"{layer_name}_weight_data_str.txt"
            )

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
            sign_data = np.array(sign_data, dtype=np.bool_)

            compressed_data_width = string_to_uint32_array(data_width_str)
            compressed_data_width_size = compressed_data_width.size

            compressed_weight_data = string_to_uint32_array(weight_data_str)
            compressed_weight_data_size = compressed_weight_data.size

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
            parent = model_isdc
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr_name = parts[-1]

            new_layer = replace_linear_layer_DEVICE(module, compressed)
            setattr(parent, attr_name, new_layer)

            replaced_isdc_count += 1

    print(f"Total ISDCLinear layers: {replaced_isdc_count}")

    if args.benchmark:
        print("\n===== Running Benchmark =====")

        vocab_size = model.config.vocab_size

        input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(args.batch_size, args.seq_len),
            dtype=torch.long
        )

        print("\n--- ISDC Benchmark ---")
        benchmark_model(
            model_isdc,
            input_ids,
            DEV,
            warmup=args.warmup,
            iters=args.iters
        )
        cleanup_model(model_isdc, DEV)


        print("\n--- Original CPU-Offload Benchmark ---")
        benchmark_model(
            model_original_cpu,
            input_ids,
            DEV,
            warmup=args.warmup,
            iters=args.iters
        )
        cleanup_model(model_original_cpu, DEV)


        print("\n--- Original GPU Benchmark ---")
        benchmark_model(
            model,
            input_ids,
            DEV,
            warmup=args.warmup,
            iters=args.iters
        )
        cleanup_model(model, DEV)


    if args.eval_ppl:
        print("\n===== Running Perplexity Evaluation =====")

        for dataset in ['wikitext2', 'ptb', 'c4']:
            dataloader, testloader = get_loaders(
                dataset,
                seed=args.seed,
                model=args.model,
                seqlen=model.seqlen
            )

            print(f"\n--- Original CPU-Offload PPL: {dataset} ---")
            opt_eval(model_original_cpu, testloader, DEV, dataset, args.log_wandb)

            print(f"\n--- ISDC PPL: {dataset} ---")
            opt_eval(model_isdc, testloader, DEV, dataset, args.log_wandb)

    if args.save:
        torch.save(model_original_cpu, args.save + "_original_cpu.pth")
        torch.save(model_isdc, args.save + "_isdc.pth")

    print("\nAll tests completed!")
