import os
import math
import torch
import numpy as np
import importlib.util
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'turboquant-pytorch'))
from compressors import TurboQuantCompressorV2, TurboQuantCompressorMSE

def load_model(checkpoint_dir, model_file):
    spec = importlib.util.spec_from_file_location("model", model_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    GPT = mod.GPT
    GPTConfig = mod.GPTConfig

    ckpt_path = os.path.join(checkpoint_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location='cuda')
    cfg = checkpoint['model_args']
    config = GPTConfig(**cfg)
    model = GPT(config)
    state_dict = checkpoint['model']
    unwanted = [k for k in state_dict if k.startswith('_orig_mod.')]
    for k in unwanted:
        state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to('cuda')
    return model

def get_val_tokens(data_dir, context_length, num_batches=10):
    val_path = os.path.join(data_dir, 'val.bin')
    data = np.memmap(val_path, dtype=np.uint16, mode='r')
    batches = []
    for i in range(num_batches):
        start = i * context_length
        end = start + context_length + 1
        if end > len(data):
            break
        chunk = torch.from_numpy(data[start:end].astype(np.int64)).unsqueeze(0).to('cuda')
        batches.append(chunk)
    return batches

def measure_standard(model, batches):
    total_loss = 0
    count = 0
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for batch in batches:
            x = batch[:, :-1]
            y = batch[:, 1:]
            logits, loss = model(x, y)
            total_loss += loss.item()
            count += 1
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    perplexity = math.exp(total_loss / count)
    return perplexity, peak_mb

def measure_turboquant(model, batches, bits=3):
    head_dim = model.config.n_embd // model.config.n_head
    k_compressor = TurboQuantCompressorV2(head_dim=head_dim, bits=bits, seed=42, device='cuda')
    v_compressor = TurboQuantCompressorMSE(head_dim=head_dim, bits=bits, seed=43, device='cuda')

    total_loss = 0
    count = 0
    torch.cuda.reset_peak_memory_stats()

    original_forwards = []
    for block in model.transformer.h:
        original_forwards.append(block.attn.forward)

    def make_tq_forward(attn_module):
        def tq_forward(x):
            B, T, C = x.size()
            q, k, v = attn_module.c_attn(x).split(attn_module.n_embd, dim=2)
            nh = attn_module.n_head
            hs = C // nh

            k = k.view(B, T, nh, hs).transpose(1, 2)
            q = q.view(B, T, nh, hs).transpose(1, 2)
            v = v.view(B, T, nh, hs).transpose(1, 2)

            k_compressed = k_compressor.compress(k)
            v_compressed = v_compressor.compress(v)

            scores = k_compressor.asymmetric_attention_scores(q, k_compressed)
            scores = scores / math.sqrt(hs)

            causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))
            attn_weights = torch.nn.functional.softmax(scores, dim=-1).to(x.dtype)

            v_decompressed = v_compressor.decompress(v_compressed).to(x.dtype)
            y = attn_weights @ v_decompressed

            y = y.transpose(1, 2).contiguous().view(B, T, C)
            y = attn_module.resid_dropout(attn_module.c_proj(y))
            return y
        return tq_forward

    for block in model.transformer.h:
        block.attn.forward = make_tq_forward(block.attn)

    with torch.no_grad():
        for batch in batches:
            x = batch[:, :-1]
            y = batch[:, 1:]
            logits, loss = model(x, y)
            total_loss += loss.item()
            count += 1

    for i, block in enumerate(model.transformer.h):
        block.attn.forward = original_forwards[i]

    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    perplexity = math.exp(total_loss / count)
    return perplexity, peak_mb

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data', 'fineweb-edu')
    context_lengths = [128, 256, 512]

    checkpoints = [
        ('Baseline', os.path.join(base_dir, 'out-fineweb'), os.path.join(base_dir, 'model.py')),
        ('BitNet',   os.path.join(base_dir, 'out-bitnet'),  os.path.join(base_dir, 'model_bitnet.py')),
    ]

    print(f"\n{'Model':<10} {'Context':<10} {'Std PPL':<12} {'Std Mem MB':<14} {'TQ PPL':<12} {'TQ Mem MB'}")
    print('-' * 72)

    for model_name, ckpt_dir, model_file in checkpoints:
        model = load_model(ckpt_dir, model_file)
        for ctx in context_lengths:
            batches = get_val_tokens(data_dir, ctx)
            if not batches:
                print(f"{model_name:<10} {ctx:<10} insufficient data")
                continue
            std_ppl, std_mem = measure_standard(model, batches)
            tq_ppl, tq_mem = measure_turboquant(model, batches)
            print(f"{model_name:<10} {ctx:<10} {std_ppl:<12.2f} {std_mem:<14.1f} {tq_ppl:<12.2f} {tq_mem:.1f}")
        del model
        torch.cuda.empty_cache()

    print("\nDone.")