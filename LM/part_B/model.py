import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2LMHeadModel

class LoRACLinear(nn.Module):
    def __init__(self, base_layer, hidden_dim, rank=8, alpha=16):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze the pre-trained base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.lora_A_q = nn.Parameter(torch.zeros(hidden_dim, rank))
        self.lora_B_q = nn.Parameter(torch.zeros(rank, hidden_dim))

        self.lora_A_k = nn.Parameter(torch.zeros(hidden_dim, rank))
        self.lora_B_k = nn.Parameter(torch.zeros(rank, hidden_dim))

        self.lora_A_v = nn.Parameter(torch.zeros(hidden_dim, rank))
        self.lora_B_v = nn.Parameter(torch.zeros(rank, hidden_dim))

        nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_q)
        nn.init.kaiming_uniform_(self.lora_A_k, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_k)
        nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(x)

        dtype = x.dtype
        delta_q = (x @ self.lora_A_q.to(dtype)) @ self.lora_B_q.to(dtype) * self.scaling
        delta_k = (x @ self.lora_A_k.to(dtype)) @ self.lora_B_k.to(dtype) * self.scaling
        delta_v = (x @ self.lora_A_v.to(dtype)) @ self.lora_B_v.to(dtype) * self.scaling

        lora_delta = torch.cat([delta_q, delta_k, delta_v], dim=-1)

        return base_output + lora_delta

class CustomGPT2Attention(GPT2Attention):
    def __init__(self, config, layer_idx=None, rank=8, alpha=16):
        super().__init__(config, layer_idx=layer_idx)
        self.c_attn = LoRACLinear(self.c_attn, config.n_embd, rank=rank, alpha=alpha)


class GPT2_LoRA(GPT2LMHeadModel):
    def __init__(self, config, rank=8, alpha=16):
        super().__init__(config)
        self.rank = rank
        self.alpha = alpha

        for i in range(len(self.transformer.h)):
            old_attn = self.transformer.h[i].attn
            layer_idx = getattr(old_attn, "layer_idx", i)
            
            # Create new attention block
            new_attn = CustomGPT2Attention(config, layer_idx=layer_idx, rank=rank, alpha=alpha)
            
            # Preserve original weights inside base_layer
            new_attn.c_attn.base_layer.weight.data.copy_(old_attn.c_attn.weight.data)
            if old_attn.c_attn.bias is not None:
                new_attn.c_attn.base_layer.bias.data.copy_(old_attn.c_attn.bias.data)
                
            new_attn.c_proj.weight.data.copy_(old_attn.c_proj.weight.data)
            if old_attn.c_proj.bias is not None:
                new_attn.c_proj.bias.data.copy_(old_attn.c_proj.bias.data)

            self.transformer.h[i].attn = new_attn

        self.tie_weights()

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)