import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers import AutoTokenizer, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

class CustomGPT2Attention(GPT2Attention):
    def __init__(self, config, rank, alpha):
        super().__init__(config)
        
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # In GPT-2 di HuggingFace, self.c_attn proietta l'hidden_state in 3 * embed_dim (ovvero Q, K, V concatenate)
        embed_dim = config.hidden_size
        
        self.lora_A_q = nn.Parameter(torch.zeros(embed_dim, rank))
        self.lora_A_k = nn.Parameter(torch.zeros(embed_dim, rank))
        self.lora_A_v = nn.Parameter(torch.zeros(embed_dim, rank))

        # Matrici B: da rank -> embed_dim (Inizializzate a ZERO così all'inizio ΔW = 0)
        self.lora_B_q = nn.Parameter(torch.zeros(rank, embed_dim))
        self.lora_B_k = nn.Parameter(torch.zeros(rank, embed_dim))
        self.lora_B_v = nn.Parameter(torch.zeros(rank, embed_dim))

        # Inizializzazione dei pesi LoRA
        nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_k, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))
        
        nn.init.zeros_(self.lora_B_q)
        nn.init.zeros_(self.lora_B_k)
        nn.init.zeros_(self.lora_B_v)

     def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class GPT2_LoRA(GPT2LMHeadModel):
    def __init__(self, *model_args, rank, alpha, **model_kwargs):
        super().__init__(*model_args, **model_kwargs)
        
        for param in self.parameters():
            param.requires_grad = False
            
        for block in self.transformer.h:
            # substitute block.attn with a new instance of CustomGPT2Attention
            new_attn = CustomGPT2Attention(config, rank=rank, alpha=alpha)
            new_attn.load_state_dict(block.attn.state_dict(), strict=False)
            # keep the weights from block.attn and apply them to the new instance using .load_state_dict()
            block.attn = new_attn
            pass
        for name, param in self.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
    
    def forward(self, *args, **kwargs):
        return super().forward(*args,**kwargs)