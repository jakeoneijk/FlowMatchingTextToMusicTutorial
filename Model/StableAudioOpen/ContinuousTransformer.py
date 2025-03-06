import torch
from torch import nn
import torch.nn.functional as F

from TorchJaekwon.Util import UtilTorch

from Model.StableAudioOpen.TransformerModule import RotaryEmbedding, TransformerBlock

class ContinuousTransformer(nn.Module):
    def __init__(
        self,
        dim = 384,
        depth = 12,
        dim_in = 64,
        dim_out = 64,
        dim_heads = 64,
        cross_attend=False,
        cond_token_dim=None,
        global_cond_dim=None,
        causal=False,
        rotary_pos_emb=True,
        zero_init_branch_outputs=True,
        **kwargs
        ):

        super().__init__()

        self.dim = dim
        self.depth = depth
        self.causal = causal
        self.layers = nn.ModuleList([])

        self.project_in = nn.Linear(dim_in, dim, bias=False) if dim_in is not None else nn.Identity()
        self.project_out = nn.Linear(dim, dim_out, bias=False) if dim_out is not None else nn.Identity()

        if rotary_pos_emb:
            self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32))
        else:
            self.rotary_pos_emb = None

        for i in range(depth):
            self.layers.append(
                TransformerBlock(
                    dim,
                    dim_heads = dim_heads,
                    cross_attend = cross_attend,
                    dim_context = cond_token_dim,
                    global_cond_dim = global_cond_dim,
                    causal = causal,
                    zero_init_branch_outputs = zero_init_branch_outputs,
                    layer_ix=i,
                    **kwargs
                )
            )
        
    def forward(
        self,
        x,
        mask = None,
        prepend_embeds = None,
        prepend_mask = None,
        global_cond = None,
        return_info = False,
        **kwargs
    ):
        batch, seq, device = *x.shape[:2], x.device

        info = {
            "hidden_states": [],
        }

        x = self.project_in(x)

        if prepend_embeds is not None:
            prepend_length, prepend_dim = prepend_embeds.shape[1:]

            assert prepend_dim == x.shape[-1], 'prepend dimension must match sequence dimension'

            x = torch.cat((prepend_embeds, x), dim = -2)

            if prepend_mask is not None or mask is not None:
                mask = mask if mask is not None else torch.ones((batch, seq), device = device, dtype = torch.bool)
                prepend_mask = prepend_mask if prepend_mask is not None else torch.ones((batch, prepend_length), device = device, dtype = torch.bool)

                mask = torch.cat((prepend_mask, mask), dim = -1)

        # Attention layers 

        if self.rotary_pos_emb is not None:
            rotary_pos_emb = self.rotary_pos_emb.forward_from_seq_len(x.shape[1])
        else:
            rotary_pos_emb = None

        # Iterate over the transformer layers
        for layer in self.layers:
            #x = layer(x, rotary_pos_emb = rotary_pos_emb, global_cond=global_cond, **kwargs)
            x = UtilTorch.checkpoint(layer, x, rotary_pos_emb = rotary_pos_emb, global_cond=global_cond, **kwargs)

            if return_info:
                info["hidden_states"].append(x)

        x = self.project_out(x)

        if return_info:
            return x, info
        
        return x