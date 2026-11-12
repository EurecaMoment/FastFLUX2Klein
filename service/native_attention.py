"""Use native PyTorch SDPA without version/backend dispatch during compilation.

This service uses one GPU, no attention masks or GQA in the Flux2 transformer.
The tensor layout and calculation match Diffusers' native attention backend.
"""
import torch
import diffusers.models.transformers.transformer_flux2 as flux2

def native_attention(query,key,value,attn_mask=None,dropout_p=0.0,is_causal=False,scale=None,enable_gqa=False,attention_kwargs=None,backend=None,parallel_config=None):
    if parallel_config is not None or attention_kwargs or backend is not None:
        raise ValueError("This optimized path supports native single-GPU attention only")
    if attn_mask is not None and attn_mask.ndim==2:
        attn_mask=attn_mask[:,None,None,:]
    result=torch.nn.functional.scaled_dot_product_attention(query.transpose(1,2),key.transpose(1,2),value.transpose(1,2),attn_mask=attn_mask,dropout_p=dropout_p,is_causal=is_causal,scale=scale,enable_gqa=enable_gqa)
    return result.transpose(1,2)

def install():
    flux2.dispatch_attention_fn=native_attention
