"""Measured optimizations for the resident BF16 FLUX.2 Klein service."""
from collections import OrderedDict
from contextlib import contextmanager
import types
import torch


class TensorLRU:
    """Single GPU-worker-owned host cache; values never retain device allocations."""
    def __init__(self, limit_bytes):
        self.entries = OrderedDict()
        self.limit_bytes = int(limit_bytes)
        self.bytes = 0

    def get(self, key):
        value = self.entries.pop(key, None)
        if value is not None:
            self.entries[key] = value
        return value

    def put(self, key, value):
        host = value.detach().to("cpu").contiguous().pin_memory()
        size = host.numel() * host.element_size()
        if size > self.limit_bytes:
            return
        old = self.entries.pop(key, None)
        if old is not None:
            self.bytes -= old.numel() * old.element_size()
        while self.entries and self.bytes + size > self.limit_bytes:
            _, old = self.entries.popitem(last=False)
            self.bytes -= old.numel() * old.element_size()
        self.entries[key] = host
        self.bytes += size

    def clear(self):
        self.entries.clear()
        self.bytes = 0


class _FeaturesComplete(Exception):
    pass


@torch.inference_mode()
def qwen_features(text_encoder, tokenizer, prompt, dtype=None, device=None,
                  max_sequence_length=512, hidden_states_layers=(9, 18, 27)):
    """Return exactly the three original hidden states, without later layers/logits.

    Temporary hooks stop the *base* model after its last required layer. No weight,
    attention mask, padding, sequence length, or feature layer is changed.
    """
    device = device or text_encoder.device
    dtype = dtype or text_encoder.dtype
    prompts = [prompt] if isinstance(prompt, str) else prompt
    ids, masks = [], []
    for text in prompts:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        tokens = tokenizer(rendered, return_tensors="pt", padding="max_length",
                           truncation=True, max_length=max_sequence_length)
        ids.append(tokens["input_ids"])
        masks.append(tokens["attention_mask"])
    selected, handles = {}, []
    last = max(hidden_states_layers)
    base = text_encoder.model

    def capture(number):
        def hook(module, inputs, output):
            selected[number] = output[0] if isinstance(output, tuple) else output
            if number == last:
                raise _FeaturesComplete()
        return hook
    try:
        for number in hidden_states_layers:
            handles.append(base.layers[number - 1].register_forward_hook(capture(number)))
        try:
            base(input_ids=torch.cat(ids).to(device),
                 attention_mask=torch.cat(masks).to(device),
                 output_hidden_states=False, use_cache=False)
        except _FeaturesComplete:
            pass
        if len(selected) != len(hidden_states_layers):
            raise RuntimeError("Qwen feature extraction did not reach all requested layers")
        result = torch.stack([selected[k] for k in hidden_states_layers], dim=1)
        b, channels, length, width = result.shape
        return result.permute(0, 2, 1, 3).reshape(b, length, channels * width).to(dtype=dtype)
    finally:
        for handle in handles:
            handle.remove()


def install_feature_encoder(pipe):
    pipe._get_qwen3_prompt_embeds = qwen_features


@torch.inference_mode()
def park_unused_text_weights(pipe):
    """Move mathematically unused modules once; no per-request offloading occurs."""
    modules=list(pipe.text_encoder.model.layers[27:])
    head=pipe.text_encoder.lm_head
    # Do not move a tied input/output embedding that is still used by the model.
    if head.weight.data_ptr()!=pipe.text_encoder.model.embed_tokens.weight.data_ptr():
        modules.append(head)
    seen=set()
    released=0
    for module in modules:
        for parameter in module.parameters():
            if parameter.device.type=='cuda' and parameter.data_ptr() not in seen:
                released+=parameter.numel()*parameter.element_size()
                seen.add(parameter.data_ptr())
        module.to('cpu')
    return released


@torch.inference_mode()
def fuse_qkv_without_duplicate_weights(transformer):
    """Fuse only supported QKV projections; original projections share storage."""
    count = 0
    for block in transformer.transformer_blocks:
        attention = block.attn
        attention.fuse_projections()
        if not getattr(attention, "fused_projections", False):
            continue
        for names, fused_name in [
            (("to_q", "to_k", "to_v"), "to_qkv"),
            (("add_q_proj", "add_k_proj", "add_v_proj"), "to_added_qkv"),
        ]:
            fused = getattr(attention, fused_name, None)
            if fused is None:
                continue
            offset = 0
            for name in names:
                original = getattr(attention, name)
                width = original.out_features
                original.weight = torch.nn.Parameter(fused.weight[offset:offset+width], requires_grad=False)
                if original.bias is not None:
                    original.bias = torch.nn.Parameter(fused.bias[offset:offset+width], requires_grad=False)
                offset += width
        count += 1
    return count


class DiffusionController:
    """Per-batch exact invariant memoization and explicit approximate Taylor reuse."""
    def __init__(self, transformer):
        self.transformer = transformer
        self.active = None
        self.approximate_blocks = []
        # Keep Python decisions outside the compiled repeated blocks.
        original_blocks = list(transformer.single_transformer_blocks)
        for index, block in enumerate(original_blocks):
            transformer.single_transformer_blocks[index] = _TaylorBlock(block, self, index)

    @contextmanager
    def context(self, mode, steps, interval=2, warmup=2, blocks=12):
        state = {"mode": mode, "step": 0, "steps": steps, "interval": interval,
                 "warmup": warmup, "blocks": blocks, "memo_hits": 0,
                 "predicted_blocks": 0, "history": {}}
        previous = self.active
        self.active = state
        patched = []
        try:
            if mode != "none":
                for name in ("context_embedder", "pos_embed"):
                    module = getattr(self.transformer, name)
                    original = module.forward
                    memo = {}
                    def cached(value, *args, _fn=original, _memo=memo, **kwargs):
                        if args or kwargs:
                            return _fn(value, *args, **kwargs)
                        key = (value.data_ptr(), tuple(value.shape), tuple(value.stride()),
                               value.dtype, value.device, value._version if not value.is_inference() else 0)
                        if key in _memo:
                            state["memo_hits"] += 1
                            return _memo[key][1]
                        result = _fn(value)
                        _memo[key] = (value, result)
                        return result
                    patched.append((module, original))
                    module.forward = cached
            yield state
        finally:
            for module, original in patched:
                module.forward = original
            state["history"].clear()
            self.active = previous

    def on_step_end(self, pipe, step, timestep, kwargs):
        if self.active is not None:
            self.active["step"] = step + 1
        return kwargs


class _TaylorBlock(torch.nn.Module):
    """Predict a late block's residual branch; normal calls keep regional compilation."""
    def __init__(self, block, controller, index):
        super().__init__()
        self.block = block
        # Avoid registering the controller/whole transformer as a child module.
        object.__setattr__(self, "controller", controller)
        self.index = index

    def forward(self, hidden_states, *args, **kwargs):
        state = self.controller.active
        total = len(self.controller.transformer.single_transformer_blocks)
        if state is None or state["mode"] != "taylorseer" or self.index < total - state["blocks"]:
            return self.block(hidden_states, *args, **kwargs)
        step = state["step"]
        history = state["history"].get(self.index)
        predict = (history is not None and step >= state["warmup"]
                   and step < state["steps"] - 1
                   and (step - state["warmup"]) % state["interval"] != state["interval"] - 1)
        if predict:
            last_step, value, derivative = history
            state["predicted_blocks"] += 1
            return hidden_states + (value + (step-last_step)*derivative).to(hidden_states.dtype)
        result = self.block(hidden_states, *args, **kwargs)
        # Residual-branch extrapolation follows the changing current input rather
        # than extrapolating the entire hidden state across denoising timesteps.
        value = (result.float() - hidden_states.float()).to(hidden_states.dtype)
        derivative = (torch.zeros_like(value) if history is None else
                      ((value.float() - history[1].float()) / (step-history[0])).to(hidden_states.dtype))
        state["history"][self.index] = (step, value, derivative)
        return result



class TransformerGraph:
    """Optional single-shape CUDA Graph with a regional fallback for all other work.

    Tensor inputs are copied on every replay, including prompts and position IDs.
    Approximate caching always takes the ordinary regional path.
    """
    def __init__(self, transformer, controller, kwargs):
        self.transformer = transformer
        self.controller = controller
        self.original_forward = transformer.forward
        self.inputs = {key:value.clone() if isinstance(value,torch.Tensor) else value
                       for key,value in kwargs.items()}
        self.graph = None
        self.output = None
        self.hits = 0

    def raw(self):
        return self.original_forward(**self.inputs)[0]

    @torch.inference_mode()
    def capture(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.raw()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self.raw()
        self.graph = graph
        self.output = output

    def install(self):
        self.transformer.forward = self.forward

    def uninstall(self):
        self.transformer.forward = self.original_forward

    def matches(self,args,kwargs):
        state = self.controller.active
        if args or (state is not None and state["mode"]=="taylorseer"):
            return False
        if set(kwargs)!=set(self.inputs):
            return False
        for key,value in kwargs.items():
            static = self.inputs[key]
            if isinstance(static,torch.Tensor):
                if (not isinstance(value,torch.Tensor) or value.shape!=static.shape
                    or value.dtype!=static.dtype or value.device!=static.device):
                    return False
            elif value!=static:
                return False
        return True

    def forward(self,*args,**kwargs):
        if not self.matches(args,kwargs):
            return self.original_forward(*args,**kwargs)
        for key,value in kwargs.items():
            if isinstance(value,torch.Tensor):
                self.inputs[key].copy_(value)
        self.graph.replay()
        self.hits += 1
        return (self.output,)


