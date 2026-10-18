# Special Cases

```{warning}
`HookedTransformer` is deprecated as of TransformerLens 3.0 and will be removed in the next major version. New code should use [`TransformerBridge`](migrating_to_v3.md) instead. Existing `HookedTransformer` code continues to work through the 3.x branch via a compatibility layer. See the [migration guide](migrating_to_v3.md) for conversion recipes.
```

## Mixture of Experts error rates
Due to the Top-K gating performed in the hidden layer of Mixture of Experts models, small errors can be amplified 
greatly in cases where a different expert is selected, which leads to a higher than normal variance in the error rate
of the final logits. In testing done on Mixtral running in half precision, the standard deviation of the absolute error 
rate of the logits compared to those from the default model was found to be around 2e-3.

There are two main ways to mitigate this:

1. **Skip weight preprocessing.** On the bridge, simply load with `TransformerBridge.boot_transformers(...)` and do not call `enable_compatibility_mode()` - the bridge preserves raw HF weights by default, so no additional flag is needed. On the legacy `HookedTransformer` path, use `HookedTransformer.from_pretrained_no_processing` instead of `HookedTransformer.from_pretrained`.
2. **Increase the precision of the data type used in the model.**

## Qwen3.5 text-only models

Qwen3.5 has no `HookedTransformer.from_pretrained` legacy path. Load it through `TransformerBridge`. Dense text-only checkpoints can be loaded with:

```python
from transformer_lens.model_bridge import TransformerBridge

bridge = TransformerBridge.boot_transformers("Qwen/Qwen3.5-0.8B")
```

Qwen3.5 uses a hybrid stack. Full-attention layers expose the usual hooks under
`blocks.N.attn.*`; linear-attention layers expose GatedDeltaNet hooks under
`blocks.N.linear_attn.*`, including `hook_q_pre_conv`, `hook_k_pre_conv`,
`hook_v_pre_conv`, `hook_beta`, `hook_log_decay`, `hook_recurrence_out`, and
`hook_out`. Full multimodal `Qwen3_5ForConditionalGeneration`, image/video
inputs, and Qwen3.5 MoE checkpoints are not supported by this adapter.

## Cohere Command-R family

Command-R has no `HookedTransformer.from_pretrained` legacy path. Load it through
`TransformerBridge`:

```python
from transformer_lens.model_bridge import TransformerBridge

bridge = TransformerBridge.boot_transformers("CohereLabs/c4ai-command-r-v01")
```

**Scope.** The adapter targets `CohereForCausalLM`, which covers Command-R
(`c4ai-command-r-v01`, `c4ai-command-r-08-2024`), Command-R+
(`c4ai-command-r-plus`, `c4ai-command-r-plus-08-2024`) and the Aya 23 / Aya Expanse
checkpoints. Command R7B (`c4ai-command-r7b-12-2024`), Command A
(`c4ai-command-a-03-2025`) and the `tiny-aya-*` models are `Cohere2ForCausalLM`, and
`North-Mini-Code` is `Cohere2MoeForCausalLM`; neither architecture has an adapter yet,
so `boot_transformers` raises "unsupported architecture" for them.

**Gated weights.** Every `CohereLabs/*` checkpoint is gated on the HuggingFace Hub. Accept
the licence on the model page and export `HF_TOKEN` before loading; without it the config
download fails and the compatibility matrix lists the model as "Config unavailable".

**Numerics to be aware of.**

- *Logit scale.* `CohereForCausalLM` multiplies its logits by `config.logit_scale`
  (`1/16` for Command-R, larger for Command-R+). The raw bridge applies it through HF's
  forward; `enable_compatibility_mode()` folds it into `unembed.weight` first, so
  `bridge.unembed.weight` is scaled relative to `lm_head.weight` while the tied
  `embed.weight` is left untouched.
- *Tokenizer BOS.* `tokenizer_config.json` declares `add_bos_token=False`, but the
  tokenizer's `__call__` prepends `<BOS_TOKEN>` with `add_special_tokens=True`. The
  adapter therefore sets `default_prepend_bos=True`; pass `prepend_bos=False` to
  `to_tokens` only when tokenizing a fragment of a prompt.
- *Interleaved RoPE.* Cohere rotates adjacent dimension pairs `(2i, 2i+1)` rather than
  Llama's `(i, i + d_head/2)`; the adapter sets `cfg.rotary_adjacent_pairs=True` so the
  bridge's attention path matches HF. `hook_rot_q` / `hook_rot_k` fire after this rotation.
- *QK-norm (Command-R+).* Checkpoints with `use_qk_norm=True` carry a per-head
  `CohereLayerNorm` on Q and K (weight shape `[n_heads, d_head]`), applied before the head
  transpose. The bridge exposes it as `blocks.N.attn.q_norm` / `k_norm`, with
  `hook_q_normed` / `hook_k_normed` on the `[batch, pos, head, d_head]` view. Command-R
  and Aya checkpoints have `use_qk_norm=False` and no such submodules.
- *Parallel blocks.* Attention and the gated MLP read the same `ln1` output and there is no
  `ln2`; `hook_resid_mid` is not defined for this architecture.
