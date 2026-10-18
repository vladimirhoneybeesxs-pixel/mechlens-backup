"""Integration tests for Cohere architecture adapter (CohereForCausalLM).

Model: trl-internal-testing/tiny-CohereForCausalLM
  - 2 layers, ~8M params, CPU-safe, no gating required
  - tie_word_embeddings=True by default
  - logit_scale=0.125 (canonical Command-R is 0.0625; tiny diverges so
    regression tests catch silent-fallback bugs in the passthrough)

NOTE: The Hub tiny model has use_qk_norm=False. Command R+ checkpoints set
use_qk_norm=True, which adds per-head CohereLayerNorm q_norm/k_norm modules
(weight shape (n_heads, head_dim)) inside CohereAttention. The attention bridge
reimplements attention rather than delegating, so that path is NOT covered by
test_forward_matches_hf; TestCohereQKNorm below exercises it with an in-memory
random CohereForCausalLM(use_qk_norm=True) built from CohereConfig, which needs
no Hub access and so cannot be blocked by gating.
"""

import copy
from typing import Any

import pytest
import torch
from transformers import AutoModelForCausalLM, CohereConfig, CohereForCausalLM

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    NormalizationBridge,
    RMSNormalizationBridge,
)
from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
    PositionEmbeddingsAttentionBridge,
)
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)

MODEL = "trl-internal-testing/tiny-CohereForCausalLM"


@pytest.fixture(scope="module")
def cohere_bridge():
    """Load tiny Cohere bridge once per module (no weight processing)."""
    return TransformerBridge.boot_transformers(MODEL, device="cpu")


@pytest.fixture(scope="module")
def cohere_bridge_processed():
    """Bridge with preprocess_weights applied (fold only, no centering).

    process_weights must be called explicitly — boot_transformers does not call
    it automatically. We disable all ProcessWeights options so only the adapter's
    preprocess_weights (logit_scale fold + untie) runs.
    """
    bridge = TransformerBridge.boot_transformers(MODEL, device="cpu")
    bridge.process_weights(
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    )
    return bridge


@pytest.fixture(scope="module")
def cohere_hf() -> Any:
    """Load the raw HF model for side-by-side comparisons."""
    return AutoModelForCausalLM.from_pretrained(MODEL).eval()


# ---------------------------------------------------------------------------
# 1. Bridge creation — exercises all 4 registration points + component_mapping
# ---------------------------------------------------------------------------


class TestCohereBridgeCreation:
    """Verify the bridge loads cleanly and exposes the expected structure."""

    def test_boot_transformers_succeeds(self, cohere_bridge: TransformerBridge) -> None:
        assert cohere_bridge is not None

    def test_block_count(self, cohere_bridge: TransformerBridge) -> None:
        # tiny model has 2 layers
        assert len(cohere_bridge.blocks) == 2

    def test_has_core_components(self, cohere_bridge: TransformerBridge) -> None:
        assert hasattr(cohere_bridge, "embed")
        assert hasattr(cohere_bridge, "unembed")
        assert hasattr(cohere_bridge, "ln_final")
        assert hasattr(cohere_bridge, "rotary_emb")

    def test_no_ln2_in_blocks(self, cohere_bridge: TransformerBridge) -> None:
        # Parallel block: no post_attention_layernorm
        for block in cohere_bridge.blocks:
            assert not hasattr(block, "ln2"), "Parallel block must not have ln2"

    def test_ln1_subtracts_mean_like_hf_layernorm(
        self, cohere_bridge: TransformerBridge, cohere_hf: Any
    ) -> None:
        """uses_rms_norm=False must make ln1 mean-subtract (true LayerNorm), not RMS.

        Drives blocks[0].ln1 on a deliberately non-zero-mean input and compares
        against the raw HF CohereLayerNorm (which subtracts the mean). If
        uses_rms_norm regressed to True, the bridge would skip mean subtraction
        and diverge from HF by O(1), tripping this assertion.
        """
        ln1 = cohere_bridge.blocks[0].ln1
        hf_ln1 = cohere_hf.model.layers[0].input_layernorm
        d_model = cohere_bridge.cfg.d_model
        torch.manual_seed(0)
        # +5.0 shift guarantees a large mean so LN and RMS outputs differ markedly.
        x = torch.randn(1, 4, d_model) + 5.0
        with torch.no_grad():
            bridge_out = ln1(x)
            hf_out = hf_ln1(x)
        ln_diff = (bridge_out - hf_out).abs().max().item()
        assert ln_diff < 1e-4, f"ln1 does not match HF CohereLayerNorm: max_diff={ln_diff:.6f}"
        # And it must NOT match an RMS-only (no mean subtraction) normalization.
        variance = x.float().pow(2).mean(-1, keepdim=True)
        rms_only = (
            x.float()
            * torch.rsqrt(variance + ln1.original_component.variance_epsilon)
            * ln1.original_component.weight.float()
        ).to(bridge_out.dtype)
        rms_diff = (bridge_out - rms_only).abs().max().item()
        assert rms_diff > 1e-2, (
            "ln1 output matches RMS-only normalization; uses_rms_norm must be False "
            f"so the mean is subtracted (rms_diff={rms_diff:.6f})"
        )

    def test_cfg_logit_scale_matches_hf(
        self, cohere_bridge: TransformerBridge, cohere_hf: Any
    ) -> None:
        """Regression: logit_scale must propagate from HF (not silently fall back to 0.0625)."""
        bridge_scale = getattr(cohere_bridge.cfg, "logit_scale")
        assert bridge_scale == cohere_hf.config.logit_scale
        # Anchor 0.125 so a passthrough regression that defaults to 0.0625 also trips here.
        assert bridge_scale == pytest.approx(0.125)

    def test_cfg_rope_parameters_matches_hf(
        self, cohere_bridge: TransformerBridge, cohere_hf: Any
    ) -> None:
        """Regression: rope_parameters must propagate from HF (same passthrough trap as logit_scale)."""
        assert getattr(cohere_bridge.cfg, "rope_parameters") == cohere_hf.config.rope_parameters


# ---------------------------------------------------------------------------
# 2. Forward equivalence — HF logits ≈ bridge logits
# ---------------------------------------------------------------------------


class TestCohereForwardEquivalence:
    """Verify bridge and HF produce identical logits for the same input."""

    def test_forward_returns_logits(self, cohere_bridge: TransformerBridge) -> None:
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            output = cohere_bridge(tokens)
        assert output.shape[0] == 1
        assert output.shape[1] == 4
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_matches_hf(self, cohere_bridge: TransformerBridge, cohere_hf: Any) -> None:
        """Bridge delegates to HF native forward — logits should be identical."""
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            bridge_out = cohere_bridge(tokens)
            hf_out = cohere_hf(tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-4, f"Bridge vs HF max diff = {max_diff:.6f}"


# ---------------------------------------------------------------------------
# 3. Logit scale applied end-to-end
# ---------------------------------------------------------------------------


class TestCohereLogitScaleEndToEnd:
    """Verify logit_scale is correctly folded into the loaded model.

    process_weights must be called before the fold takes effect — boot_transformers
    alone does NOT call process_weights. cohere_bridge_processed uses a fixture that
    calls process_weights with all standard options disabled so only preprocess_weights
    (the logit_scale fold) runs.
    """

    def test_unembed_weight_is_scaled_relative_to_hf(
        self, cohere_bridge_processed: TransformerBridge, cohere_hf: Any
    ) -> None:
        # After preprocess_weights, lm_head.weight inside the bridge should equal
        # HF lm_head.weight * logit_scale (both [d_vocab, d_model]).
        logit_scale = getattr(cohere_bridge_processed.cfg, "logit_scale")
        tl_weight = cohere_bridge_processed.unembed.original_component.weight  # [d_vocab, d_model]
        hf_weight = cohere_hf.lm_head.weight.detach()  # [d_vocab, d_model]
        expected = hf_weight * logit_scale
        max_diff = (tl_weight - expected).abs().max().item()
        assert max_diff < 1e-5, (
            f"unembed.weight not correctly scaled: max_diff={max_diff:.6f}, "
            f"logit_scale={logit_scale}"
        )

    def test_logit_scale_folded_not_applied_twice(
        self, cohere_bridge: TransformerBridge, cohere_hf: Any
    ) -> None:
        """Confirm logit_scale isn't double-applied.

        The bridge (before process_weights) delegates to HF's forward, which includes
        the logit_scale multiply. If forward still matches HF, the fold hasn't been
        applied a second time on top of HF's own scale.
        """
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            bridge_out = cohere_bridge(tokens)
            hf_out = cohere_hf(tokens).logits
        # If logit_scale were applied twice, outputs would differ by a factor of 16
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-4, f"Possible double-application of logit_scale; diff={max_diff:.6f}"


# ---------------------------------------------------------------------------
# 4. Tied embedding preserved — embed.W_E must NOT be scaled
# ---------------------------------------------------------------------------


class TestCohereTiedEmbedding:
    """Verify preprocess_weights does not corrupt embed.W_E in the tied case.

    Both tests use cohere_bridge_processed (process_weights called with fold-only
    options) because the untie + fold only happens inside process_weights.
    """

    def test_embed_weight_equals_hf_embed_tokens(
        self, cohere_bridge_processed: TransformerBridge, cohere_hf: Any
    ) -> None:
        # After the fold, embed.W_E must still equal HF's unscaled embed_tokens.weight.
        # If the fold corrupted embed (in-place on the shared tensor), this fails.
        hf_embed = cohere_hf.model.embed_tokens.weight.detach()  # [d_vocab, d_model]
        tl_embed = cohere_bridge_processed.embed.W_E  # [d_vocab, d_model]
        max_diff = (tl_embed - hf_embed).abs().max().item()
        assert (
            max_diff < 1e-6
        ), f"embed.W_E was corrupted (possibly by logit_scale fold): max_diff={max_diff:.6f}"

    @pytest.mark.parametrize("logit_scale", [0.0625, 1.0])
    def test_embed_and_unembed_weights_differ(self, logit_scale: float) -> None:
        # After the logit_scale fold, embed.W_E and unembed.weight must NOT be
        # identical for a non-trivial scale. logit_scale=1.0 is kept as a regression
        # guard for the no-op case, where the two weights stay tied.
        #
        # cfg.logit_scale is set before process_weights so the fold (which reads it
        # inside preprocess_weights) runs with the parametrized value.
        bridge = TransformerBridge.boot_transformers(MODEL, device="cpu")
        bridge.cfg.logit_scale = logit_scale  # type: ignore[attr-defined]
        bridge.process_weights(
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            fold_value_biases=False,
            refactor_factored_attn_matrices=False,
        )
        tl_embed = bridge.embed.W_E
        tl_unembed = bridge.unembed.original_component.weight
        weights_identical = torch.allclose(tl_embed, tl_unembed)
        if logit_scale == 1.0:
            assert weights_identical, (
                "embed.W_E and unembed.weight should remain tied when logit_scale=1.0 "
                "(the fold is a no-op)"
            )
        else:
            assert not weights_identical, (
                "embed.W_E and unembed.weight are identical — "
                "logit_scale fold may not have been applied or untied correctly"
            )


# ---------------------------------------------------------------------------
# 5. HF delegation — RoPE, attention, normalization go through HF modules
# ---------------------------------------------------------------------------


class TestCohereHFDelegation:
    """Spot-check that bridge submodules delegate to actual HF modules.

    This confirms setup_component_testing wired rotary_emb correctly and that
    NormalizationBridge and PositionEmbeddingsAttentionBridge hold live HF objects.
    """

    def test_ln1_original_component_is_hf_norm(self, cohere_bridge: TransformerBridge) -> None:
        # bridge.blocks[0].ln1.original_component should be the live HF CohereLayerNorm
        ln1 = cohere_bridge.blocks[0].ln1
        assert isinstance(ln1, NormalizationBridge)
        assert ln1.original_component is not None
        # CohereLayerNorm has variance_epsilon (not eps)
        assert hasattr(
            ln1.original_component, "variance_epsilon"
        ), "original_component is not CohereLayerNorm (missing variance_epsilon)"

    def test_attn_original_component_is_hf_attention(
        self, cohere_bridge: TransformerBridge
    ) -> None:
        attn = cohere_bridge.blocks[0].attn
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert attn.original_component is not None
        # CohereAttention has q_proj
        assert hasattr(
            attn.original_component, "q_proj"
        ), "original_component is not CohereAttention (missing q_proj)"

    def test_ln_final_original_component_is_hf_norm(self, cohere_bridge: TransformerBridge) -> None:
        assert cohere_bridge.ln_final.original_component is not None
        assert hasattr(cohere_bridge.ln_final.original_component, "variance_epsilon")


# ---------------------------------------------------------------------------
# 6. Parallel-attn hooks fire correctly
# ---------------------------------------------------------------------------


class TestCohereParallelHooks:
    """Verify hook placement for the parallel attention+MLP block."""

    def test_no_hook_resid_mid(self, cohere_bridge: TransformerBridge) -> None:
        # Parallel block has no intermediate residual stream
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = cohere_bridge.run_with_cache(tokens)
        assert not any("hook_resid_mid" in k for k in cache.keys())

    def test_attn_and_mlp_hooks_fire(self, cohere_bridge: TransformerBridge) -> None:
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = cohere_bridge.run_with_cache(tokens)
        for i in range(2):
            assert f"blocks.{i}.attn.hook_in" in cache
            assert f"blocks.{i}.attn.hook_out" in cache
            assert f"blocks.{i}.mlp.hook_in" in cache
            assert f"blocks.{i}.mlp.hook_out" in cache

    def test_residual_hooks_fire(self, cohere_bridge: TransformerBridge) -> None:
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = cohere_bridge.run_with_cache(tokens)
        for i in range(2):
            assert f"blocks.{i}.hook_resid_pre" in cache
            assert f"blocks.{i}.hook_resid_post" in cache


# ---------------------------------------------------------------------------
# RoPE convention — Cohere pairs adjacent dims (repeat_interleave cos/sin)
# ---------------------------------------------------------------------------


class TestCohereRotaryConvention:
    """The bridge must rotate (2i, 2i+1) pairs to match CohereRotaryEmbedding."""

    def test_cfg_rotary_adjacent_pairs(self, cohere_bridge: TransformerBridge) -> None:
        assert cohere_bridge.cfg.rotary_adjacent_pairs is True

    def test_hf_cos_sin_are_interleaved(self, cohere_hf: Any) -> None:
        # Sanity-check the HF convention the flag encodes: adjacent slots carry
        # the same frequency, unlike Llama's cat(freqs, freqs).
        x = torch.zeros(1, 6, cohere_hf.config.hidden_size)
        position_ids = torch.arange(6).unsqueeze(0)
        cos, sin = cohere_hf.model.rotary_emb(x, position_ids)
        torch.testing.assert_close(cos[..., 0::2], cos[..., 1::2])
        torch.testing.assert_close(sin[..., 0::2], sin[..., 1::2])

    def test_forward_matches_hf_on_long_sequence(
        self, cohere_bridge: TransformerBridge, cohere_hf: Any
    ) -> None:
        # Position 0 is immune to a wrong rotation (sin=0); exercise many positions.
        torch.manual_seed(0)
        tokens = torch.randint(0, cohere_hf.config.vocab_size, (2, 24))
        with torch.no_grad():
            bridge_out = cohere_bridge(tokens)
            hf_out = cohere_hf(tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-4, f"Bridge vs HF max diff = {max_diff:.6f}"


# ---------------------------------------------------------------------------
# QK-norm (use_qk_norm=True) — Command R+ configuration, in-memory model
# ---------------------------------------------------------------------------


def _tiny_qk_norm_config(use_qk_norm: bool = True) -> CohereConfig:
    """Tiny Cohere config in the Command R+ shape: GQA + per-head QK LayerNorm."""
    return CohereConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        use_qk_norm=use_qk_norm,
        logit_scale=0.125,
        layer_norm_eps=1e-5,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )


def _tiny_qk_norm_bridge(
    use_qk_norm: bool = True,
) -> tuple[TransformerBridge, CohereForCausalLM]:
    torch.manual_seed(0)
    cfg = _tiny_qk_norm_config(use_qk_norm)
    hf_reference = CohereForCausalLM(cfg).eval()
    hf_reference.config._attn_implementation = "eager"
    for layer in hf_reference.model.layers:
        layer.self_attn.config._attn_implementation = "eager"
    if use_qk_norm:
        # Randomise the norm weights so an identity-weight pass cannot mask a
        # broadcasting bug or a skipped norm.
        with torch.no_grad():
            for layer in hf_reference.model.layers:
                layer.self_attn.q_norm.weight.copy_(torch.randn_like(layer.self_attn.q_norm.weight))
                layer.self_attn.k_norm.weight.copy_(torch.randn_like(layer.self_attn.k_norm.weight))
    hf_model = CohereForCausalLM(copy.deepcopy(cfg)).eval()
    hf_model.load_state_dict(hf_reference.state_dict())
    bridge = build_bridge_from_module(
        hf_model,
        "CohereForCausalLM",
        hf_config=copy.deepcopy(cfg),
        tokenizer=None,
        device="cpu",
    ).eval()
    bridge.adapter.setup_component_testing(hf_model, bridge_model=bridge)
    return bridge, hf_reference


class TestCohereQKNorm:
    """use_qk_norm=True (Command R+): boot, structure, hooks and HF logit parity."""

    # Sequence length deliberately != n_heads so a post-transpose (H, D) broadcast
    # cannot accidentally line up with (S, D).
    TOKENS = torch.tensor([[3, 17, 42, 5, 99, 8, 64]])

    def test_boot_succeeds_with_qk_norm(self) -> None:
        bridge, _ = _tiny_qk_norm_bridge(use_qk_norm=True)
        assert len(bridge.blocks) == 2

    def test_cfg_use_qk_norm_propagated(self) -> None:
        bridge, hf = _tiny_qk_norm_bridge(use_qk_norm=True)
        assert bridge.cfg.use_qk_norm is True  # type: ignore[attr-defined]
        assert bridge.cfg.use_qk_norm == hf.config.use_qk_norm  # type: ignore[attr-defined]
        bridge_off, hf_off = _tiny_qk_norm_bridge(use_qk_norm=False)
        assert bridge_off.cfg.use_qk_norm is False  # type: ignore[attr-defined]
        assert bridge_off.cfg.use_qk_norm == hf_off.config.use_qk_norm  # type: ignore[attr-defined]

    def test_qk_norm_bridges_wrap_hf_norms(self) -> None:
        bridge, _ = _tiny_qk_norm_bridge(use_qk_norm=True)
        hf_model = bridge.original_model
        for i, block in enumerate(bridge.blocks):
            attn = block.attn
            assert isinstance(attn, PositionEmbeddingsAttentionBridge)
            assert isinstance(attn.q_norm, NormalizationBridge)
            assert not isinstance(attn.q_norm, RMSNormalizationBridge)
            assert attn.q_norm.original_component is hf_model.model.layers[i].self_attn.q_norm
            assert attn.k_norm.original_component is hf_model.model.layers[i].self_attn.k_norm
            # Per-head weights, GQA-aware: q over n_heads, k over n_kv_heads.
            assert tuple(attn.q_norm.original_component.weight.shape) == (4, 8)
            assert tuple(attn.k_norm.original_component.weight.shape) == (2, 8)

    def test_qk_norm_phase_is_pre_transpose(self) -> None:
        bridge, _ = _tiny_qk_norm_bridge(use_qk_norm=True)
        for block in bridge.blocks:
            assert block.attn._qk_norm_phase == "pre_transpose"

    def test_no_qk_norm_declared_when_disabled(self) -> None:
        bridge, _ = _tiny_qk_norm_bridge(use_qk_norm=False)
        for block in bridge.blocks:
            assert "q_norm" not in block.attn.submodules
            assert "k_norm" not in block.attn.submodules
            assert block.attn._qk_norm_phase is None

    def test_forward_matches_hf_with_qk_norm(self) -> None:
        bridge, hf = _tiny_qk_norm_bridge(use_qk_norm=True)
        with torch.no_grad():
            bridge_out = bridge(self.TOKENS)
            hf_out = hf(self.TOKENS).logits
        assert bridge_out.shape == hf_out.shape == (1, 7, 128)
        assert torch.isfinite(bridge_out).all()
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"

    def test_forward_matches_hf_without_qk_norm_control(self) -> None:
        # Control: same in-memory harness with QK-norm off still matches HF.
        bridge, hf = _tiny_qk_norm_bridge(use_qk_norm=False)
        with torch.no_grad():
            bridge_out = bridge(self.TOKENS)
            hf_out = hf(self.TOKENS).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"

    def test_qk_norm_hooks_fire_with_hf_shapes(self) -> None:
        bridge, _ = _tiny_qk_norm_bridge(use_qk_norm=True)
        _, cache = bridge.run_with_cache(self.TOKENS)
        q_normed = cache["blocks.0.attn.hook_q_normed"]
        k_normed = cache["blocks.0.attn.hook_k_normed"]
        # Norm runs on the pre-transpose [batch, pos, head, d_head] view, as in HF.
        assert tuple(q_normed.shape) == (1, 7, 4, 8)
        assert tuple(k_normed.shape) == (1, 7, 2, 8)
        # Each head is mean-centred over d_head (true LayerNorm, not RMSNorm)
        # before the per-head weight is applied; check via the inner norm hook.
        inner = cache["blocks.0.attn.q_norm.hook_normalized"]
        assert tuple(inner.shape) == (1, 7, 4, 8)
        torch.testing.assert_close(
            inner.mean(-1), torch.zeros_like(inner.mean(-1)), atol=1e-5, rtol=0
        )

    def test_qk_norm_changes_output(self) -> None:
        # Guard against a silently skipped norm: perturbing q_norm.weight must move logits.
        bridge, _ = _tiny_qk_norm_bridge(use_qk_norm=True)
        with torch.no_grad():
            base = bridge(self.TOKENS).clone()
            bridge.blocks[0].attn.q_norm.original_component.weight.mul_(3.0)
            perturbed = bridge(self.TOKENS)
        assert not torch.allclose(base, perturbed)

    def test_process_weights_preserves_parity_with_qk_norm(self) -> None:
        # Compatibility-mode weight processing must round-trip q_norm/k_norm weights
        # unchanged (no offset / reshape conversions) and keep the logit_scale fold.
        bridge, hf = _tiny_qk_norm_bridge(use_qk_norm=True)
        bridge.process_weights(
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            fold_value_biases=False,
            refactor_factored_attn_matrices=False,
        )
        with torch.no_grad():
            bridge_out = bridge(self.TOKENS)
            hf_out = hf(self.TOKENS).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-4, f"Processed bridge vs HF max diff = {max_diff}"
