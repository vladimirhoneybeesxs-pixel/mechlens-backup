"""Unit tests for QK-norm phase dispatch in PositionEmbeddingsAttentionBridge.

HF architectures place the Q/K norm at three different points relative to the
head reshape, and the bridge reimplements attention (no HF delegation), so it
must pick the phase from the norm weight's shape:

- ``(head_dim,)`` or non-learnable  -> ``post_reshape`` (Gemma-3 / Qwen-3 style,
  norm over head_dim on the transposed ``[B, H, S, D]`` tensor)
- ``(n_heads * head_dim,)``         -> ``pre_reshape`` (OLMo-2 style, RMS over
  the flat projection output)
- ``(n_heads, head_dim)``           -> ``pre_transpose`` (Cohere / Command R+
  style: CohereLayerNorm applied to the ``[B, S, H, D]`` view *before* the
  transpose, so the per-head weight aligns with the trailing ``(H, D)`` dims)

The Cohere case is the regression guard: applying a ``(H, D)`` weight after the
transpose would broadcast against ``(S, D)`` and either raise or silently
produce wrong logits.
"""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from transformer_lens.model_bridge.generalized_components import (
    LinearBridge,
    NormalizationBridge,
    RMSNormalizationBridge,
)
from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
    PositionEmbeddingsAttentionBridge,
)

N_HEADS = 4
N_KV_HEADS = 2
HEAD_DIM = 8
D_MODEL = N_HEADS * HEAD_DIM


class _PerHeadLayerNorm(nn.Module):
    """Stand-in for CohereLayerNorm: mean-subtracting LN with a (heads, head_dim) weight."""

    def __init__(self, hidden_size: tuple[int, ...], eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = (x - mean).pow(2).mean(-1, keepdim=True)
        return self.weight * (x - mean) * torch.rsqrt(var + self.variance_epsilon)


class _FakeRMSNorm(nn.Module):
    """Minimal RMSNorm stand-in with a 1-D weight of the given size."""

    def __init__(self, size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.variance_epsilon = 1e-5


class _FakeAttention(nn.Module):
    """Minimal HF-attention stand-in exposing what the bridge introspects."""

    def __init__(self, q_norm: nn.Module | None, k_norm: nn.Module | None) -> None:
        super().__init__()
        self.head_dim = HEAD_DIM
        self.num_key_value_groups = N_HEADS // N_KV_HEADS
        self.q_proj = nn.Linear(D_MODEL, N_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(D_MODEL, N_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(D_MODEL, N_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(N_HEADS * HEAD_DIM, D_MODEL, bias=False)
        if q_norm is not None:
            self.q_norm = q_norm
        if k_norm is not None:
            self.k_norm = k_norm


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        n_heads=N_HEADS,
        n_key_value_heads=N_KV_HEADS,
        d_model=D_MODEL,
        d_head=HEAD_DIM,
        eps=1e-5,
        uses_rms_norm=False,
        positional_embedding_type="rotary",
    )


def _bridge(
    declare_norms: bool, norm_cls: type = NormalizationBridge
) -> PositionEmbeddingsAttentionBridge:
    cfg = _cfg()
    submodules = {
        "q": LinearBridge(name="q_proj"),
        "k": LinearBridge(name="k_proj"),
        "v": LinearBridge(name="v_proj"),
        "o": LinearBridge(name="o_proj"),
    }
    if declare_norms:
        submodules["q_norm"] = norm_cls(name="q_norm", config=cfg)
        submodules["k_norm"] = norm_cls(name="k_norm", config=cfg)
    return PositionEmbeddingsAttentionBridge(name="self_attn", config=cfg, submodules=submodules)


class TestQKNormPhaseDispatch:
    def test_no_norm_declared_and_none_on_module_is_none(self) -> None:
        bridge = _bridge(declare_norms=False)
        bridge.set_original_component(_FakeAttention(None, None))
        assert bridge._qk_norm_phase is None

    def test_cohere_per_head_weight_is_pre_transpose(self) -> None:
        """(n_heads, head_dim) weight -> norm must run on the [B, S, H, D] view."""
        bridge = _bridge(declare_norms=True)
        bridge.set_original_component(
            _FakeAttention(
                _PerHeadLayerNorm((N_HEADS, HEAD_DIM)),
                _PerHeadLayerNorm((N_KV_HEADS, HEAD_DIM)),
            )
        )
        assert bridge._qk_norm_phase == "pre_transpose"

    def test_head_dim_weight_is_post_reshape(self) -> None:
        bridge = _bridge(declare_norms=True, norm_cls=RMSNormalizationBridge)
        bridge.set_original_component(
            _FakeAttention(_FakeRMSNorm(HEAD_DIM), _FakeRMSNorm(HEAD_DIM))
        )
        assert bridge._qk_norm_phase == "post_reshape"

    def test_flat_projection_weight_is_pre_reshape(self) -> None:
        bridge = _bridge(declare_norms=True, norm_cls=RMSNormalizationBridge)
        bridge.set_original_component(
            _FakeAttention(_FakeRMSNorm(N_HEADS * HEAD_DIM), _FakeRMSNorm(N_KV_HEADS * HEAD_DIM))
        )
        assert bridge._qk_norm_phase == "pre_reshape"

    def test_hf_has_norm_but_adapter_omits_it_raises(self) -> None:
        """The failure mode that made every use_qk_norm=True Cohere checkpoint unbootable."""
        bridge = _bridge(declare_norms=False)
        with pytest.raises(RuntimeError, match="q_norm"):
            bridge.set_original_component(
                _FakeAttention(
                    _PerHeadLayerNorm((N_HEADS, HEAD_DIM)),
                    _PerHeadLayerNorm((N_KV_HEADS, HEAD_DIM)),
                )
            )

    def test_ambiguous_weight_shape_raises(self) -> None:
        bridge = _bridge(declare_norms=True)
        with pytest.raises(RuntimeError, match="cannot determine QK-norm phase"):
            bridge.set_original_component(
                _FakeAttention(_PerHeadLayerNorm((3, 5)), _PerHeadLayerNorm((3, 5)))
            )


class TestPreTransposeNormNumerics:
    """The per-head LayerNorm only broadcasts correctly on the pre-transpose view."""

    def test_per_head_weight_aligns_on_pre_transpose_view(self) -> None:
        torch.manual_seed(0)
        norm = _PerHeadLayerNorm((N_HEADS, HEAD_DIM))
        with torch.no_grad():
            norm.weight.copy_(torch.randn(N_HEADS, HEAD_DIM))
        batch, seq = 2, 5  # seq != n_heads so a mis-aligned broadcast cannot pass by luck
        q = torch.randn(batch, seq, N_HEADS, HEAD_DIM)

        pre = norm(q).transpose(1, 2)  # HF order: norm, then transpose

        # Post-transpose application is shape-incompatible for seq != n_heads.
        with pytest.raises(RuntimeError):
            norm(q.transpose(1, 2))

        # And the pre-transpose result is exactly a per-head LN with per-head scale.
        mean = q.mean(-1, keepdim=True)
        var = (q - mean).pow(2).mean(-1, keepdim=True)
        expected = (norm.weight * (q - mean) * torch.rsqrt(var + 1e-5)).transpose(1, 2)
        torch.testing.assert_close(pre, expected)
