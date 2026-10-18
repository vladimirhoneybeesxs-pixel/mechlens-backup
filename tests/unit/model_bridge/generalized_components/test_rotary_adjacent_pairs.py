"""Unit tests for RoPE pairing conventions in PositionEmbeddingsAttentionBridge.

Two conventions exist in HF:

- Llama (default): cos/sin = cat(freqs, freqs); dim ``i`` pairs with ``i + D/2``.
- Cohere / GPT-J / GPT-NeoX (``cfg.rotary_adjacent_pairs=True``): cos/sin =
  repeat_interleave(freqs, 2); dim ``2i`` pairs with ``2i+1``.

The bridge reimplements attention and therefore must choose the rotation that
matches the cos/sin layout the delegated HF rotary module emits. Using the
Llama rotation against interleaved cos/sin is not a rotation and diverges from
HF at every position > 0 (the Cohere registry failure).
"""
from types import SimpleNamespace

import torch

from transformer_lens.model_bridge.generalized_components import LinearBridge
from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
    PositionEmbeddingsAttentionBridge,
)

B, H, S, D = 2, 4, 5, 8


def _bridge(adjacent_pairs: bool) -> PositionEmbeddingsAttentionBridge:
    cfg = SimpleNamespace(
        n_heads=H,
        d_model=H * D,
        d_head=D,
        positional_embedding_type="rotary",
        rotary_adjacent_pairs=adjacent_pairs,
    )
    return PositionEmbeddingsAttentionBridge(
        name="self_attn",
        config=cfg,
        submodules={
            "q": LinearBridge(name="q_proj"),
            "k": LinearBridge(name="k_proj"),
            "v": LinearBridge(name="v_proj"),
            "o": LinearBridge(name="o_proj"),
        },
    )


def _freqs(base: float = 10000.0) -> torch.Tensor:
    inv_freq = 1.0 / (base ** (torch.arange(0, D, 2).float() / D))  # [D/2]
    pos = torch.arange(S).float()
    return torch.outer(pos, inv_freq)  # [S, D/2]


def _cos_sin_interleaved() -> tuple[torch.Tensor, torch.Tensor]:
    emb = torch.repeat_interleave(_freqs(), 2, dim=-1)  # [S, D]
    return emb.cos().expand(B, S, D), emb.sin().expand(B, S, D)


def _cos_sin_half_split() -> tuple[torch.Tensor, torch.Tensor]:
    f = _freqs()
    emb = torch.cat((f, f), dim=-1)
    return emb.cos().expand(B, S, D), emb.sin().expand(B, S, D)


def _reference_interleaved_rotation(x: torch.Tensor) -> torch.Tensor:
    """Rotate each adjacent pair (x[2i], x[2i+1]) by angle pos * inv_freq[i]."""
    f = _freqs()  # [S, D/2]
    cos, sin = f.cos(), f.sin()
    x_pairs = x.reshape(B, H, S, D // 2, 2)
    x1, x2 = x_pairs[..., 0], x_pairs[..., 1]
    r1 = x1 * cos - x2 * sin
    r2 = x1 * sin + x2 * cos
    return torch.stack((r1, r2), dim=-1).reshape(B, H, S, D)


class TestRotateHalfAdjacentPairs:
    def test_pairs_adjacent_dims(self) -> None:
        x = torch.arange(float(D)).view(1, 1, 1, D)
        out = PositionEmbeddingsAttentionBridge._rotate_half_adjacent_pairs(x)
        expected = torch.tensor([-1.0, 0.0, -3.0, 2.0, -5.0, 4.0, -7.0, 6.0]).view(1, 1, 1, D)
        torch.testing.assert_close(out, expected)

    def test_is_shape_preserving(self) -> None:
        x = torch.randn(B, H, S, D)
        assert PositionEmbeddingsAttentionBridge._rotate_half_adjacent_pairs(x).shape == x.shape


class TestApplyRotaryPosEmbConvention:
    def test_adjacent_pairs_matches_reference_rotation(self) -> None:
        torch.manual_seed(0)
        bridge = _bridge(adjacent_pairs=True)
        q, k = torch.randn(B, H, S, D), torch.randn(B, H, S, D)
        cos, sin = _cos_sin_interleaved()
        q_out, k_out = bridge._apply_rotary_pos_emb(q, k, cos, sin)
        torch.testing.assert_close(q_out, _reference_interleaved_rotation(q))
        torch.testing.assert_close(k_out, _reference_interleaved_rotation(k))

    def test_adjacent_pairs_preserves_norm(self) -> None:
        # A true rotation keeps every (2i, 2i+1) pair's length. The Llama rotation
        # against interleaved cos/sin does not — this is the bug being guarded.
        torch.manual_seed(1)
        bridge = _bridge(adjacent_pairs=True)
        q = torch.randn(B, H, S, D)
        cos, sin = _cos_sin_interleaved()
        q_out, _ = bridge._apply_rotary_pos_emb(q, q, cos, sin)
        torch.testing.assert_close(q_out.norm(dim=-1), q.norm(dim=-1))

    def test_llama_rotation_against_interleaved_cos_sin_is_wrong(self) -> None:
        torch.manual_seed(2)
        bridge = _bridge(adjacent_pairs=False)  # default path = Llama rotation
        q = torch.randn(B, H, S, D)
        cos, sin = _cos_sin_interleaved()
        q_out, _ = bridge._apply_rotary_pos_emb(q, q, cos, sin)
        # Position 0 is untouched either way; positions > 0 must differ.
        torch.testing.assert_close(q_out[:, :, 0], q[:, :, 0])
        assert not torch.allclose(q_out[:, :, 1:], _reference_interleaved_rotation(q)[:, :, 1:])

    def test_default_matches_hf_llama_apply_rotary_pos_emb(self) -> None:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        torch.manual_seed(3)
        bridge = _bridge(adjacent_pairs=False)
        q, k = torch.randn(B, H, S, D), torch.randn(B, H, S, D)
        cos, sin = _cos_sin_half_split()
        q_out, k_out = bridge._apply_rotary_pos_emb(q, k, cos, sin)
        q_ref, k_ref = apply_rotary_pos_emb(q, k, cos, sin)
        torch.testing.assert_close(q_out, q_ref)
        torch.testing.assert_close(k_out, k_ref)

    def test_adjacent_pairs_matches_hf_cohere_apply_rotary_pos_emb(self) -> None:
        from transformers.models.cohere.modeling_cohere import (
            apply_rotary_pos_emb as cohere_apply_rotary_pos_emb,
        )

        torch.manual_seed(4)
        bridge = _bridge(adjacent_pairs=True)
        q, k = torch.randn(B, H, S, D), torch.randn(B, H, S, D)
        cos, sin = _cos_sin_interleaved()
        q_out, k_out = bridge._apply_rotary_pos_emb(q, k, cos, sin)
        q_ref, k_ref = cohere_apply_rotary_pos_emb(q, k, cos, sin)
        torch.testing.assert_close(q_out, q_ref)
        torch.testing.assert_close(k_out, k_ref)

    def test_adjacent_pairs_preserves_dtype(self) -> None:
        bridge = _bridge(adjacent_pairs=True)
        q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
        cos, sin = _cos_sin_interleaved()
        q_out, _ = bridge._apply_rotary_pos_emb(
            q, q, cos.to(torch.bfloat16), sin.to(torch.bfloat16)
        )
        assert q_out.dtype == torch.bfloat16
