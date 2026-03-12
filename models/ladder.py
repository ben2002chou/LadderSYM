from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from timm.layers import DropPath
from timm.models._builder import build_model_with_cfg
from timm.models.vision_transformer import (
    Attention,
    LayerScale,
    Mlp,
    VisionTransformer,
    checkpoint_filter_fn,
)
from .ladder_backbone import ladder_model_factory
 
 
BASE_CLS = ladder_model_factory(VisionTransformer)




class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.1,
        attn_drop: float = 0.1,
        init_values: Optional[float] = None,
        drop_path: float = 0.1,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,  # Ensure Mlp class is defined or imported appropriately
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1_a1 = norm_layer(dim)
        self.norm1_a2 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,  # Assuming Attention class is updated to use this
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values)
            if init_values is not None
            else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.norm2_a1 = norm_layer(dim)
        self.norm2_a2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values)
            if init_values is not None
            else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, modality: Optional[str] = None) -> torch.Tensor:
        if modality is None:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        elif modality == "a1":
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1_a1(x))))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2_a1(x))))
        elif modality == "a2":
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1_a2(x))))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2_a2(x))))
        return x

 
class LadderSymEncoder(BASE_CLS):
    """LadderSym dual-stream encoder for reference/mistake audio.

    The encoder runs two patch-token streams:
    - reference stream: score-aligned audio
    - mistake stream: performance audio

    It alternates cross-stream alignment and per-stream transformer blocks, then
    returns token embeddings for both streams.
    """
 
    def embed(
        self,
        reference_inputs: torch.Tensor,
        mistake_inputs: torch.Tensor,
    ) -> torch.Tensor:
        reference_tokens = self.reference_embedder(reference_inputs)
        mistake_tokens = self.mistake_embedder(mistake_inputs)
        return reference_tokens, mistake_tokens
 
    def forward_features(
        self,
        reference_inputs: Tensor,
        mistake_inputs: Tensor,
    ) -> Tensor:
        """Encode reference and mistake spectrograms into token features.

        Args:
            reference_inputs: Tensor shaped ``(B, mel_bins, frames)`` for score/reference audio.
            mistake_inputs: Tensor shaped ``(B, mel_bins, frames)`` for performance/mistake audio.

        Returns:
            Tuple ``(reference_tokens, mistake_tokens)`` where each element has
            shape ``(B, num_patches, embed_dim)``.
        """
        # Reference embedding.
        reference_inputs = reference_inputs.unsqueeze(1)
        reference_inputs = reference_inputs.transpose(2, 3)
        reference_tokens = self.reference_embedder(reference_inputs)

        # Mistake embedding.
        mistake_inputs = mistake_inputs.unsqueeze(1)
        mistake_inputs = mistake_inputs.transpose(2, 3)
        mistake_tokens = self.mistake_embedder(mistake_inputs)


        # Add positional embeddings.
        reference_tokens = self._pos_embed(reference_tokens, self.reference_pos_embed)
        mistake_tokens = self._pos_embed(mistake_tokens, self.mistake_pos_embed)

        # Drop out and normalize mistake stream.
        mistake_tokens = self.patch_drop(mistake_tokens)
        mistake_tokens = self.norm_pre(mistake_tokens)
 
        # Iterate ladder blocks.
        for i in range(0, len(self.blocks), 2):
            norm_index = i // 2

            # Reference -> mistake alignment normalization + head.
            reference_tokens = self.reference_to_mistake_norm[norm_index](reference_tokens)
            mistake_tokens = self.reference_to_mistake_norm[norm_index](mistake_tokens)

            aligned_mistake_tokens = self.reference_to_mistake_alignment_head[i](
                reference_tokens,
                mistake_tokens,
            )
            mistake_tokens = self.reference_to_mistake_fusion(
                rw=aligned_mistake_tokens,
                target=mistake_tokens,
            )

            # Block.
            mistake_tokens = self.blocks[i](mistake_tokens)
 
            # Mistake -> reference alignment normalization + head.
            reference_tokens = self.mistake_to_reference_norm[norm_index](reference_tokens)
            mistake_tokens = self.mistake_to_reference_norm[norm_index](mistake_tokens)

            aligned_reference_tokens = self.mistake_to_reference_alignment_head[i](
                mistake_tokens,
                reference_tokens,
            )
            reference_tokens = self.mistake_to_reference_fusion(
                rw=aligned_reference_tokens,
                target=reference_tokens,
            )
            reference_tokens = self.blocks[i + 1](reference_tokens)
 
        # Final layer normalization.
        reference_tokens = self.norm(reference_tokens)
        mistake_tokens = self.mistake_norm(mistake_tokens)
 
        return reference_tokens, mistake_tokens
    
    def forward(
        self,
        reference_inputs: torch.Tensor | None = None,
        mistake_inputs: torch.Tensor | None = None,
        pre_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Run LadderSym encoding and projection.

        Backward compatibility:
        Older call sites may pass ``a1``/``a2``. These are treated as
        aliases for ``reference_inputs``/``mistake_inputs``.
        """
        legacy_reference_inputs = kwargs.pop("a1", None)
        legacy_mistake_inputs = kwargs.pop("a2", None)

        if reference_inputs is None:
            reference_inputs = legacy_reference_inputs
        if mistake_inputs is None:
            mistake_inputs = legacy_mistake_inputs
        if reference_inputs is None or mistake_inputs is None:
            raise ValueError("Both reference_inputs and mistake_inputs are required.")

        reference_tokens, mistake_tokens = self.forward_features(
            reference_inputs,
            mistake_inputs,
        )
        features = torch.cat([reference_tokens, mistake_tokens], dim=1)
        features = self.proj(features)
        return features
 
def _create_ladder(
    variant: str, pretrained: bool = False, **kwargs
) -> VisionTransformer:
    out_indices = kwargs.pop("out_indices", 3)
    _filter_fn = checkpoint_filter_fn
    strict = False
    if "siglip" in variant and kwargs.get("global_pool", None) != "map":
        strict = False

    return build_model_with_cfg(
        LadderSymEncoder,
        variant,
        pretrained,
        pretrained_filter_fn=_filter_fn,
        pretrained_strict=strict,
        feature_cfg=dict(out_indices=out_indices, feature_cls="getter"),
        **kwargs,
    )

def laddersym_base_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """LadderSym base encoder used by the MT3 decoder."""
    model_args = dict(
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        alignment_head_type="ca",
        fusion_type="residual",
        mistake_embedder_type="patch",
        mistake_ps=16,
    )
    model = _create_ladder(
        "vit_base_patch16_224", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model
