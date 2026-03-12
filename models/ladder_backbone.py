from functools import partial
from typing import Callable, Literal, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import timm
from timm.layers import Mlp, DropPath, get_norm_layer, to_2tuple, trunc_normal_
from timm.layers.typing import LayerType
from timm.models.vision_transformer import Attention, LayerScale


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        **kwargs,
    ):
        print("Custom PatchEmbed is being used")
        print("in_chans", in_chans, flush=True)
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class CrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(embed_dim, embed_dim)
        self.q_norm = nn.LayerNorm(embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.k_norm = nn.LayerNorm(embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.v_norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        input_tokens: torch.Tensor,
        query_tokens: torch.Tensor,
    ) -> torch.Tensor:
        b, n, c = input_tokens.shape
        _, m, _ = query_tokens.shape

        q = self.q_norm(self.q(query_tokens)).reshape(
            b, m, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        k = self.k_norm(self.k(input_tokens)).reshape(
            b, n, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        v = self.v_norm(self.v(input_tokens)).reshape(
            b, n, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)

        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            scale=self.scale,
        )
        out = out.transpose(1, 2).reshape(b, m, c)
        return self.proj(out)


class ResidualFusion(nn.Module):
    def forward(self, rw: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return target + rw


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
        mlp_layer: nn.Module = Mlp,
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
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
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
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
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


def _create_cross_attention_heads(embed_dim: int, num_heads: int, depth: int) -> nn.ModuleList:
    return nn.ModuleList(
        [CrossAttention(embed_dim=embed_dim, num_heads=num_heads) for _ in range(depth)]
    )


def ladder_model_factory(model_cls):
    class LadderBackbone(model_cls):
        def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            audio_length: int = 256,
            mel_bins: int = 512,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 1,
            num_classes: int = 1000,
            global_pool: Literal["", "avg", "token", "map"] = "token",
            embed_dim: int = 768,
            proj_dim: int = 512,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            no_embed_class: bool = False,
            reg_tokens: int = 0,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            drop_rate: float = 0.0,
            pos_drop_rate: float = 0.0,
            patch_drop_rate: float = 0.0,
            proj_drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            weight_init: Literal["skip", "jax", "jax_nlhb", "moco", ""] = "",
            fix_init: bool = False,
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[LayerType] = None,
            act_layer: Optional[LayerType] = None,
            block_fn: Type[nn.Module] = Block,
            mlp_layer: Type[nn.Module] = Mlp,
            **kwargs,
        ) -> None:
            super().__init__(
                img_size,
                patch_size,
                in_chans,
                num_classes,
                global_pool,
                embed_dim,
                depth,
                num_heads,
                mlp_ratio,
                qkv_bias,
                qk_norm,
                init_values,
                class_token,
                no_embed_class,
                reg_tokens,
                pre_norm,
                fc_norm,
                dynamic_img_size,
                dynamic_img_pad,
                drop_rate,
                pos_drop_rate,
                patch_drop_rate,
                proj_drop_rate,
                attn_drop_rate,
                drop_path_rate,
                weight_init,
                fix_init,
                embed_layer,
                norm_layer,
                act_layer,
                block_fn,
                mlp_layer,
            )

            print("in_chans laddersym_encoder", in_chans, flush=True)
            timm.models.vision_transformer.PatchEmbed = PatchEmbed
            timm.models.vision_transformer.Block = Block
            norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)

            self.img_size = img_size
            self.audio_length = audio_length
            self.mel_bins = mel_bins

            # Keep legacy attribute names for checkpoint compatibility.
            self.memory_ps = memory_ps = kwargs.pop(
                "reference_ps",
                kwargs.pop("memory_ps", patch_size),
            )
            self.process_ps = process_ps = kwargs.pop(
                "mistake_ps",
                kwargs.pop("process_ps", None),
            )
            self.process_embedder_type = kwargs.pop(
                "mistake_embedder_type",
                kwargs.pop("process_embedder_type", "patch"),
            )
            self.rw_head_type = kwargs.pop(
                "alignment_head_type",
                kwargs.pop("rw_head_type", "ca"),
            )
            self.fusion_type = kwargs.pop("fusion_type", "residual")
            self.memory_block = kwargs.pop(
                "reference_state_block",
                kwargs.pop("memory_blocks", ""),
            )

            if process_ps is None:
                raise ValueError("mistake_ps/process_ps must be specified")
            if self.process_embedder_type != "patch":
                raise ValueError("Only mistake_embedder_type/process_embedder_type='patch' is supported")
            if self.rw_head_type != "ca":
                raise ValueError("Only alignment_head_type/rw_head_type='ca' is supported")
            if self.fusion_type != "residual":
                raise ValueError("Only fusion_type='residual' is supported")
            if self.memory_block not in ("", None):
                raise ValueError("Only reference_state_block/memory_blocks='' is supported")

            self.memory_embedder = embed_layer(
                img_size=img_size,
                patch_size=memory_ps,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
            print("Using Patch Embedder for Mistake stream.")
            self.process_embedder = embed_layer(
                img_size=img_size,
                patch_size=process_ps,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
            reference_size = self.memory_size = self.memory_embedder.num_patches
            mistake_size = self.process_size = self.process_embedder.num_patches

            mistake_embed_len = int(self.audio_length * self.mel_bins / 256)
            reference_embed_len = int(self.audio_length * self.mel_bins / 256)
            self.process_pos_embed = nn.Parameter(
                torch.randn(1, mistake_embed_len, embed_dim) * 0.02
            )
            self.memory_pos_embed = nn.Parameter(
                torch.randn(1, reference_embed_len, embed_dim) * 0.02
            )

            self.memory_h = self.memory_w = self.img_size[0] // memory_ps
            self.process_h = self.process_w = self.img_size[0] // process_ps

            # Legacy read/write names are kept only for checkpoint key compatibility.
            # Runtime code uses alignment aliases (reference_to_mistake / mistake_to_reference).
            self.read_norm = nn.ModuleList(
                [norm_layer(embed_dim) for _ in range(0, len(self.blocks), 2)]
            )
            self.write_norm = nn.ModuleList(
                [norm_layer(embed_dim) for _ in range(0, len(self.blocks), 2)]
            )

            self.process_norm = norm_layer(embed_dim)
            self.process_head = nn.Linear(embed_dim, num_classes)
            self.fc_norm_process = norm_layer(embed_dim)

            # Legacy forward path compatibility.
            self.mem_blocks = nn.ModuleList([nn.Identity() for _ in range(depth)])

            self.write_head = _create_cross_attention_heads(
                embed_dim=embed_dim,
                num_heads=num_heads,
                depth=depth,
            )
            self.read_head = _create_cross_attention_heads(
                embed_dim=embed_dim,
                num_heads=num_heads,
                depth=depth,
            )
            self.read_fusion = ResidualFusion()
            self.write_fusion = ResidualFusion()

            if weight_init != "skip":
                self.init_weights(weight_init)
            if fix_init:
                self.fix_init_weight()

            trunc_normal_(self.process_pos_embed, std=0.02)
            trunc_normal_(self.memory_pos_embed, std=0.02)
            nn.init.xavier_uniform_(self.process_embedder.proj.weight)
            nn.init.zeros_(self.process_embedder.proj.bias)

            self.proj = nn.Linear(embed_dim, proj_dim)

            print("Model Configuration:")
            print(f"  model: {self.__class__.__name__}")
            print(f"  class token: {class_token}")
            print(f"  reg tokens: {reg_tokens}")
            print(f"  global pool: {global_pool}")
            print(f"  fc norm: {self.fc_norm}")
            print(f"  reference size: {reference_size}")
            print(f"  mistake size: {mistake_size}")
            print(f"  reference_ps: {memory_ps}")
            print(f"  mistake_ps: {process_ps}")
            print(f"  alignment_head_type: {self.rw_head_type}")
            print(f"  fusion_type: {self.fusion_type}")
            print(f"  reference_state_block: {self.memory_block}")

        @property
        def reference_embedder(self):
            return self.memory_embedder

        @property
        def mistake_embedder(self):
            return self.process_embedder

        @property
        def reference_pos_embed(self):
            return self.memory_pos_embed

        @property
        def mistake_pos_embed(self):
            return self.process_pos_embed

        @property
        def reference_to_mistake_norm(self):
            return self.read_norm

        @property
        def mistake_to_reference_norm(self):
            return self.write_norm

        @property
        def reference_to_mistake_alignment_head(self):
            return self.read_head

        @property
        def mistake_to_reference_alignment_head(self):
            return self.write_head

        @property
        def reference_to_mistake_fusion(self):
            return self.read_fusion

        @property
        def mistake_to_reference_fusion(self):
            return self.write_fusion

        @property
        def mistake_norm(self):
            return self.process_norm

        def _pos_embed(self, x: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
            x = x + pos_embed
            return self.pos_drop(x)

        def forward_features(
            self,
            x: torch.Tensor,
            keep_mask_reference: torch.Tensor | None = None,
            keep_mask_mistake: torch.Tensor | None = None,
            keep_mask_memory: torch.Tensor | None = None,
            keep_mask_process: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if keep_mask_reference is None:
                keep_mask_reference = keep_mask_memory
            if keep_mask_mistake is None:
                keep_mask_mistake = keep_mask_process

            reference_tokens = self.reference_embedder(x)
            mistake_tokens = self.mistake_embedder(x)

            reference_tokens = self._pos_embed(reference_tokens, self.reference_pos_embed)
            mistake_tokens = self._pos_embed(mistake_tokens, self.mistake_pos_embed)

            if isinstance(keep_mask_reference, torch.Tensor):
                _, _, d = reference_tokens.shape
                reference_tokens = torch.gather(
                    reference_tokens,
                    dim=1,
                    index=keep_mask_reference.unsqueeze(-1).expand(-1, -1, d),
                )
            if isinstance(keep_mask_mistake, torch.Tensor):
                _, _, d = mistake_tokens.shape
                mistake_tokens = torch.gather(
                    mistake_tokens,
                    dim=1,
                    index=keep_mask_mistake.unsqueeze(-1).expand(-1, -1, d),
                )

            mistake_tokens = self.patch_drop(mistake_tokens)
            mistake_tokens = self.norm_pre(mistake_tokens)

            for i, block in enumerate(self.blocks):
                norm_index = i // 2
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

                mistake_tokens = block(mistake_tokens)

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
                reference_tokens = self.mem_blocks[i](reference_tokens)

            reference_tokens = self.norm(reference_tokens)
            mistake_tokens = self.mistake_norm(mistake_tokens)
            return reference_tokens, mistake_tokens

        def forward(
            self,
            x: torch.Tensor,
            pre_logits: bool = False,
            **kwargs,
        ) -> torch.Tensor:
            reference_tokens, mistake_tokens = self.forward_features(x)
            reference_pred, mistake_pred = self.forward_head(
                reference_tokens,
                mistake_tokens,
                pre_logits=pre_logits,
            )
            return reference_pred, mistake_pred

        def forward_head(self, reference_tokens, mistake_tokens, pre_logits=False):
            reference_tokens = reference_tokens.mean(dim=1)
            mistake_tokens = mistake_tokens.mean(dim=1)

            reference_tokens = self.fc_norm(reference_tokens)
            mistake_tokens = self.fc_norm_process(mistake_tokens)

            mistake_tokens = self.head_drop(mistake_tokens)
            reference_tokens = self.head_drop(reference_tokens)
            if pre_logits:
                return reference_tokens, mistake_tokens
            return self.head(reference_tokens), self.process_head(mistake_tokens)

    return LadderBackbone
