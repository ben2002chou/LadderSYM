import os

os.environ["TORCH_HOME"] = "./pretrained_models"
import random
import torch
import torch.nn as nn
import timm
from typing import Optional
from timm.layers import to_2tuple, DropPath
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed, Block, LayerScale
from .pos_embed import get_2d_sincos_pos_embed


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        """
        PatchEmbed module for image or audio input.

        Args:
            img_size (int or tuple): Size of the input image or audio.
            patch_size (int or tuple): Size of each patch.
            in_chans (int): Number of input channels.
            embed_dim (int): Dimension of the output embeddings.
        """
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # (B, in_chans, H, W)
        x = self.proj(x)         # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        # Note: norm1_a1, norm1_a2 removed, only keep a single norm
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values)
            if init_values is not None
            else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Since we're removing a1 references, we do a single path
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class ViTEncoder(nn.Module):
    """
    Single-stream version:
      - Removed a1 references, only keep a2 or "x" as input.
      - Same initialization, sin-cos embedding for pos_embed_a2, single patch embed, single stream of blocks, plus blocks_u if needed.
    """
    def __init__(
        self,
        audio_length=256,
        mel_bins=512,
        patch_size=16,
        img_size=224,
        embed_dim=768,
        proj_dim=512,
        modality_depth=11,    # Formerly "modality_specific_depth"
        num_heads=12,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        tr_pos=True,
    ):
        super().__init__()
        # Overwrite Timm's references with our single-stream patch embed & block
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        timm.models.vision_transformer.Block = Block

        self.audio_length = audio_length
        self.mel_bins = mel_bins

        self.patch_embed = PatchEmbed(img_size, patch_size, 1, embed_dim)
        self.patch_embed.num_patches = int(self.audio_length * self.mel_bins / 256)

        print(
            "Number of Audio Patches (single stream): {:d}".format(
                self.patch_embed.num_patches
            )
        )
        print("modality_depth: ", modality_depth, flush=True)
        print("joint depth: ", 12 - modality_depth, flush=True)

        # Single "modality" token
        self.modality_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Single pos_embed for this audio
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, embed_dim),
            requires_grad=tr_pos,
        )

        # "modality_depth" for the single stream
        self.blocks_mod = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(modality_depth)
            ]
        )
        # "blocks_u" for the remaining
        self.blocks_u = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(12 - modality_depth)
            ]
        )

        self.norm_mod = norm_layer(embed_dim)
        self.norm = norm_layer(embed_dim)

        self.proj = nn.Linear(embed_dim, proj_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # Sin-cos pos embedding for pos_embed
        # We interpret patch_embed.num_patches as a square. If it's not square, adapt as needed.
        pos_embed_data = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            8,  # or another dimension if your grid is not 8x( num_patches/8 )
            int(self.patch_embed.num_patches / 8),
            cls_token=False,
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed_data).float().unsqueeze(0)
        )

        # xavier for patch_embed proj
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.modality_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following your snippet
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        """
        Single-stream forward
        x: (B, T, mel_bins) => we do same reshaping as your code did for a2:
           1) unsqueeze channel
           2) transpose if needed
        """
        # input shape: (B, T, mel_bins)
        # -> unsqueeze(1), then transpose(2, 3) to match (B, in_chans=1, H, W)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        # patchify
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # add pos embed
        x = x + self.pos_embed
        # add single modality token
        x = x + self.modality_token

        # pass through "modality_depth" blocks
        for blk in self.blocks_mod:
            x = blk(x)

        # pass through "joint" blocks
        for blk in self.blocks_u:
            x = blk(x)

        x = self.norm(x)
        x = self.proj(x)  # (B, num_patches, proj_dim)
        return x