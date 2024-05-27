# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import math
import torch
import torch.nn as nn

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .modules import ModuleParallel, LayerNormParallel


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ModuleParallel(nn.Linear(in_features, hidden_features))
        self.dwconv = DWConv(hidden_features)
        self.act = ModuleParallel(act_layer())
        self.fc2 = ModuleParallel(nn.Linear(hidden_features, out_features))
        self.drop = ModuleParallel(nn.Dropout(drop))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = [self.dwconv(x[0], H, W), self.dwconv(x[1], H, W)]
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Mlp_2(nn.Module):
    """Multilayer perceptron."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        n_heads=8,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = ModuleParallel(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = ModuleParallel(nn.Linear(dim, dim * 2, bias=qkv_bias))
        self.attn_drop = ModuleParallel(nn.Dropout(attn_drop))
        self.proj = ModuleParallel(nn.Linear(dim, dim))
        self.proj_drop = ModuleParallel(nn.Dropout(proj_drop))

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = ModuleParallel(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            )
            self.norm = LayerNormParallel(dim)

        self.cross_heads = n_heads
        self.cross_attn_0_to_1 = nn.MultiheadAttention(
            dim, self.cross_heads, dropout=0.0, batch_first=False
        )
        self.cross_attn_1_to_0 = nn.MultiheadAttention(
            dim, self.cross_heads, dropout=0.0, batch_first=False
        )

        self.relation_judger = nn.Sequential(
            Mlp_2(dim * 2, dim, dim), torch.nn.Softmax(dim=-1)
        )

        self.k_noise = nn.Embedding(2, dim)
        self.v_noise = nn.Embedding(2, dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(
        self,
        x,
        H,
        W,
    ):
        B, N, C = x[0].shape
        q = self.q(x)
        q = [
            q_.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            for q_ in q
        ]

        if self.sr_ratio > 1:
            x = [x_.permute(0, 2, 1).reshape(B, C, H, W) for x_ in x]
            x = self.sr(x)
            x = [x_.reshape(B, C, -1).permute(0, 2, 1) for x_ in x]
            x = self.norm(x)
            kv = self.kv(x)
            kv = [
                kv_.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(
                    2, 0, 3, 1, 4
                )
                for kv_ in kv
            ]
        else:
            kv = self.kv(x)
            kv = [
                kv_.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(
                    2, 0, 3, 1, 4
                )
                for kv_ in kv
            ]
        k, v = [kv[0][0], kv[1][0]], [kv[0][1], kv[1][1]]

        attn = [(q_ @ k_.transpose(-2, -1)) * self.scale for (q_, k_) in zip(q, k)]
        attn = [attn_.softmax(dim=-1) for attn_ in attn]
        attn = self.attn_drop(attn)

        x = [
            (attn_ @ v_).transpose(1, 2).reshape(B, N, C)
            for (attn_, v_) in zip(attn, v)
        ]

        # cross-attn per batch
        new_x0 = []
        new_x1 = []
        for bs in range(B):
            ## 1. 0_to_1 cross attn and skip connect
            q = x[0][bs].unsqueeze(0)

            judger_input = torch.cat(
                [x[0][bs].unsqueeze(0), x[1][bs].unsqueeze(0)], dim=-1
            )

            relation_score = self.relation_judger(judger_input)

            noise_k = self.k_noise.weight[0] + q
            noise_v = self.v_noise.weight[0] + q

            k = torch.cat([noise_k, torch.mul(q, relation_score)], dim=0)
            v = torch.cat([noise_v, x[1][bs].unsqueeze(0)], dim=0)

            new_x0.append(x[0][bs] + self.cross_attn_0_to_1(q, k, v)[0].squeeze(0))

            ## 2. 1_to_0 cross attn and skip connect
            q = x[1][bs].unsqueeze(0)

            judger_input = torch.cat(
                [x[1][bs].unsqueeze(0), x[0][bs].unsqueeze(0)], dim=-1
            )

            relation_score = self.relation_judger(judger_input)

            noise_k = self.k_noise.weight[1] + q
            noise_v = self.v_noise.weight[1] + q

            k = torch.cat([noise_k, torch.mul(q, relation_score)], dim=0)
            v = torch.cat([noise_v, x[0][bs].unsqueeze(0)], dim=0)

            new_x1.append(x[1][bs] + self.cross_attn_1_to_0(q, k, v)[0].squeeze(0))

        new_x0 = torch.stack(new_x0)
        new_x1 = torch.stack(new_x1)
        x[0] = new_x0
        x[1] = new_x1

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=LayerNormParallel,
        sr_ratio=1,
        n_heads=8,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            n_heads=n_heads,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = (
            ModuleParallel(DropPath(drop_path))
            if drop_path > 0.0
            else ModuleParallel(nn.Identity())
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B = x[0].shape[0]

        f = self.drop_path(
            self.attn(
                self.norm1(x),
                H,
                W,
            )
        )
        x = [x_ + f_ for (x_, f_) in zip(x, f)]
        f = self.drop_path(self.mlp(self.norm2(x), H, W))
        x = [x_ + f_ for (x_, f_) in zip(x, f)]

        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = ModuleParallel(
            nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=stride,
                padding=(patch_size[0] // 2, patch_size[1] // 2),
            )
        )
        self.norm = LayerNormParallel(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x[0].shape
        x = [x_.flatten(2).transpose(1, 2) for x_ in x]
        x = self.norm(x)
        return x, H, W


class MixVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dims=[64, 128, 256, 512],
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=LayerNormParallel,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        n_heads=8,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=7,
            stride=4,
            in_chans=in_chans,
            embed_dim=embed_dims[0],
        )
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
        )
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2],
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3],
        )

        # transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratios[0],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                    n_heads=n_heads,
                )
                for i in range(depths[0])
            ]
        )
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratios[1],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[1],
                    n_heads=n_heads,
                )
                for i in range(depths[1])
            ]
        )
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[2],
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratios[2],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[2],
                    n_heads=n_heads,
                )
                for i in range(depths[2])
            ]
        )
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratios[3],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[3],
                    n_heads=n_heads,
                )
                for i in range(depths[3])
            ]
        )
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "pos_embed1",
            "pos_embed2",
            "pos_embed3",
            "pos_embed4",
            "cls_token",
        }  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        B = x[0].shape[0]
        outs0, outs1 = [], []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):

            x = blk(
                x,
                H,
                W,
            )
        x = self.norm1(x)
        x = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x]
        outs0.append(x[0])
        outs1.append(x[1])

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):

            x = blk(
                x,
                H,
                W,
            )
        x = self.norm2(x)
        x = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x]
        outs0.append(x[0])
        outs1.append(x[1])

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):

            x = blk(
                x,
                H,
                W,
            )
        x = self.norm3(x)
        x = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x]
        outs0.append(x[0])
        outs1.append(x[1])

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):

            x = blk(
                x,
                H,
                W,
            )
        x = self.norm4(x)
        x = [x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for x_ in x]
        outs0.append(x[0])
        outs1.append(x[1])

        return [outs0, outs1]

    def forward(self, x):
        x = self.forward_features(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class mit_b0(MixVisionTransformer):
    def __init__(self, n_heads, dpr, drop_rate, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4,
            embed_dims=[32, 64, 160, 256],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=LayerNormParallel,
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=drop_rate,
            drop_path_rate=dpr,
            n_heads=n_heads,
        )


class mit_b1(MixVisionTransformer):
    def __init__(self, n_heads, dpr, drop_rate, **kwargs):
        super(mit_b1, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=LayerNormParallel,
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=drop_rate,
            drop_path_rate=dpr,
            n_heads=n_heads,
        )


class mit_b2(MixVisionTransformer):
    def __init__(self, n_heads, dpr, drop_rate, **kwargs):
        super(mit_b2, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=LayerNormParallel,
            depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=drop_rate,
            drop_path_rate=dpr,
            n_heads=n_heads,
        )


class mit_b3(MixVisionTransformer):
    def __init__(self, n_heads, dpr, drop_rate, **kwargs):
        super(mit_b3, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=LayerNormParallel,
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=drop_rate,
            drop_path_rate=dpr,
            n_heads=n_heads,
        )


class mit_b4(MixVisionTransformer):
    def __init__(self, n_heads, dpr, drop_rate, **kwargs):
        super(mit_b4, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=LayerNormParallel,
            depths=[3, 8, 27, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=drop_rate,
            drop_path_rate=dpr,
            n_heads=n_heads,
        )


class mit_b5(MixVisionTransformer):
    def __init__(self, n_heads, dpr, drop_rate, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=LayerNormParallel,
            depths=[3, 6, 40, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=drop_rate,
            drop_path_rate=dpr,
            n_heads=n_heads,
        )
