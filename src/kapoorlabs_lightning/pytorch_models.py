import itertools
from typing import Literal
import numpy as np
import os
import torch
import torch.nn as nn
import logging
import math
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import Dataset
from tifffile import imwrite
from .graph_functions import get_graph_feature, knn, local_cov, local_maxpool
from trackastra.model import TrackingTransformer

logger = logging.getLogger(__name__)


class FeedForward(nn.Module):
    def __init__(self, d_model, expand: float = 2, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(d_model, int(d_model * expand))
        self.fc2 = nn.Linear(int(d_model * expand), d_model, bias=bias)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def _pos_embed_fourier1d_init(
    cutoff: float = 256, n: int = 32, cutoff_start = 1
):
    return (
        torch.exp(torch.linspace(-math.log(cutoff_start), -math.log(cutoff), n))
        .unsqueeze(0)
        .unsqueeze(0)
    )

def _time_embed_fourier1d_init(
    cutoff: float = 256, n: int = 32
):
    return (
            torch.linspace(1, cutoff, steps=n)
            .unsqueeze(0)
            .unsqueeze(0)
        )

class TemporalEncoding(nn.Module):
    def __init__(self, num_frequencies: int = 16):
        """Simplified temporal positional encoding for temporal data.
        
        Args:
            num_frequencies: Number of different sine/cosine frequencies used for encoding.
        """
        super().__init__()
        self.num_frequencies = num_frequencies

    def forward(self, x):
        """Add positional encoding to the input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, feature_dim).
        
        Returns:
            Tensor with added positional encoding, shape (batch_size, sequence_length, feature_dim).
        """
        batch_size, sequence_length, feature_dim = x.shape

        # Generate positional encoding matrix on-the-fly
        position = torch.arange(0, sequence_length, dtype=torch.float32, device=x.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, feature_dim, 2, dtype=torch.float32, device=x.device) * -(math.log(10000.0) / feature_dim)
        )

        P = torch.zeros((sequence_length, feature_dim), device=x.device)
        P[:, 0::2] = torch.sin(position * div_term)
        P[:, 1::2] = torch.cos(position * div_term)

        # Add positional encoding to x
        return x + P.unsqueeze(0)  # Shape: (1, sequence_length, feature_dim) -> Broadcast to batch size


    

class PositionalEncoding(nn.Module):
    def __init__(
        self,
        cutoffs: tuple[float] = (256,),
        n_pos: tuple[int] = (32,),
        cutoffs_start=None,
    ):
        """Positional encoding with given cutoff and number of frequencies for each dimension.
        number of dimension is inferred from the length of cutoffs and n_pos.
        """
        super().__init__()
        if cutoffs_start is None:
            cutoffs_start = (1,) * len(cutoffs)

        assert len(cutoffs) == len(n_pos)
        self.freqs = nn.ParameterList(
            [
                nn.Parameter(_pos_embed_fourier1d_init(cutoff, n // 2))
                for cutoff, n, cutoff_start in zip(cutoffs, n_pos, cutoffs_start)
            ]
        )

    def forward(self, coords: torch.Tensor):
        _B, _N, D = coords.shape
        assert D == len(self.freqs)
        embed = torch.cat(
            tuple(
                torch.cat(
                    (
                        torch.sin(0.5 * math.pi * x.unsqueeze(-1) * freq),
                        torch.cos(0.5 * math.pi * x.unsqueeze(-1) * freq),
                    ),
                    axis=-1,
                )
                / math.sqrt(len(freq))
                for x, freq in zip(coords.moveaxis(-1, 0), self.freqs)
            ),
            axis=-1,
        )

        return embed


class NoPositionalEncoding(nn.Module):
    def __init__(self, d):
        """One learnable input token that ignores positional information."""
        super().__init__()
        self.d = d
        # self.token = nn.Parameter(torch.randn(d))

    def forward(self, coords: torch.Tensor):
        B, N, _ = coords.shape
        return (
            # torch.ones((B, N, self.d), device=coords.device) * 0.1
            # torch.randn((1, 1, self.d), device=coords.device).expand(B, N, -1) * 0.01
            torch.randn((B, N, self.d), device=coords.device) * 0.01
            + torch.randn((1, 1, self.d), device=coords.device).expand(B, N, -1) * 0.1
        )
        # return self.token.view(1, 1, -1).expand(B, N, -1)


def _bin_init_exp(cutoff: float, n: int):
    return torch.exp(torch.linspace(0, math.log(cutoff + 1), n))


def _bin_init_linear(cutoff: float, n: int):
    return torch.linspace(-cutoff, cutoff, n)


def _pos_rot_embed_fourier1d_init(cutoff: float = 128, n: int = 32):
    # Maximum initial frequency is 1
    return torch.exp(torch.linspace(0, -math.log(cutoff), n)).unsqueeze(0).unsqueeze(0)


# https://github.com/cvg/LightGlue/blob/b1cd942fc4a3a824b6aedff059d84f5c31c297f6/lightglue/lightglue.py#L51
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of scalars as 2d vectors by pi/2.
    Refer to eq 34 in https://arxiv.org/pdf/2104.09864.pdf.
    """
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, cutoffs: tuple[float] = (256,), n_pos: tuple[int] = (32,)):
        """Rotary positional encoding with given cutoff and number of frequencies for each dimension.
        number of dimension is inferred from the length of cutoffs and n_pos.

        see
        https://arxiv.org/pdf/2104.09864.pdf
        """
        super().__init__()
        assert len(cutoffs) == len(n_pos)
        if not all(n % 2 == 0 for n in n_pos):
            raise ValueError("n_pos must be even")

        self._n_dim = len(cutoffs)
        # theta in RoFormer https://arxiv.org/pdf/2104.09864.pdf
        self.freqs = nn.ParameterList(
            [
                nn.Parameter(_pos_rot_embed_fourier1d_init(cutoff, n // 2))
                for cutoff, n in zip(cutoffs, n_pos)
            ]
        )

    def get_co_si(self, coords: torch.Tensor):
        _B, _N, D = coords.shape
        assert D == len(self.freqs)
        co = torch.cat(
            tuple(
                torch.cos(0.5 * math.pi * x.unsqueeze(-1) * freq) / math.sqrt(len(freq))
                for x, freq in zip(coords.moveaxis(-1, 0), self.freqs)
            ),
            axis=-1,
        )
        si = torch.cat(
            tuple(
                torch.sin(0.5 * math.pi * x.unsqueeze(-1) * freq) / math.sqrt(len(freq))
                for x, freq in zip(coords.moveaxis(-1, 0), self.freqs)
            ),
            axis=-1,
        )

        return co, si

    def forward(self, q: torch.Tensor, k: torch.Tensor, coords: torch.Tensor):
        _B, _N, D = coords.shape
        _B, _H, _N, _C = q.shape

        if not D == self._n_dim:
            raise ValueError(f"coords must have {self._n_dim} dimensions, got {D}")

        co, si = self.get_co_si(coords)

        co = co.unsqueeze(1).repeat_interleave(2, dim=-1)
        si = si.unsqueeze(1).repeat_interleave(2, dim=-1)
        q2 = q * co + _rotate_half(q) * si
        k2 = k * co + _rotate_half(k) * si

        return q2, k2


class RelativePositionalBias(nn.Module):
    def __init__(
        self,
        n_head: int,
        cutoff_spatial: float,
        cutoff_temporal: float,
        n_spatial: int = 32,
        n_temporal: int = 16,
    ):
        """Learnt relative positional bias to add to self-attention matrix.

        Spatial bins are exponentially spaced, temporal bins are linearly spaced.

        Args:
            n_head (int): Number of pos bias heads. Equal to number of attention heads
            cutoff_spatial (float): Maximum distance in space.
            cutoff_temporal (float): Maxium distance in time. Equal to window size of transformer.
            n_spatial (int, optional): Number of spatial bins.
            n_temporal (int, optional): Number of temporal bins in each direction. Should be equal to window size. Total = 2 * n_temporal + 1. Defaults to 16.
        """
        super().__init__()
        self._spatial_bins = _bin_init_exp(cutoff_spatial, n_spatial)
        self._temporal_bins = _bin_init_linear(cutoff_temporal, 2 * n_temporal + 1)
        self.register_buffer("spatial_bins", self._spatial_bins)
        self.register_buffer("temporal_bins", self._temporal_bins)
        self.n_spatial = n_spatial
        self.n_head = n_head
        self.bias = nn.Parameter(
            -0.5 + torch.rand((2 * n_temporal + 1) * n_spatial, n_head)
        )

    def forward(self, coords: torch.Tensor):
        _B, _N, _D = coords.shape
        t = coords[..., 0]
        yx = coords[..., 1:]
        temporal_dist = t.unsqueeze(-1) - t.unsqueeze(-2)
        spatial_dist = torch.cdist(yx, yx)

        spatial_idx = torch.bucketize(spatial_dist, self.spatial_bins)
        torch.clamp_(spatial_idx, max=len(self.spatial_bins) - 1)
        temporal_idx = torch.bucketize(temporal_dist, self.temporal_bins)
        torch.clamp_(temporal_idx, max=len(self.temporal_bins) - 1)

        # do some index gymnastics such that backward is not super slow
        # https://discuss.pytorch.org/t/how-to-select-multiple-indexes-over-multiple-dimensions-at-the-same-time/98532/2
        idx = spatial_idx.flatten() + temporal_idx.flatten() * self.n_spatial
        bias = self.bias.index_select(0, idx).view((*spatial_idx.shape, self.n_head))
        # -> B, nH, N, N
        bias = bias.transpose(-1, 1)
        return bias


class RelativePositionalAttention(nn.Module):
    def __init__(
        self,
        coord_dim: int,
        embed_dim: int,
        n_head: int,
        cutoff_spatial: float = 256,
        cutoff_temporal: float = 16,
        n_spatial: int = 32,
        n_temporal: int = 16,
        dropout: float = 0.0,
        mode: Literal["bias", "rope", "none"] = "bias",
    ):
        super().__init__()

        if not embed_dim % (2 * n_head) == 0:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by 2 times n_head {2 * n_head}"
            )

        # qkv projection
        self.q_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_pro = nn.Linear(embed_dim, embed_dim, bias=True)

        # output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        # regularization
        self.dropout = dropout
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.cutoff_spatial = cutoff_spatial

        if mode == "bias" or mode is True:
            self.pos_bias = RelativePositionalBias(
                n_head=n_head,
                cutoff_spatial=cutoff_spatial,
                cutoff_temporal=cutoff_temporal,
                n_spatial=n_spatial,
                n_temporal=n_temporal,
            )
        elif mode == "rope":
            # each part needs to be divisible by 2
            n_split = 2 * (embed_dim // (2 * (coord_dim + 1) * n_head))

            self.rot_pos_enc = RotaryPositionalEncoding(
                cutoffs=((cutoff_temporal,) + (cutoff_spatial,) * coord_dim),
                n_pos=(embed_dim // n_head - coord_dim * n_split,)
                + (n_split,) * coord_dim,
            )
        elif mode == "none":
            pass
        elif mode is None or mode is False:
            logger.warning(
                "attn_positional_bias is not set (None or False), no positional bias."
            )
        else:
            raise ValueError(f"Unknown mode {mode}")

        self._mode = mode

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        coords: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ):
        B, N, D = query.size()
        q = self.q_pro(query)  # (B, N, D)
        k = self.k_pro(key)  # (B, N, D)
        v = self.v_pro(value)  # (B, N, D)
        # (B, nh, N, hs)
        k = k.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        q = q.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        v = v.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)

        attn_mask = torch.zeros(
            (B, self.n_head, N, N), device=query.device, dtype=q.dtype
        )

        # add negative value but not too large to keep mixed precision loss from becoming nan
        attn_ignore_val = -1e3

        # spatial cutoff
        yx = coords[..., 1:]
        spatial_dist = torch.cdist(yx, yx)
        spatial_mask = (spatial_dist > self.cutoff_spatial).unsqueeze(1)
        attn_mask.masked_fill_(spatial_mask, attn_ignore_val)

        # dont add positional bias to self-attention if coords is None
        if coords is not None:
            if self._mode == "bias":
                attn_mask = attn_mask + self.pos_bias(coords)
            elif self._mode == "rope":
                q, k = self.rot_pos_enc(q, k, coords)
            else:
                pass

            dist = torch.cdist(coords, coords, p=2)
            attn_mask += torch.exp(-0.1 * dist.unsqueeze(1))

        # if given key_padding_mask = (B,N) then ignore those tokens (e.g. padding tokens)
        if padding_mask is not None:
            ignore_mask = torch.logical_or(
                padding_mask.unsqueeze(1), padding_mask.unsqueeze(2)
            ).unsqueeze(1)
            attn_mask.masked_fill_(ignore_mask, attn_ignore_val)

        self.attn_mask = attn_mask.clone()

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0
        )

        y = y.transpose(1, 2).contiguous().view(B, N, D)
        # output projection
        y = self.proj(y)
        return y


class BidirectionalRelativePositionalAttention(RelativePositionalAttention):
    def forward(
        self,
        query1: torch.Tensor,
        query2: torch.Tensor,
        coords: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ):
        B, N, D = query1.size()
        q1 = self.q_pro(query1)  # (B, N, D)
        q2 = self.q_pro(query2)  # (B, N, D)
        v1 = self.v_pro(query1)  # (B, N, D)
        v2 = self.v_pro(query2)  # (B, N, D)

        # (B, nh, N, hs)
        q1 = q1.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        v1 = v1.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        q2 = q2.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        v2 = v2.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)

        attn_mask = torch.zeros(
            (B, self.n_head, N, N), device=query1.device, dtype=q1.dtype
        )

        # add negative value but not too large to keep mixed precision loss from becoming nan
        attn_ignore_val = -1e3

        # spatial cutoff
        yx = coords[..., 1:]
        spatial_dist = torch.cdist(yx, yx)
        spatial_mask = (spatial_dist > self.cutoff_spatial).unsqueeze(1)
        attn_mask.masked_fill_(spatial_mask, attn_ignore_val)

        # dont add positional bias to self-attention if coords is None
        if coords is not None:
            if self._mode == "bias":
                attn_mask = attn_mask + self.pos_bias(coords)
            elif self._mode == "rope":
                q1, q2 = self.rot_pos_enc(q1, q2, coords)
            else:
                pass

            dist = torch.cdist(coords, coords, p=2)
            attn_mask += torch.exp(-0.1 * dist.unsqueeze(1))

        # if given key_padding_mask = (B,N) then ignore those tokens (e.g. padding tokens)
        if padding_mask is not None:
            ignore_mask = torch.logical_or(
                padding_mask.unsqueeze(1), padding_mask.unsqueeze(2)
            ).unsqueeze(1)
            attn_mask.masked_fill_(ignore_mask, attn_ignore_val)

        self.attn_mask = attn_mask.clone()

        y1 = nn.functional.scaled_dot_product_attention(
            q1,
            q2,
            v1,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0,
        )
        y2 = nn.functional.scaled_dot_product_attention(
            q2,
            q1,
            v2,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0,
        )

        y1 = y1.transpose(1, 2).contiguous().view(B, N, D)
        y1 = self.proj(y1)
        y2 = y2.transpose(1, 2).contiguous().view(B, N, D)
        y2 = self.proj(y2)
        return y1, y2


class BidirectionalCrossAttention(nn.Module):
    def __init__(
        self,
        coord_dim: int = 2,
        d_model=256,
        num_heads=4,
        dropout=0.1,
        window: int = 16,
        cutoff_spatial: int = 256,
        positional_bias: Literal["bias", "rope", "none"] = "bias",
        positional_bias_n_spatial: int = 32,
    ):
        super().__init__()
        self.positional_bias = positional_bias
        self.attn = BidirectionalRelativePositionalAttention(
            coord_dim,
            d_model,
            num_heads,
            cutoff_spatial=cutoff_spatial,
            n_spatial=positional_bias_n_spatial,
            cutoff_temporal=window,
            n_temporal=window,
            dropout=dropout,
            mode=positional_bias,
        )

        self.mlp = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        coords: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ):
        x = self.norm1(x)
        y = self.norm1(y)

        # cross attention
        # setting coords to None disables positional bias
        x2, y2 = self.attn(
            x,
            y,
            coords=coords if self.positional_bias else None,
            padding_mask=padding_mask,
        )
        # print(torch.norm(x2).item()/torch.norm(x).item())
        x = x + x2
        x = x + self.mlp(self.norm2(x))
        y = y + y2
        y = y + self.mlp(self.norm2(y))

        return x, y


class TrackAsuraTransformer(TrackingTransformer):
    def __init__(
        self,
        coord_dim: int = 3,
        feat_dim: int = 0,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dropout: float = 0.1,
        pos_embed_per_dim: int = 32,
        feat_embed_per_dim: int = 1,
        window: int = 6,
        spatial_pos_cutoff: int = 256,
        attn_positional_bias: Literal["bias", "rope", "none"] = "rope",
        attn_positional_bias_n_spatial: int = 16,
        causal_norm: Literal[
            "none", "linear", "softmax", "quiet_softmax"
        ] = "quiet_softmax",
    ):

        super().__init__(
            coord_dim=coord_dim,
            feat_dim=feat_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            pos_embed_per_dim=pos_embed_per_dim,
            feat_embed_per_dim=feat_embed_per_dim,
            window=window,
            spatial_pos_cutoff=spatial_pos_cutoff,
            attn_positional_bias=attn_positional_bias,
            attn_positional_bias_n_spatial=attn_positional_bias_n_spatial,
            causal_norm=causal_norm,
        )


class TrackAsuraEncoderLayer(nn.Module):
    def __init__(
        self,
        coord_dim: int = 2,
        d_model=256,
        num_heads=4,
        dropout=0.1,
        cutoff_spatial: int = 256,
        window: int = 16,
        positional_bias: Literal["bias", "rope", "none"] = "bias",
        positional_bias_n_spatial: int = 32,
    ):
        super().__init__()
        self.positional_bias = positional_bias
        self.attn = RelativePositionalAttention(
            coord_dim,
            d_model,
            num_heads,
            cutoff_spatial=cutoff_spatial,
            n_spatial=positional_bias_n_spatial,
            cutoff_temporal=window,
            n_temporal=window,
            dropout=dropout,
            mode=positional_bias,
        )
        self.mlp = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ):
        x = self.norm1(x)

        # setting coords to None disables positional bias
        a = self.attn(
            x,
            x,
            x,
            coords=coords if self.positional_bias else None,
            padding_mask=padding_mask,
        )

        x = x + a
        x = x + self.mlp(self.norm2(x))

        return x


class TrackAsuraDecoderLayer(nn.Module):
    def __init__(
        self,
        coord_dim: int = 2,
        d_model=256,
        num_heads=4,
        dropout=0.1,
        window: int = 16,
        cutoff_spatial: int = 256,
        positional_bias: Literal["bias", "rope", "none"] = "bias",
        positional_bias_n_spatial: int = 32,
    ):
        super().__init__()
        self.positional_bias = positional_bias
        self.attn = RelativePositionalAttention(
            coord_dim,
            d_model,
            num_heads,
            cutoff_spatial=cutoff_spatial,
            n_spatial=positional_bias_n_spatial,
            cutoff_temporal=window,
            n_temporal=window,
            dropout=dropout,
            mode=positional_bias,
        )

        self.mlp = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        coords: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ):
        x = self.norm1(x)
        y = self.norm2(y)
        # cross attention
        # setting coords to None disables positional bias
        a = self.attn(
            x,
            y,
            y,
            coords=coords if self.positional_bias else None,
            padding_mask=padding_mask,
        )

        x = x + a
        x = x + self.mlp(self.norm3(x))

        return x


class TrackAsuraAttention(RelativePositionalAttention):
    def __init__(
        self,
        coord_dim: int,
        embed_dim: int,
        n_head: int,
        cutoff_spatial: float = 256,
        cutoff_temporal: float = 16,
        n_spatial: int = 32,
        n_temporal: int = 16,
        dropout: float = 0.0,
        mode: Literal["bias", "rope", "none"] = "bias",
    ):
        super().__init__(
            coord_dim=coord_dim,
            embed_dim=embed_dim,
            n_head=n_head,
            cutoff_spatial=cutoff_spatial,
            cutoff_temporal=cutoff_temporal,
            n_spatial=n_spatial,
            n_temporal=n_temporal,
            dropout=dropout,
            mode=mode,
        )


class TrackAsuraBias(RelativePositionalBias):
    def __init__(
        self,
        n_head: int,
        cutoff_spatial: float,
        cutoff_temporal: float,
        n_spatial: int = 32,
        n_temporal: int = 16,
    ):

        super().__init__(
            n_head=n_head,
            cutoff_spatial=cutoff_spatial,
            cutoff_temporal=cutoff_temporal,
            n_spatial=n_spatial,
            n_temporal=n_temporal,
        )


class TrackAsuraRotaryPositionalEncoding(RotaryPositionalEncoding):
    def __init__(self, cutoffs: tuple[float] = (256,), n_pos: tuple[int] = (32,)):

        super().__init__(cutoffs=cutoffs, n_pos=n_pos)


class TrackAsuraBidirectionalRelativePositionalAttention(
    BidirectionalRelativePositionalAttention
):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        query1: torch.Tensor,
        query2: torch.Tensor,
        coords: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ):
        return super().forward(query1, query2, coords, padding_mask)


class TrackAsuraBidirectionalCrossAttention(BidirectionalCrossAttention):
    def __init__(
        self,
        coord_dim: int = 2,
        d_model=256,
        num_heads=4,
        dropout=0.1,
        window: int = 16,
        cutoff_spatial: int = 256,
        positional_bias: Literal["bias", "rope", "none"] = "bias",
        positional_bias_n_spatial: int = 32,
    ):

        super().__init__(
            coord_dim=coord_dim,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            window=window,
            cutoff_spatial=cutoff_spatial,
            positional_bias=positional_bias,
            positional_bias_n_spatial=positional_bias_n_spatial,
        )


class DenseLayer(nn.Module):
    """ """

    def __init__(self, input_channels, growth_rate, bottleneck_size, kernel_size):
        super().__init__()
        self.use_bottleneck = bottleneck_size > 0
        self.num_bottleneck_output_filters = growth_rate * bottleneck_size
        if self.use_bottleneck:
            self.bn2 = nn.BatchNorm1d(input_channels)
            self.act2 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv1d(
                input_channels,
                self.num_bottleneck_output_filters,
                kernel_size=1,
                stride=1,
            )
        self.bn1 = nn.BatchNorm1d(self.num_bottleneck_output_filters)
        self.act1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(
            self.num_bottleneck_output_filters,
            growth_rate,
            kernel_size=kernel_size,
            stride=1,
            dilation=1,
            padding=kernel_size // 2,
        )

    def forward(self, x):
        if self.use_bottleneck:
            x = self.bn2(x)
            x = self.act2(x)
            x = self.conv2(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv1(x)
        return x


class DenseBlock(nn.ModuleDict):
    """ """

    def __init__(
        self, num_layers, input_channels, growth_rate, kernel_size, bottleneck_size
    ):
        super().__init__()
        self.num_layers = num_layers
        for i in range(self.num_layers):
            self.add_module(
                f"denselayer{i}",
                DenseLayer(
                    input_channels + i * growth_rate,
                    growth_rate,
                    bottleneck_size,
                    kernel_size,
                ),
            )

    def forward(self, x):
        layer_outputs = [x]
        for _, layer in self.items():
            x = layer(x)
            layer_outputs.append(x)
            x = torch.cat(layer_outputs, dim=1)
        return x


class TransitionBlock(nn.Module):
    """ """

    def __init__(self, input_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(
            input_channels, out_channels, kernel_size=1, stride=1, dilation=1
        )
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        x = self.conv(x)
        x = self.pool(x)
        return x


class HybridAttentionDenseNet(nn.Module):
    def __init__(
        self,
        input_channels,
        num_classes,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=32,
        bottleneck_size=4,
        kernel_size=3,
        attention_dim=64,
        n_pos=(8,),  
    ):
        super().__init__()

        if isinstance(n_pos, tuple):
            self.num_frequencies = n_pos[0]
        else:
            self.num_frequencies = n_pos

        self.temporal_encoding = TemporalEncoding( num_frequencies=self.num_frequencies)

        self.features = nn.Sequential()

        self.features.add_module(
            "conv_init",
            nn.Conv1d(
                input_channels,
                num_init_features,
                kernel_size=7,
                stride=2,
                padding=3,
                dilation=1,
            ),
        )
        self.features.add_module("bn_init", nn.BatchNorm1d(num_init_features))
        self.features.add_module("relu_init", nn.ReLU(inplace=True))
        self.features.add_module("maxpool_init", nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                input_channels=num_features,
                growth_rate=growth_rate,
                kernel_size=kernel_size,
                bottleneck_size=bottleneck_size,
            )
            self.features.add_module(f"denseblock{i}", block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = TransitionBlock(
                    input_channels=num_features, out_channels=num_features // 2
                )
                self.features.add_module(f"transitionblock{i}", trans)
                num_features = num_features // 2

            attention_layer = nn.Sequential(
                nn.Linear(num_features, attention_dim),
                nn.Tanh(),
                nn.Linear(attention_dim, 1, bias=False),
            )
            self.features.add_module(f"attentionblock{i}", attention_layer)

        self.final_bn = nn.BatchNorm1d(num_features)
        self.final_act = nn.ReLU(inplace=True)

        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):

        for name, layer in self.features.named_children():
            if "attentionblock" in name:
                x = x.permute(0, 2, 1)  
                x = self.temporal_encoding(x)  

                attention_scores = torch.tanh(layer(x))  
                attention_weights = torch.softmax(attention_scores, dim=1)  
                
                
                x = x * attention_weights  
                x = x.permute(0, 2, 1)  
            else:
                x = layer(x)

        x = torch.mean(x, dim=2)
        x = self.final_bn(x)
        x = self.final_act(x)

        
        out = self.fc(x)  
        return out
    
def get_attention_importance(model, inputs):
    model.eval()
    
    baseline_output = model(inputs).detach()  
    baseline_probabilities = torch.softmax(baseline_output, dim=1)
    baseline_predicted_class = torch.argmax(baseline_probabilities, dim=1)
    batch_size = inputs.shape[0]
    num_features = inputs.shape[1]
    importance_scores = []
    
    for b in range(batch_size):
        feature_importances = []
        for i in range(num_features):
            input_masked = inputs.clone()
            input_masked[b, i, :] = 0  
            
            masked_output = model(input_masked).detach()
            masked_probabilities = torch.softmax(masked_output, dim=1)
            masked_predicted_class = torch.argmax(masked_probabilities, dim=1)
            importance = (baseline_predicted_class[b] != masked_predicted_class[b]).float().item()
            feature_importances.append(importance)
        
        importance_scores.append(feature_importances)

    return np.array(importance_scores)


def plot_feature_importance_heatmap(model, inputs, save_dir, save_name):
    """
    Saves a heatmap of feature importance across multiple tracks.

    Parameters:
        model (nn.Module): The trained model with attention layers.
        inputs (list of torch.Tensor): input tensors, each with shape (N, F, T) for each track.
        save_dir (str): Directory to save the plot.
        save_name (str): Filename to save the plot as.
       
    """
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Collect feature importance for each track
    all_importances = []
    avg_importance = get_attention_importance(model, inputs)
    all_importances.append(avg_importance)

    # Convert to a 2D array where each row is a track and each column is a feature
    importance_matrix = np.array(all_importances)
    print(importance_matrix.shape)
    save_path = os.path.join(save_dir, save_name)
    imwrite(save_path, importance_matrix.astype(np.float32))
    


class AttentionNet(nn.Module):
    def __init__(
        self, input_channels, num_classes, attention_dim=64, cutoffs=(25,), n_pos=(8,)
    ):
        super().__init__()
        self.positional_encoding = PositionalEncoding(cutoffs=cutoffs, n_pos=n_pos)
        self.attention = nn.Linear(input_channels, attention_dim)
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)
        self.fc = nn.Linear(input_channels, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape to (N, T, F) for attention over time dimension
        x = self.positional_encoding(x)
        # Compute attention scores
        attention_scores = torch.tanh(self.attention(x))
        attention_weights = self.context_vector(attention_scores)
        attention_weights = torch.softmax(attention_weights, dim=1)

        attended_out = torch.sum(x * attention_weights, dim=1)

        out = self.fc(attended_out)
        return out


class DenseNet(nn.Module):
    def __init__(
        self,
        input_channels,
        num_classes,
        growth_rate: int = 32,
        block_config: tuple = (6, 12, 24, 16),
        num_init_features: int = 32,
        bottleneck_size: int = 4,
        kernel_size: int = 3,
    ):

        super().__init__()
        self._initialize_weights()

        self.features = nn.Sequential(
            nn.Conv1d(
                input_channels,
                num_init_features,
                kernel_size=7,
                stride=2,
                padding=3,
                dilation=1,
            ),
            nn.BatchNorm1d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                input_channels=num_features,
                growth_rate=growth_rate,
                kernel_size=kernel_size,
                bottleneck_size=bottleneck_size,
            )
            self.features.add_module(f"denseblock{i}", block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = TransitionBlock(
                    input_channels=num_features, out_channels=num_features // 2
                )
                self.features.add_module(f"transition{i}", trans)
                num_features = num_features // 2

        self.final_bn = nn.BatchNorm1d(num_features)
        self.final_act = nn.ReLU(inplace=True)
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier_1 = nn.Linear(num_features, num_classes)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.GroupNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                init.constant_(m.bias, 0)

    def forward_features(self, x):
        out = self.features(x)
        out = self.final_bn(out)
        out = self.final_act(out)
        out = self.final_pool(out)
        return out

    def forward(self, x):
        features = self.forward_features(x)
        features = features.squeeze(-1)
        out_1 = self.classifier_1(features)
        return out_1

    def reset_classifier(self):
        self.classifier = nn.Identity()

    def get_classifier(self):
        return self.classifier


class MitosisNet(nn.Module):
    def __init__(self, input_channels, num_classes):

        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_channels, out_channels=32, kernel_size=3
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(nn.functional.relu(self.conv1(x)))
        x = self.pool2(nn.functional.relu(self.conv2(x)))
        x = self.global_pooling(x).squeeze()
        x = self.fc(x)
        return x


class MitosisDataset(Dataset):
    def __init__(self, arrays, labels):
        self.arrays = arrays
        self.labels = labels
        self.input_channels = arrays.shape[2]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, idx):
        array = self.arrays[idx]
        array = torch.tensor(array).permute(1, 0).float().to(self.device)
        label = torch.tensor(self.labels[idx]).to(self.device)
        return array, label


class ClusteringLayer(nn.Module):
    def __init__(self, num_features=10, num_clusters=10, alpha=1.0):
        super().__init__()
        self.num_features = num_features
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.weight = nn.Parameter(torch.Tensor(self.num_clusters, self.num_features))
        self.weight = nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = x.unsqueeze(1) - self.weight
        x = torch.mul(x, x)
        x = torch.sum(x, dim=2)
        x = 1.0 + (x / self.alpha)
        x = 1.0 / x
        x = x ** ((self.alpha + 1.0) / 2.0)
        x = torch.t(x) / torch.sum(x, dim=1)
        x = torch.t(x)
        return x

    def extra_repr(self):
        return "in_features={}, out_features={}, alpha={}".format(
            self.num_features, self.num_clusters, self.alpha
        )

    def set_weight(self, tensor):
        self.weight = nn.Parameter(tensor)


class CloudAutoEncoder(nn.Module):
    def __init__(
        self,
        num_features,
        k=20,
        encoder_type="dgcnn",
        decoder_type="foldingnet",
        shape="plane",
        sphere_path="./sphere.npy",
        gaussian_path="./gaussian.npy",
        std=0.3,
    ):
        super().__init__()
        self.k = k
        self.num_features = num_features
        self.shape = shape
        self.sphere_path = sphere_path
        self.gaussian_path = gaussian_path
        self.std = std

        assert encoder_type.lower() in [
            "foldingnet",
            "dgcnn",
            "dgcnn_orig",
        ], "Please select an encoder type from either foldingnet or dgcnn."

        assert decoder_type.lower() in [
            "foldingnet",
            "foldingnetbasic",
        ], "Please select an decoder type from either foldingnet."

        self.encoder_type = encoder_type.lower()
        self.decoder_type = decoder_type.lower()
        if self.encoder_type == "dgcnn":
            self.encoder = DGCNNEncoder(num_features=self.num_features, k=self.k)
        # elif self.encoder_type == "dgcnn_orig":
        #     self.encoder = DGCNN(num_features=self.num_features, k=self.k)
        else:
            self.encoder = FoldNetEncoder(num_features=self.num_features, k=self.k)

        if self.decoder_type == "foldingnet":
            self.decoder = FoldNetDecoder(
                num_features=self.num_features,
                shape=self.shape,
                sphere_path=self.sphere_path,
                gaussian_path=self.gaussian_path,
                std=self.std,
            )
        else:
            self.decoder = FoldingNetBasicDecoder(
                num_features=self.num_features,
                shape=self.shape,
                sphere_path=self.sphere_path,
                gaussian_path=self.gaussian_path,
                std=self.std,
            )

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(x=features)
        return output, features


class DeepEmbeddedClustering(nn.Module):
    def __init__(self, encoder, decoder, num_clusters):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_clusters = num_clusters
        self.clustering_layer = ClusteringLayer(
            num_features=self.encoder.num_features,
            num_clusters=self.num_clusters,
        )

    def forward(self, x):
        features = self.encoder(x)
        clusters = self.clustering_layer(features)
        output = self.decoder(features)
        return output, features, clusters


class FoldingModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.folding1 = nn.Sequential(
            nn.Linear(512 + 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

        self.folding2 = nn.Sequential(
            nn.Linear(512 + 3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )

    def forward(self, x, grid):
        cw_exp = x.expand(-1, grid.shape[1], -1)

        cat1 = torch.cat((cw_exp, grid), dim=2)
        folding_result1 = self.folding1(cat1)
        cat2 = torch.cat((cw_exp, folding_result1), dim=2)
        folding_result2 = self.folding2(cat2)
        return folding_result2


class FoldingNetBasicDecoder(nn.Module):
    def __init__(
        self,
        num_features,
        shape="plane",
        sphere_path="./sphere.npy",
        gaussian_path="./gaussian.npy",
        std=0.3,
    ):
        super().__init__()

        # initialise deembedding
        self.lin_features_len = 512
        self.num_features = num_features
        if self.num_features < self.lin_features_len:
            self.deembedding = nn.Linear(
                self.num_features, self.lin_features_len, bias=False
            )

        if shape == "plane":
            # make grid
            range_x = torch.linspace(-std, std, 45)
            range_y = torch.linspace(-std, std, 45)
            x_coor, y_coor = torch.meshgrid(range_x, range_y, indexing="ij")
            self.grid = torch.stack([x_coor, y_coor], axis=-1).float().reshape(-1, 2)
        elif shape == "sphere":
            self.grid = torch.tensor(np.load(sphere_path))
        elif self.shape == "gaussian":
            self.grid = torch.tensor(np.load(gaussian_path))

        # initialise folding module
        self.folding = FoldingModule()

    def forward(self, x):
        if self.num_features < self.lin_features_len:
            x = self.deembedding(x)
            x = x.unsqueeze(1)

        else:
            x = x.unsqueeze(1)
        grid = self.grid.cuda().unsqueeze(0).expand(x.shape[0], -1, -1)
        outputs = self.folding(x, grid)
        return outputs


class FoldNetDecoder(nn.Module):
    def __init__(
        self,
        num_features,
        shape="plane",
        sphere_path="./sphere.npy",
        gaussian_path="./gaussian.npy",
        std=0.3,
    ):
        super().__init__()
        self.m = 2025  # 45 * 45.
        self.std = std
        self.meshgrid = [[-std, std, 45], [-std, std, 45]]
        self.shape = shape
        if shape == "sphere":
            self.sphere = np.load(sphere_path)
        if shape == "gaussian":
            self.gaussian = np.load(gaussian_path)
        self.num_features = num_features
        if self.shape == "plane":
            self.folding1 = nn.Sequential(
                nn.Conv1d(512 + 2, 512, 1),
                nn.ReLU(),
                nn.Conv1d(512, 512, 1),
                nn.ReLU(),
                nn.Conv1d(512, 3, 1),
            )
        else:
            self.folding1 = nn.Sequential(
                nn.Conv1d(512 + 3, 512, 1),
                nn.ReLU(),
                nn.Conv1d(512, 512, 1),
                nn.ReLU(),
                nn.Conv1d(512, 3, 1),
            )

        self.folding2 = nn.Sequential(
            nn.Conv1d(512 + 3, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 3, 1),
        )

        self.lin_features_len = 512
        if self.num_features < self.lin_features_len:
            self.deembedding = nn.Linear(
                self.num_features, self.lin_features_len, bias=False
            )

    def build_grid(self, batch_size):
        if self.shape == "plane":
            x = np.linspace(*self.meshgrid[0])
            y = np.linspace(*self.meshgrid[1])
            points = np.array(list(itertools.product(x, y)))
        elif self.shape == "sphere":
            points = self.sphere
        elif self.shape == "gaussian":
            points = self.gaussian

        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)
        points = torch.tensor(points)
        return points.float()

    def forward(self, x):
        if self.num_features < self.lin_features_len:
            x = self.deembedding(x)
            x = x.unsqueeze(1)

        else:
            x = x.unsqueeze(1)

        x = x.transpose(1, 2).repeat(1, 1, self.m)
        points = self.build_grid(x.shape[0]).transpose(1, 2)
        if x.get_device() != -1:
            points = points.cuda(x.get_device())
        cat1 = torch.cat((x, points), dim=1)

        folding_result1 = self.folding1(cat1)
        cat2 = torch.cat((x, folding_result1), dim=1)
        folding_result2 = self.folding2(cat2)
        output = folding_result2.transpose(1, 2)
        return output


class DGCNNEncoder(nn.Module):
    def __init__(self, num_features, k=20):
        super().__init__()
        self.k = k
        self.num_features = num_features

        self.conv1 = nn.Sequential(
            nn.Conv2d(3 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.clustering = None
        self.lin_features_len = 512
        if (self.num_features < self.lin_features_len) or (
            self.num_features > self.lin_features_len
        ):
            self.flatten = Flatten()
            self.embedding = nn.Linear(
                self.lin_features_len, self.num_features, bias=False
            )

    def forward(self, x):
        x = x.transpose(2, 1)

        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x0 = self.conv5(x)
        x = x0.max(dim=-1, keepdim=False)[0]
        feat = x.unsqueeze(1)

        if (self.num_features < self.lin_features_len) or (
            self.num_features > self.lin_features_len
        ):
            x = self.flatten(feat)
            features = self.embedding(x)
        else:
            features = torch.reshape(torch.squeeze(feat), (batch_size, 512))

        return features


class FoldNetEncoder(nn.Module):
    def __init__(self, num_features, k):
        super().__init__()
        if k is None:
            self.k = 16
        else:
            self.k = k
        self.n = 2048
        self.num_features = num_features
        self.mlp1 = nn.Sequential(
            nn.Conv1d(12, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )
        self.linear1 = nn.Linear(64, 64)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.linear2 = nn.Linear(128, 128)
        self.conv2 = nn.Conv1d(128, 1024, 1)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
        )
        self.clustering = None
        self.lin_features_len = 512
        if (self.num_features < self.lin_features_len) or (
            self.num_features > self.lin_features_len
        ):
            self.embedding = nn.Linear(
                self.lin_features_len, self.num_features, bias=False
            )

    def graph_layer(self, x, idx):
        x = local_maxpool(x, idx)
        x = self.linear1(x)
        x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))
        x = local_maxpool(x, idx)
        x = self.linear2(x)
        x = x.transpose(2, 1)
        x = self.conv2(x)
        return x

    def forward(self, pts):
        pts = pts.transpose(2, 1)
        batch_size = pts.size(0)
        idx = knn(pts, k=self.k)
        x = local_cov(pts, idx)
        x = self.mlp1(x)
        x = self.graph_layer(x, idx)
        x = torch.max(x, 2, keepdim=True)[0]
        x = self.mlp2(x)
        feat = x.transpose(2, 1)
        if (self.num_features < self.lin_features_len) or (
            self.num_features > self.lin_features_len
        ):
            x = self.flatten(feat)
            features = self.embedding(x)
        else:
            features = torch.reshape(torch.squeeze(feat), (batch_size, 512))

        return features


class DGCNN(nn.Module):
    def __init__(self, num_features=2, k=20, emb_dims=512):
        super().__init__()
        self.num_features = num_features
        self.k = k
        self.emb_dims = emb_dims
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(self.emb_dims)
        self.dropout = 0.2

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.linear1 = nn.Linear(self.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=self.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=self.dropout)
        self.linear3 = nn.Linear(256, num_features)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class Flatten(nn.Module):
    def forward(self, input):
        """
        Note that input.size(0) is usually the batch size.
        So what it does is that given any input with input.size(0)
        number of batches,
        will flatten to be 1 * nb_elements.
        """
        batch_size = input.size(0)
        out = input.view(batch_size, -1)
        return out


__all__ = [
    "TrackAsuraTransformer",
    "TrackAsuraEncoderLayer",
    "TrackAsuraDecoderLayer",
    "TrackAsuraAttention",
    "TrackAsuraBias",
    "TrackAsuraRotaryPositionalEncoding",
    "DenseNet",
    "MitosisNet",
    "AttentionNet",
    "HybridAttentionDenseNet",
    "CloudAutoEncoder",
    "DGCNNEncoder",
    "FoldNetEncoder",
    "FoldNetDecoder",
    "FoldingNetBasicDecoder",
    "Flatten",
    "FoldingModule",
    "DGCNN",
    "DeepEmbeddedClustering",
    "ClusteringLayer",
]
