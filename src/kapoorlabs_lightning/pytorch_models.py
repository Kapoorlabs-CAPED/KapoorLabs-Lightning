import itertools
from typing import Literal
import numpy as np
import os
import torch
import torch.nn as nn
import logging
import math
import torch.nn.functional as F
from torch import Tensor
from collections import OrderedDict
from typing import List
import torch.nn.init as init
from torch.utils.data import Dataset
from tifffile import imwrite

logger = logging.getLogger(__name__)


# ------------------------------- Dense Layer ------------------------------- #

class DenseLayer(nn.Module):
    """
    A single DenseNet-style layer for 1D input with optional bottleneck.
    Applies (BN → ReLU → [1x1 Conv])? → BN → ReLU → Conv(kernel_size).
    """
    def __init__(self, input_channels, growth_rate, bottleneck_size, kernel_size):
        super().__init__()
        self.use_bottleneck = bottleneck_size > 0

        if self.use_bottleneck:
            bottleneck_channels = growth_rate * bottleneck_size
            self.bn_bottleneck = nn.BatchNorm1d(input_channels)
            self.act_bottleneck = nn.ReLU(inplace=True)
            self.conv_bottleneck = nn.Conv1d(
                input_channels, bottleneck_channels, kernel_size=1, stride=1, bias=False
            )
            conv_input_channels = bottleneck_channels
        else:
            conv_input_channels = input_channels

        self.bn_main = nn.BatchNorm1d(conv_input_channels)
        self.act_main = nn.ReLU(inplace=True)
        self.conv_main = nn.Conv1d(
            conv_input_channels,
            growth_rate,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x):
        if self.use_bottleneck:
            x = self.conv_bottleneck(self.act_bottleneck(self.bn_bottleneck(x)))
        x = self.conv_main(self.act_main(self.bn_main(x)))
        return x

# ------------------------------ Dense Block ------------------------------ #

class DenseBlock(nn.ModuleDict):
    """
    A block of DenseLayers with feature concatenation.
    """
    def __init__(self, num_layers, input_channels, growth_rate, kernel_size, bottleneck_size):
        super().__init__()
        for i in range(num_layers):
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
            out = layer(torch.cat(layer_outputs, dim=1))
            layer_outputs.append(out)
        return torch.cat(layer_outputs, dim=1)

# ---------------------------- Transition Block ---------------------------- #

class TransitionBlock(nn.Module):
    """
    Transition block for downsampling: BN → ReLU → 1x1 Conv → AvgPool.
    """
    def __init__(self, input_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(input_channels, out_channels, kernel_size=1, stride=1)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(self.conv(self.act(self.bn(x))))
        return x

class AttentionPool1d(nn.Module):
    """
    Attention-based pooling over 1-D sequences with a learnable [CLS] token.
    Input : [B, T, C]  →  Output : [B, C_out]   (C must be divisible by num_heads)
    """
    def __init__(self, seq_len: int, embed_dim: int,
                 num_heads: int = 8, output_dim: int | None = None):
        super().__init__()

        # --- safety check --------------------------------------------------- #
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        # -------------------------------------------------------------------- #

        self.pos_embed  = nn.Parameter(torch.randn(seq_len + 1, embed_dim) / embed_dim**0.5)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.qkv_proj   = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.output_proj = nn.Linear(embed_dim, output_dim or embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: [B, T, C]
        B, T, C = x.shape

        cls = self.cls_token.expand(B, 1, C)
        x   = torch.cat([cls, x], dim=1) + self.pos_embed[: T + 1]   # [B, T+1, C]

        qkv = self.qkv_proj(x).reshape(B, T + 1, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                                  # [B, T+1, heads, d]

        q = q[:, 0:1].transpose(1, 2)                                # [B, heads, 1, d]
        k = k.transpose(1, 2)                                        # [B, heads, T+1, d]
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / self.head_dim**0.5
        attn = attn.softmax(dim=-1)

        pooled = (attn @ v).squeeze(2).transpose(1, 2).reshape(B, C) # [B, C]
        return self.output_proj(pooled)

# --------------------------------------------------------------------------- #
# InceptionNet                                                                #
# --------------------------------------------------------------------------- #
class InceptionNet(nn.Module):
    """
    Temporal DenseNet encoder + AttentionPool1d classifier.
    Accepts either an int or a tuple for `block_config`.
    Input  : [B, C, T]   (e.g. T = 25)      Output : logits [B, num_classes]
    """
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        growth_rate: int = 32,
        block_config = (6),          # can be int or tuple
        num_init_features: int = 32,
        bottleneck_size: int = 4,
        kernel_size: int = 3,
        attn_heads: int = 8,
        seq_len: int = 25,
    ):
        super().__init__()

        # --------------------- handle flexible args ------------------------ #
        if isinstance(block_config, int):
            block_config = (block_config,)
        if not isinstance(seq_len, int):
            seq_len = seq_len[0]
        # ------------------------------------------------------------------- #

        # --------------------------- stem ---------------------------------- #
        self.conv0 = nn.Sequential(
            nn.Conv1d(input_channels, num_init_features,
                      kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(num_init_features),
            nn.ReLU(inplace=True),
        )

        # ------------------ dense/transition stages ------------------------ #
        channels = num_init_features
        self.stages = nn.ModuleList()
        self.trans  = nn.ModuleList()
        n_transitions = len(block_config) - 1
        seq_len_attn = max(1, seq_len // (2 ** n_transitions))
        for i, n_layers in enumerate(block_config):
            self.stages.append(
                DenseBlock(
                    num_layers=n_layers,
                    input_channels=channels,
                    growth_rate=growth_rate,
                    kernel_size=kernel_size,
                    bottleneck_size=bottleneck_size,
                )
            )
            channels += n_layers * growth_rate

            if i != len(block_config) - 1:
                next_channels = channels // 2
                self.trans.append(TransitionBlock(channels, next_channels))
                channels = next_channels
            else:
                self.trans.append(None)

        # ------------- align channels to be divisible by heads ------------ #
        if channels % attn_heads != 0:
            aligned = ((channels + attn_heads - 1) // attn_heads) * attn_heads
            self.align_proj = nn.Conv1d(channels, aligned, kernel_size=1, bias=False)
            channels = aligned
        else:
            self.align_proj = nn.Identity()
        # ------------------------------------------------------------------- #

        # ------------------------- attention head -------------------------- #
        self.attn_pool = AttentionPool1d(
            seq_len=seq_len_attn,
            embed_dim=channels,
            num_heads=attn_heads,
        )
        self.classifier = nn.Linear(channels, num_classes)

    # ----------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: [B, C, T]
        x = self.conv0(x)
        for block, trans in zip(self.stages, self.trans):
            x = block(x)
            if trans is not None:
                x = trans(x)

        x = self.align_proj(x)          # [B, C_aligned, T']
        x = self.attn_pool(x.permute(0, 2, 1))   # → [B, C]
        return self.classifier(x)



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





class DenseVollNet(nn.Module):
    def __init__(
        self,
        input_shape,
        categories: int,
        box_vector: int,
        nboxes: int = 1,
        start_kernel: int = 7,
        mid_kernel: int = 3,
        startfilter: int = 64,
        growth_rate: int = 32, 
        bn_size: int = 4, 
        depth: dict = {'depth_0': 6, 'depth_1': 12, 'depth_2': 24, 'depth_3': 16},
        pool_first = True
    ):
        super().__init__()
        
        # Top module
        stage_number = len(depth)
        if pool_first:
          last_conv_factor = 2 ** (stage_number)
        else:
          last_conv_factor = 2 ** (stage_number - 1)    

        self.input_channels = input_shape[0]

        # DenseNet initialization
        print('Densenet 3D initialization')
        self.densenet = DenseNet3D(
            input_channels = self.input_channels,
            block_config=depth,
            startfilter=startfilter,
            start_kernel=start_kernel,
            mid_kernel=mid_kernel,
            growth_rate=growth_rate,
            bn_size=bn_size,
            pool_first = pool_first
        )

        # Bottom Part
        print('Model bottom initialization')
        self.conv3d_main = nn.Conv3d(
            in_channels=self.densenet.final_features,
            out_channels=categories + nboxes * box_vector,
            kernel_size=mid_kernel,
            padding="same"
        )
        self.batch_norm_main = nn.BatchNorm3d(categories + nboxes * box_vector)
        self.relu_main = nn.ReLU()

        self.conv3d_cat = nn.Conv3d(
            in_channels=categories,
            out_channels=categories,
            kernel_size=(
                input_shape[1] // last_conv_factor ,
                input_shape[2] // last_conv_factor ,
                input_shape[3] // last_conv_factor ,
            ),
            padding='valid'
        )
        self.conv3d_box = nn.Conv3d(
            in_channels=box_vector,
            out_channels=box_vector,
            kernel_size=(
                input_shape[1] // last_conv_factor ,
                input_shape[2] // last_conv_factor ,
                input_shape[3] // last_conv_factor ,
            ),
            padding='valid'
        )


    def forward(self, x):
        # Top DenseNet-like Block
        x = self.densenet(x)

        # Bottom Part - First convolution
        x = self.conv3d_main(x)
        x = self.batch_norm_main(x)
        x = self.relu_main(x)
        # Split into categories and boxes
        input_cat = x[:, :self.conv3d_cat.in_channels, :, :, :]
        input_box = x[:, self.conv3d_cat.in_channels:, :, :, :]
        # Apply convolutions
        output_cat = self.conv3d_cat(input_cat)
        output_box = self.conv3d_box(input_box)
        # Apply activations
        output_cat = F.softmax(output_cat, dim=1)
        output_box = torch.sigmoid(output_box)

        # Concatenate along the channel dimension
        outputs = torch.cat([output_cat, output_box], dim=1)

        return outputs

class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, mid_kernel = 3):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU())
        self.add_module(
            'conv1',
            nn.Conv3d(num_input_features,
                      bn_size * growth_rate,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU())
        self.add_module(
            'conv2',
            nn.Conv3d(bn_size * growth_rate,
                      growth_rate,
                      kernel_size=mid_kernel,
                      stride=1,
                      padding=1,
                      bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate, mid_kernel = 3):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate, mid_kernel=mid_kernel)
            self.add_module('denselayer{}'.format(i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU())
        self.add_module(
            'conv',
            nn.Conv3d(num_input_features,
                      num_output_features,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet3D(nn.Module):
    def __init__(self,input_channels, block_config: dict, startfilter,  start_kernel, 
                 mid_kernel, growth_rate=32, bn_size=4,
                 drop_rate=0, pool_first = True):
        super().__init__()
        if isinstance(block_config, tuple):
            block_config = {f'block_{i+1}': num_layers for i, num_layers in enumerate(block_config)}

        self.features = [('conv1',
                          nn.Conv3d(input_channels,
                                    startfilter,
                                    kernel_size=(start_kernel, start_kernel, start_kernel),
                                    padding='same')),
                         ('norm1', nn.BatchNorm3d(startfilter)),
                         ('relu1', nn.ReLU()),
                         ('pool1', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)) if pool_first else 
                         ('identity1', nn.Identity())
                          ]
        

        self.features = nn.Sequential(OrderedDict(self.features))
        num_features = startfilter
        for i, num_layers in enumerate(block_config.values()):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate,
                                mid_kernel = mid_kernel)
            self.features.add_module('denseblock{}'.format(i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition{}'.format(i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        self.final_activation = nn.ReLU() 

        self.final_features = num_features

    def forward(self, x):
        features = self.features(x)
        x = self.final_activation(features)
        return x






__all__ = [
    "DenseNet",
    "MitosisNet",
    "AttentionNet",
    "InceptionNet",
    "AttentionPool1d"
]
