#  Copyright (c) 2022. Yuki Tatsunami
#  Licensed under the Apache License, Version 2.0 (the "License");

import math
from functools import partial

import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT
from timm.models.layers import Mlp, lecun_normal_, trunc_normal_
from timm.models.helpers import build_model_with_cfg, named_apply
from timm.models.registry import register_model
from torch import nn

#  Copyright (c) 2022. Yuki Tatsunami
#  Licensed under the Apache License, Version 2.0 (the "License");

from functools import partial
from typing import Tuple

import torch
from timm.models.layers import DropPath, Mlp, PatchEmbed as TimmPatchEmbed

from torch import nn, _assert, Tensor


class RNNIdentity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(RNNIdentity, self).__init__()

    def forward(self, x: Tensor) -> Tuple[Tensor, None]:
        return x, None


class RNNBase(nn.Module):

    def __init__(self, input_size, hidden_size=None,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True):
        super().__init__()
        self.rnn = RNNIdentity()

    def forward(self, x):
        B, H, W, C = x.shape
        x, _ = self.rnn(x.view(B, -1, C))
        return x.view(B, H, W, -1)


class RNN(RNNBase):

    def __init__(self, input_size, hidden_size=None,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True,
                 nonlinearity="tanh"):
        super().__init__(input_size, hidden_size, num_layers, bias, bidirectional)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True,
                          bias=bias, bidirectional=bidirectional, nonlinearity=nonlinearity)


class GRU(RNNBase):

    def __init__(self, input_size, hidden_size=None,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True):
        super().__init__(input_size, hidden_size, num_layers, bias, bidirectional)
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          bias=bias, bidirectional=bidirectional)


class LSTM(RNNBase):

    def __init__(self, input_size, hidden_size=None,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True):
        super().__init__(input_size, hidden_size, num_layers, bias, bidirectional)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                           bias=bias, bidirectional=bidirectional)


class RNN2DBase(nn.Module):

    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True,
                 union="cat", with_fc=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 2 * hidden_size if bidirectional else hidden_size
        self.union = union

        self.with_vertical = True
        self.with_horizontal = True
        self.with_fc = with_fc

        if with_fc:
            if union == "cat":
                self.fc = nn.Linear(2 * self.output_size, input_size)
            elif union == "add":
                self.fc = nn.Linear(self.output_size, input_size)
            elif union == "vertical":
                self.fc = nn.Linear(self.output_size, input_size)
                self.with_horizontal = False
            elif union == "horizontal":
                self.fc = nn.Linear(self.output_size, input_size)
                self.with_vertical = False
            else:
                raise ValueError("Unrecognized union: " + union)
        elif union == "cat":
            pass
            if 2 * self.output_size != input_size:
                raise ValueError(f"The output channel {2 * self.output_size} is different from the input channel {input_size}.")
        elif union == "add":
            pass
            if self.output_size != input_size:
                raise ValueError(f"The output channel {self.output_size} is different from the input channel {input_size}.")
        elif union == "vertical":
            if self.output_size != input_size:
                raise ValueError(f"The output channel {self.output_size} is different from the input channel {input_size}.")
            self.with_horizontal = False
        elif union == "horizontal":
            if self.output_size != input_size:
                raise ValueError(f"The output channel {self.output_size} is different from the input channel {input_size}.")
            self.with_vertical = False
        else:
            raise ValueError("Unrecognized union: " + union)

        self.rnn_v = RNNIdentity()
        self.rnn_h = RNNIdentity()

    def forward(self, x):
        B, H, W, C = x.shape

        if self.with_vertical:
            v = x.permute(0, 2, 1, 3)
            v = v.reshape(-1, H, C)
            v, _ = self.rnn_v(v)
            v = v.reshape(B, W, H, -1)
            v = v.permute(0, 2, 1, 3)

        if self.with_horizontal:
            h = x.reshape(-1, W, C)
            h, _ = self.rnn_h(h)
            h = h.reshape(B, H, W, -1)

        if self.with_vertical and self.with_horizontal:
            if self.union == "cat":
                x = torch.cat([v, h], dim=-1)
            else:
                x = v + h
        elif self.with_vertical:
            x = v
        elif self.with_horizontal:
            x = h

        if self.with_fc:
            x = self.fc(x)

        return x


class RNN2D(RNN2DBase):

    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True,
                 union="cat", with_fc=True, nonlinearity="tanh"):
        super().__init__(input_size, hidden_size, num_layers, bias, bidirectional, union, with_fc)
        if self.with_vertical:
            self.rnn_v = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional, nonlinearity=nonlinearity)
        if self.with_horizontal:
            self.rnn_h = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional, nonlinearity=nonlinearity)


class LSTM2D(RNN2DBase):

    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True,
                 union="cat", with_fc=True):
        super().__init__(input_size, hidden_size, num_layers, bias, bidirectional, union, with_fc)
        if self.with_vertical:
            self.rnn_v = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional)
        if self.with_horizontal:
            self.rnn_h = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional)


class GRU2D(RNN2DBase):

    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True,
                 union="cat", with_fc=True):
        super().__init__(input_size, hidden_size, num_layers, bias, bidirectional, union, with_fc)
        if self.with_vertical:
            self.rnn_v = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional)
        if self.with_horizontal:
            self.rnn_h = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional)


class VanillaSequencerBlock(nn.Module):
    def __init__(self, dim, hidden_size, mlp_ratio=3.0, rnn_layer=LSTM, mlp_layer=Mlp,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU,
                 num_layers=1, bidirectional=True, drop=0., drop_path=0.):
        super().__init__()
        channels_dim = int(mlp_ratio * dim)
        self.norm1 = norm_layer(dim)
        self.rnn_tokens = rnn_layer(dim, hidden_size, num_layers=num_layers, bidirectional=bidirectional)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.rnn_tokens(self.norm1(x)))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


class Sequencer2DBlock(nn.Module):
    def __init__(self, dim, hidden_size, mlp_ratio=3.0, rnn_layer=LSTM2D, mlp_layer=Mlp,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU,
                 num_layers=1, bidirectional=True, union="cat", with_fc=True,
                 drop=0., drop_path=0.):
        super().__init__()
        channels_dim = int(mlp_ratio * dim)
        self.norm1 = norm_layer(dim)
        self.rnn_tokens = rnn_layer(dim, hidden_size, num_layers=num_layers, bidirectional=bidirectional,
                                    union=union, with_fc=with_fc)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.rnn_tokens(self.norm1(x)))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


class PatchEmbed(TimmPatchEmbed):
    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        else:
            x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
        x = self.norm(x)
        return x


class Shuffle(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if self.training:
            B, H, W, C = x.shape
            r = torch.randperm(H * W)
            x = x.reshape(B, -1, C)
            x = x[:, r, :].reshape(B, H, W, -1)
        return x


class Downsample2D(nn.Module):
    def __init__(self, input_dim, output_dim, patch_size):
        super().__init__()
        self.down = nn.Conv2d(input_dim, output_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.down(x)
        x = x.permute(0, 2, 3, 1)
        return x



def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': DEFAULT_CROP_PCT, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.proj', 'classifier': 'head',
        **kwargs
    }


def _init_weights(module: nn.Module, name: str, head_bias: float = 0., flax=False):
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            if flax:
                # Flax defaults
                lecun_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.RNN, nn.GRU, nn.LSTM)):
        stdv = 1.0 / math.sqrt(module.hidden_size)
        for weight in module.parameters():
            nn.init.uniform_(weight, -stdv, stdv)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def get_stage(index, layers, patch_sizes, embed_dims, hidden_sizes, mlp_ratios, block_layer, rnn_layer, mlp_layer,
              norm_layer, act_layer, num_layers, bidirectional, drop=0., drop_path_rate=0., **kwargs):
    assert len(layers) == len(patch_sizes) == len(embed_dims) == len(hidden_sizes) == len(mlp_ratios)
    blocks = []
    for block_idx in range(layers[index]):
        drop_path = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(block_layer(embed_dims[index], hidden_sizes[index], mlp_ratio=mlp_ratios[index],
                                  rnn_layer=rnn_layer, mlp_layer=mlp_layer, norm_layer=norm_layer,
                                  act_layer=act_layer, num_layers=num_layers,
                                  bidirectional=bidirectional, drop=drop, drop_path=drop_path))

    if index < len(embed_dims) - 1:
        blocks.append(Downsample2D(embed_dims[index], embed_dims[index + 1], patch_sizes[index + 1]))

    blocks = nn.Sequential(*blocks)
    return blocks


class VanillaSequencer(nn.Module):
    def __init__(
            self,
            num_classes=1000,
            img_size=224,
            in_chans=3,
            layers=[4, 3, 8, 3],
            patch_sizes=[14, 1, 1, 1],
            embed_dims=[384, 384, 384, 384],
            hidden_sizes=[192, 192, 192, 192],
            mlp_ratios=[3.0, 3.0, 3.0, 3.0],
            block_layer=VanillaSequencerBlock,
            rnn_layer=LSTM,
            mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            num_rnn_layers=1,
            bidirectional=True,
            shuffle=False,
            ape=False,
            drop_rate=0.,
            drop_path_rate=0.,
            nlhb=False,
            stem_norm=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dims[0]  # num_features for consistency with other models
        self.embed_dims = embed_dims
        self.stem = PatchEmbed(
            img_size=img_size, patch_size=patch_sizes[0], in_chans=in_chans,
            embed_dim=embed_dims[0], norm_layer=norm_layer if stem_norm else None,
            flatten=False)
        self.shuffle = shuffle

        if self.shuffle:
            self.shuffle_patches = Shuffle()

        # absolute position embedding
        self.ape = ape
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.stem.grid_size[0], self.stem.grid_size[1], embed_dims[0]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.blocks = nn.Sequential(*[
            get_stage(
                i, layers, patch_sizes, embed_dims, hidden_sizes, mlp_ratios, block_layer=block_layer,
                rnn_layer=rnn_layer, mlp_layer=mlp_layer, norm_layer=norm_layer, act_layer=act_layer,
                num_layers=num_rnn_layers, bidirectional=bidirectional, drop=drop_rate, drop_path_rate=drop_path_rate,
            )
            for i, _ in enumerate(embed_dims)])

        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(nlhb=nlhb)

    def init_weights(self, nlhb=False):
        head_bias = -math.log(self.num_classes) if nlhb else 0.
        named_apply(partial(_init_weights, head_bias=head_bias), module=self)  # depth-first

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.stem(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        if self.shuffle:
            x = self.shuffle_patches(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=(1, 2))
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def checkpoint_filter_fn(state_dict, model):
    return state_dict


default_cfgs = dict(
    v_sequencer_s=_cfg(),
    v_sequencer_s_h=_cfg(),
    v_sequencer_s_pe=_cfg(),
)


def _create_vanilla_sequencer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for VanillaSequencer models.')

    model = build_model_with_cfg(
        VanillaSequencer, variant, pretrained,
        default_cfg=default_cfgs[variant],
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)
    return model


@register_model
def v_sequencer_s(pretrained=False, **kwargs):
    model_args = dict(
        layers=[4, 3, 8, 3],
        patch_sizes=[14, 1, 1, 1],
        embed_dims=[384, 384, 384, 384],
        hidden_sizes=[192, 192, 192, 192],
        mlp_ratios=[3.0, 3.0, 3.0, 3.0],
        rnn_layer=LSTM,
        bidirectional=True,
        shuffle=False,
        ape=False,
        **kwargs)
    model = _create_vanilla_sequencer('v_sequencer_s', pretrained=pretrained, **model_args)
    return model


@register_model
def v_sequencer_s_h(pretrained=False, **kwargs):
    model_args = dict(
        layers=[4, 3, 8, 3],
        patch_sizes=[7, 2, 1, 1],
        embed_dims=[192, 384, 384, 384],
        hidden_sizes=[96, 192, 192, 192],
        mlp_ratios=[3.0, 3.0, 3.0, 3.0],
        rnn_layer=LSTM,
        bidirectional=True,
        shuffle=False,
        ape=False,
        **kwargs)
    model = _create_vanilla_sequencer('v_sequencer_s_h', pretrained=pretrained, **model_args)
    return model


@register_model
def v_sequencer_s_pe(pretrained=False, **kwargs):
    model_args = dict(
        layers=[4, 3, 8, 3],
        patch_sizes=[14, 1, 1, 1],
        embed_dims=[384, 384, 384, 384],
        hidden_sizes=[192, 192, 192, 192],
        mlp_ratios=[3.0, 3.0, 3.0, 3.0],
        rnn_layer=LSTM,
        bidirectional=True,
        shuffle=False,
        ape=True,
        **kwargs)
    model = _create_vanilla_sequencer('v_sequencer_s_pe', pretrained=pretrained, **model_args)
    return model



if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    from thop import profile
    from pytorch_model_summary import summary
    import time

    print(torch.__version__)
    # net = swinFocus_tiny_patch4_window7_224().cuda()
    # net = v_sequencer_s(num_classes=3, img_size=224).cuda()
    net = VanillaSequencer(
        layers=[4, 3, 8, 3],
        patch_sizes=[14, 1, 1, 1],
        embed_dims=[384, 384, 384, 384],
        hidden_sizes=[192, 192, 192, 192],
        mlp_ratios=[3.0, 3.0, 3.0, 3.0],
        rnn_layer=LSTM,
        bidirectional=True,
        shuffle=False,
        ape=True,
        num_classes=96, img_size=224
    ).cuda()

    import torchsummary

    # torchsummary.summary(net)
    print(net)
    image = torch.rand(1, 3, 224, 224).cuda()
    # time_step=torch.tensor([999] * 1, device="cuda")
    # f, p = get_model_complexity_info(net, image, as_strings=True, print_per_layer_stat=False, verbose=False)
    # f, p = profile(net, inputs=(image, time_step))

    f, p = profile(net, inputs=(image,))
    # f, p = summary(net, (image, time_step))
    print('flops:%f' % f)
    print('params:%f' % p)
    print('flops: %.1f G, params: %.1f M' % (f / 1e9, p / 1e6))

    s = time.time()
    with torch.no_grad():
        out = net(image, )

    print('infer_time:', time.time() - s)
    print("FPS:%f" % (1 / (time.time() - s)))

    print(out.shape)