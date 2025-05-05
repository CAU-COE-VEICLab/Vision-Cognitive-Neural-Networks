# --------------------------------------------------------
# The potential of cognitive-inspired neural network modeling framework for computer vision processing tasks
# Licensed under The MIT License [see LICENSE for details]
# Written by Guorun Li
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional
import math
from .memory import Memory, LayerNorm, UMA, MultiScalePeception, MemoryDownSampling


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"Droprate={self.drop_prob}"


class PatchEmbed(nn.Module):
    def __init__(self,
                 patch_size=4,
                 embed_conv=3,
                 in_c=3,
                 embed_dim=96,
                 norm_layer=None,
                 ab_patch_norm_name='BN'):
        super().__init__()
        stem = [nn.Conv2d(in_c, embed_dim, kernel_size=embed_conv, stride=2, padding=3 if embed_conv == 7 else 1,
                          bias=False), nn.BatchNorm2d(embed_dim), nn.ReLU(True)]
        stem.append(nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=2))
        self.conv = nn.Sequential(*stem)
        if ab_patch_norm_name == 'BN':
            self.norm = nn.BatchNorm2d(embed_dim)
        else:
            self.norm = LayerNorm(embed_dim, eps=1e-5, data_format="channels_first") if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x



class OverlapPatchEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # x
        self.proj = nn.Conv2d(dim, dim * 2, kernel_size=2, stride=2, bias=False)
        self.norm = nn.BatchNorm2d(dim * 2) 
    def forward(self, x):
        """
        x: B, C,H,W
        """
        x = self.proj(x)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.pos = nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, C, H, W = x.shape
        # skip = x
        # x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x + self.pos(x))
        x = self.drop(x)
        # x = x + self.act(self.pos(x))
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

    def extra_repr(self) -> str:
        return f"dim={self.normalized_shape}"


class VisionCognitiveNeuralUnit(nn.Module):
    def __init__(self,
                 dim,
                 memory_dim,
                 num_scale=4,
                 use_bias=True,
                 proj_drop=0.,
                 ab_norm_attn=False,
                 ab_norm_ltm=False,
                 ):
        super().__init__()
        self.ab_norm_attn=ab_norm_attn
        self.ab_norm_ltm=ab_norm_ltm
        self.memory_dim=memory_dim
        self.num_scale=num_scale

        self.multi_scale_peception = MultiScalePeception(
            dim=dim,
            num_scale=num_scale,
            use_bias=use_bias,
        )
        self.t2d_memory_attention = Memory(
            dim=dim,
            memory_dim=memory_dim,
            use_bias=use_bias,
            proj_drop=proj_drop,
            ab_norm_attn=ab_norm_attn,
            ab_norm_ltm=ab_norm_ltm,
            model_style='conv',  # conv or trans
        )

    def forward(self, x, ltm):
        perception_attn = self.multi_scale_peception(x)
        memory_attn, ltm = self.t2d_memory_attention(perception=perception_attn, ltm=ltm)
        # return [B,C,H,W]
        return memory_attn, ltm

    def extra_repr(self) -> str:
        return f"num_scale={self.num_scale}," \
               f"ab_norm_attn={self.ab_norm_attn}, ab_norm_ltm={self.ab_norm_ltm}," \
               f"memory_dim={self.memory_dim}"



class VCNUsBlock(nn.Module):
    def __init__(self,
                 dim,
                 memory_dim,
                 num_scale=4,
                 mlp_ratio=4.,
                 use_bias=True,
                 drop=0.,
                 drop_path=0.,
                 layerscale_value=1e-4,
                 use_layerscale=False,
                 ab_norm_attn=False,
                 ab_norm_ltm=False,
                 ):
        super().__init__()
        self.dim = dim
        self.memory_dim = memory_dim
        self.mlp_ratio = mlp_ratio
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = nn.BatchNorm2d(dim) 
        self.vcnu = VisionCognitiveNeuralUnit(
            dim=dim,
            memory_dim=memory_dim,
            num_scale=num_scale,
            use_bias=use_bias,
            proj_drop=drop,
            ab_norm_attn=ab_norm_attn,
            ab_norm_ltm=ab_norm_ltm,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.gamma_1 = 1.0
        if use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            

    def forward(self, x, memory):
        # x: [B,C,H,W]
        B, C, H, W = x.shape
        x_vcnus = self.norm1(x)
        x_vcnus, memory = self.vcnu(x_vcnus, memory)
        if isinstance(self.gamma_1, nn.Parameter):
            x = x + self.drop_path(self.gamma_1.unsqueeze(-1).unsqueeze(-1) * x_vcnus)
        else:
            x = x + self.drop_path(self.gamma_1 * x_vcnus)
        return x, memory

class BasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 memory_dim,
                 depth,
                 num_scale=4,
                 mlp_ratio=4.,
                 use_bias=True,
                 drop=0.,
                 drop_path=0.,
                 use_layerscale=False,
                 layerscale_value=1e-4,
                 downsample=None,
                 use_checkpoint=False,
                 ab_norm_attn=False,
                 ab_norm_ltm=False,
                 ):
        super().__init__()
        self.dim = dim
        self.memory_dim = memory_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            VCNUsBlock(
                dim=dim,
                memory_dim=memory_dim,
                num_scale=num_scale,
                mlp_ratio=mlp_ratio,
                use_bias=use_bias,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                use_layerscale=use_layerscale,
                layerscale_value=layerscale_value,
                ab_norm_attn=ab_norm_attn,
                ab_norm_ltm=ab_norm_ltm,
            )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim)
            self.memory_downsample = MemoryDownSampling(model_style='conv')
        else:
            self.downsample = None

    def forward(self, x, memory):
        # x[b,c,h,w]
        for blk in self.blocks:
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x, memory = checkpoint.checkpoint(blk, x, memory)
            else:
                x, memory = blk(x, memory)
        if self.downsample is not None:
            x = self.downsample(x)
            memory = self.memory_downsample(memory, (x.shape[2], x.shape[3]))
        return x, memory

    def extra_repr(self) -> str:
        return f"dim={self.dim}, memory_dim={self.memory_dim}, depth={self.depth}"


class VisionCognitiveNeuralUnits(nn.Module):
    def __init__(self,
                 filter_strategy1 = 23,
                 filter_strategy2 = 7,
                 use_layerscale=False,
                 layerscale_value=1e-4,
                 image_size=224,
                 pretrain_image_size=224,
                 patch_size=4,
                 embed_conv=3,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=96,
                 depths=(3, 3, 9, 3),
                 mlp_ratio=4.,
                 use_bias=True,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # memory
        num_scale = 4  
        ab_norm_attn = True
        ab_norm_ltm = False
        self.model_style = 'conv'
        self.ab_norm_attn = ab_norm_attn
        self.ab_norm_ltm = ab_norm_ltm
        self.pretrain_image_size = pretrain_image_size
        self.fftscale = self.patch_size = patch_size
        self.filter_strategy1 = filter_strategy1
        self.filter_strategy2 = filter_strategy2
        self.uma = UMA(filter_strategy1=self.filter_strategy1,
                       filter_strategy2=self.filter_strategy2,
                       fftscale=patch_size,
                       )
        self.filter, self.downfilter, self.memory_dim = self.uma.filter, self.uma.downfilter, self.uma.memory_dim


        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            embed_conv=embed_conv,
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        self.pos_drop = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        image_size_layer = image_size // patch_size
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layers = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                memory_dim=int(self.memory_dim),
                depth=depths[i_layer],
                num_scale=num_scale,
                mlp_ratio=self.mlp_ratio,
                use_bias=use_bias,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                
                use_layerscale=use_layerscale,
                layerscale_value=layerscale_value,
                downsample=OverlapPatchEmbed if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                ab_norm_attn=ab_norm_attn,
                ab_norm_ltm=ab_norm_ltm,
            )
            self.layers.append(layers)
            image_size_layer = image_size_layer // 2
        self.layers[-1].blocks[-1].vcnu.t2d_memory_attention.update_ltm = nn.Identity()
        self.layers[-1].blocks[-1].vcnu.t2d_memory_attention.norm_ltm = nn.Identity()

        self.normhead = nn.BatchNorm2d(self.num_features) 
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)  
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def construct_memory(self, x):
        _, _, H, W = x.shape
        # [B, C, H, W]
        memorystream = self.uma(x)
        
        if self.model_style == 'conv':
            assert memorystream.shape[
                       1] == self.memory_dim, f"fast fourier transform error! pleace check fastFourierTrans() function. now the memory dim is {memorystream.shape[1]}, should be {self.memory_dim}!"
        else:
            assert memorystream.shape[
                       -1] == self.memory_dim, f"fast fourier transform error! pleace check fastFourierTrans() function. now the memory dim is {memorystream.shape[1]}, should be {self.memory_dim}!"
        return memorystream

    def forward(self, x):
        _, _, H, W = x.shape
        memory = self.construct_memory(x)

        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x, memory = layer(x, memory)

        x = self.normhead(x)
        x = self.avgpool(x.flatten(-2, -1))
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


def vcnu_small(num_classes: int = 1000, **kwargs):

    model = VisionCognitiveNeuralUnit(
        use_layerscale=True,
        layerscale_value=1e-4,

        image_size=224,
        embed_conv=7,
        num_classes=num_classes,
        embed_dim=96,
        depths=[ 3, 3, 27, 3],
        use_checkpoint=True,
        drop_path_rate=0.2,
        **kwargs)
    return model


if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    from thop import profile
    from pytorch_model_summary import summary
    import time

    print(torch.__version__)
    net = vcnu_small(3).cuda()

    print(net)
    image = torch.rand(1, 3, 224, 224).cuda()

    f, p = profile(net, inputs=(image,))
    print('flops:%f' % f)
    print('params:%f' % p)
    print('flops: %.1f G, params: %.1f M' % (f / 1e9, p / 1e6))

    s = time.time()
    with torch.no_grad():
        out = net(image, )

    print('infer_time:', time.time() - s)
    print("FPS:%f" % (1 / (time.time() - s)))

    print(out.shape)


