# --------------------------------------------------------
# VisionCNUnit
# Copyright (c) 2024 CAU
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
from timm.models.layers import to_2tuple

class MultiScalePeception(nn.Module):
    def __init__(self,
                 dim,
                 num_scale=4,
                 use_bias=True,
                 ):
        super().__init__()
        # multi-scale perception
        self.num_scale = num_scale
        self.activation = nn.GELU()
        self.linear_map = nn.Conv2d(dim, dim, 1, bias=use_bias)
        if self.num_scale > 1:
            for i in range(self.num_scale):
                multiscale_conv = nn.Conv2d(dim // self.num_scale, dim // self.num_scale, kernel_size=(3 + i * 2),
                                            padding=(1 + i), stride=1, groups=dim // self.num_scale)
                setattr(self, f"multiscale_conv_{i + 1}", multiscale_conv)
    def forward(self, x):
        # [B,C,H,W]
        B, C, H, W = x.shape
        # memory block
        semantic_feature = self.linear_map(x)
        if self.num_scale > 1:
            semantic_feature = semantic_feature.reshape(B, self.num_scale, C // self.num_scale, H, W).permute(1, 0, 2,
                                                                                                              3, 4)
            for i in range(self.num_scale):
                multiscale_conv = getattr(self, f"multiscale_conv_{i + 1}")
                semantic_feature_i = semantic_feature[i]
                semantic_feature_i = multiscale_conv(semantic_feature_i)
                if i == 0:
                    semantic_feature_out = semantic_feature_i
                else:
                    semantic_feature_out = torch.cat([semantic_feature_out, semantic_feature_i], 1)
        else:
            semantic_feature_out = semantic_feature
        # return
        # trans style [B, L, C]
        # conv style [B,C,H,W]
        return semantic_feature_out

    

class MemoryDownSampling(nn.Module):
    def __init__(self, input_resolution=None, model_style='trans', ):
        super().__init__()
        # memory
        self.input_resolution = to_2tuple(input_resolution) if isinstance(input_resolution, int) or isinstance(input_resolution, tuple)\
            else None
        self.model_style = model_style
        self.memory_pooling = torch.nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, memory, hw_shape=None):
        # input (N, C, H, W)
        # memory (N, H, W, C)
        H, W = self.input_resolution if self.input_resolution is not None else hw_shape
        B = memory.shape[0]
        if self.model_style == 'conv':
            # [B, C, H, W]
            memory_dim = memory.shape[1]
            memory = self.memory_pooling(memory)
        else:
            # [B, H, W, C] OR [B, L, C]
            memory_dim = memory.shape[-1]
            if len(memory.shape) == 4:
                memory = self.memory_pooling(memory.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()
            elif len(memory.shape) == 3:
                memory = self.memory_pooling(memory.view(B, H, W, memory_dim).permute(0, 3, 1, 2)).flatten(2).transpose(
                    1, 2).contiguous()
            else:
                raise ValueError(f"Check your memory tensor! the size of memory is {memory.shape}")
        return memory

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

class MemoryIdentity(nn.Module):
    def __init__(self,
                 ):
        super().__init__()

    def forward(self, perception, ltm):
        return perception, ltm

class Memory(nn.Module):
    def __init__(self,
                 dim,
                 memory_dim,
                 use_bias=True,
                 proj_drop=0.,
                 ab_norm_attn=False,
                 ab_norm_ltm=False,
                 model_style='conv',  # conv or trans
                 ):
        super().__init__()
        self.model_style = model_style
        self.ab_norm_ltm_ = ab_norm_ltm  # update ltm
        self.ab_norm_attn_ = ab_norm_attn  # create attention

        self.dim = dim
        self.memory_dim = memory_dim

        self.activation = nn.GELU()
        self.wm = nn.Conv2d(memory_dim, dim, 1, padding=0, bias=use_bias) if self.model_style == 'conv' else nn.Linear(memory_dim, dim, bias=use_bias)
        self.attn = nn.Conv2d(2 * dim, dim, 1, bias=use_bias) if self.model_style == 'conv' else nn.Linear(2 * dim, dim, bias=use_bias)
        self.update_ltm = nn.Conv2d(dim, memory_dim, 1, bias=use_bias) if self.model_style == 'conv' else nn.Linear(dim, memory_dim, bias=use_bias)

        # drop and norm
        if self.ab_norm_attn_:
            if self.model_style == 'conv':
                self.norm_attn = nn.BatchNorm2d(2 * dim)
            else:
                self.norm_attn = nn.LayerNorm(2 * dim)
        else: self.norm_attn = nn.Identity()

        if self.ab_norm_ltm_:
            if self.model_style == 'conv':
                self.norm_ltm = nn.BatchNorm2d(memory_dim)
            else:
                self.norm_ltm = nn.LayerNorm(memory_dim)
        else: self.norm_ltm = nn.Identity()

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, perception, ltm):
        # transformer style [B, L, C] or [B, H, W, C]
        # convolution style [B, C, H, W]
        working_memory = self.wm(ltm)
        if self.model_style == 'conv':
            working_memory = torch.cat([working_memory, perception], dim=1)
        else:
            working_memory = torch.cat([working_memory, perception], dim=-1)
        attention = self.attn(self.activation(self.norm_attn(working_memory)))

        if isinstance(self.update_ltm, nn.Identity):
            ltm = ltm
        else:
            ltm = self.activation(self.norm_ltm(self.update_ltm(attention) + ltm))


        attention = self.proj_drop(attention)
        # return
        # trans style [B, L, C]
        # conv style [B,C,H,W]
        return attention, ltm

    def extra_repr(self) -> str:
        return f"dim={self.dim}, memory_dim={self.memory_dim}, model_style={self.model_style}, " \
               f"ab_norm_attn_={self.ab_norm_attn_}, ab_norm_ltm_={self.ab_norm_ltm_},"



class UMA(nn.Module):
    def __init__(self,
                 filter_strategy1=23,
                 filter_strategy2=7,
                 fftscale=4,
                 ):
        super().__init__()
        self.filter_strategy1 = filter_strategy1
        self.filter_strategy2 = filter_strategy2
        self.ablation_strategy = 'UMA'  # strategy1, strategy2, UMA
        self.use_sequencefunc = 'linear'  # statistic  fibonacci , linear
        self.fftscale = fftscale
        self.filter, self.downfilter, self.memory_dim = self.setfilter(ablation_strategy=self.ablation_strategy,
                                                                       use_sequencefunc=self.use_sequencefunc,
                                                                       threshold_strategy1=self.filter_strategy1,
                                                                       threshold_strategy2=self.filter_strategy2,
                                                                       scale=self.fftscale)

    def fibonacci(self, number):
        fib_sequence = [1, 1]
        fib_filter = [[1, 1], [2, 1]]
        for i in range(2, number):
            next_fib = fib_sequence[i - 1] + fib_sequence[i - 2]
            tmp = [2 * next_fib, next_fib]
            fib_sequence.append(next_fib)
            fib_filter.append(tmp)
        return fib_filter

    def setStatisticFilter(self, ablation_strategy, threshold_strategy1, threshold_strategy2, scale=4):
        fib_sequence_strategy1 = []
        fib_sequence_strategy2 = []
        next_index = 1
        tmp_index = 0
        while tmp_index <= threshold_strategy1:
            if next_index <= 3:
                tmp = [2 * next_index, next_index]
                tmp_index = next_index
                if tmp_index > threshold_strategy1:
                    break
            else:
                x = 2*next_index-3
                tmp = [2 * x, x]
                tmp_index = x
                if tmp_index > threshold_strategy1:
                    break
            fib_sequence_strategy1.append(tmp)
            next_index += 1

        next_index -= 1
        number_strategy1 = next_index

        next_index = 1
        tmp_index = 0
        while tmp_index <= threshold_strategy2:
            if next_index <= 3:
                tmp = [2 * next_index, next_index]
                tmp_index = next_index
                if tmp_index > threshold_strategy2:
                    break
            else:
                x = 2 * next_index - 3
                tmp = [2 * x, x]
                tmp_index = x
                if tmp_index > threshold_strategy2:
                    break
            fib_sequence_strategy2.append(tmp)
            next_index += 1
        next_index -= 1
        number_strategy2 = next_index

        if ablation_strategy == 'strategy1':
            memorybased_dim = number_strategy1
        elif ablation_strategy == 'strategy2':
            memorybased_dim = number_strategy2
        else:
            memorybased_dim = number_strategy1 + number_strategy2

        return fib_sequence_strategy1, fib_sequence_strategy2, memorybased_dim

    def setFibonacciFilter(self, ablation_strategy, threshold_strategy1, threshold_strategy2, scale=4):
        """
        :param input_shape: input shape
        :param scale: patch size or downsampling factor of embedding layer
        :return: fft_downsampling, downsampling_fft
        """
        fib_sequence = [1, 1]
        next_fib = 0
        number = 2
        while next_fib <= threshold_strategy1:
            next_fib = fib_sequence[number - 1] + fib_sequence[number - 2]
            number += 1
            fib_sequence.append(next_fib)
        number -= 1
        number_strategy1 = number

        number = 2
        next_fib = 0
        fib_sequence = [1, 1]
        while next_fib <= threshold_strategy2:
            next_fib = fib_sequence[number - 1] + fib_sequence[number - 2]
            number += 1
            fib_sequence.append(next_fib)
        number -= 1
        number_strategy2 = number

        if ablation_strategy == 'strategy1':
            memorybased_dim = number_strategy1
        elif ablation_strategy == 'strategy2':
            memorybased_dim = number_strategy2
        else:
            memorybased_dim = number_strategy1 + number_strategy2
        return self.fibonacci(number_strategy1), self.fibonacci(number_strategy2), memorybased_dim

    def setLinearFilter(self, ablation_strategy, threshold_strategy1, threshold_strategy2, scale=4):
        fib_sequence_strategy1 = []
        fib_sequence_strategy2 = []
        next_fib = 1
        while next_fib <= threshold_strategy1:
            tmp = [2*next_fib,next_fib]
            fib_sequence_strategy1.append(tmp)
            next_fib+=1

        next_fib -= 1
        number_strategy1 = next_fib

        next_fib = 1
        while next_fib <= threshold_strategy2:
            tmp = [2 * next_fib, next_fib]
            fib_sequence_strategy2.append(tmp)
            next_fib += 1
        next_fib -= 1
        number_strategy2 = next_fib
        if ablation_strategy == 'strategy1':
            memorybased_dim = number_strategy1
        elif ablation_strategy == 'strategy2':
            memorybased_dim = number_strategy2
        else:
            memorybased_dim = number_strategy1 + number_strategy2
        return fib_sequence_strategy1, fib_sequence_strategy2, memorybased_dim

    def setfilter(self, ablation_strategy, use_sequencefunc, threshold_strategy1,
                  threshold_strategy2,
                  scale=4) :
        print(use_sequencefunc)
        if use_sequencefunc == 'statistic':
            return self.setStatisticFilter(ablation_strategy, threshold_strategy1, threshold_strategy2, scale)
        if use_sequencefunc == 'fibonacci':
            return self.setFibonacciFilter(ablation_strategy, threshold_strategy1, threshold_strategy2, scale)
        if use_sequencefunc == 'linear':
            return self.setLinearFilter(ablation_strategy, threshold_strategy1, threshold_strategy2, scale)

    def transfourier(self, tensor, filter_h, filter_w):
        _, _, x, y = tensor.shape
        # FFT
        dtf = torch.fft.rfftn(tensor)
        dtf_shift = torch.fft.fftshift(dtf)
        # Initialize the filter
        ffilter = torch.zeros((int(filter_h), int(filter_w))).cuda()
        target_height, target_width = x, y // 2 + 1
        # compute the padding
        pad_height = max(target_height - ffilter.shape[0], 0)
        pad_width = max(target_width - ffilter.shape[1], 0)
        top_pad = pad_height // 2
        bottom_pad = pad_height - top_pad
        left_pad = pad_width // 2
        right_pad = pad_width - left_pad
        # padding
        fourier_filter = F.pad(ffilter, (left_pad, right_pad, top_pad, bottom_pad), mode='constant', value=1)
        fourier_filter = fourier_filter.unsqueeze(0).unsqueeze(0)
        # filtering
        dtf_shift = dtf_shift * fourier_filter
        # IFFT
        f_ishift = torch.fft.ifftshift(dtf_shift)
        img_back = torch.fft.irfftn(f_ishift)
        # [vis, useful_paramaters]
        return 20 * torch.log(torch.abs(dtf_shift)), torch.abs(img_back)

    def fastFourierTrans(self, tensor, filter: list, select: str, scale=4):
        # tensor [B,C,H,W]
        out_img_ret = 0
        out_fre_ret = 0
        if select == "strategy1":
            for index, element in enumerate(filter):
                out_fre, out_img = self.transfourier(tensor, filter_h=element[-2], filter_w=element[-1])
                out_img = torch.exp(-out_img)
                if index == 0:
                    out_img_ret = out_img
                    out_fre_ret = out_fre
                else:
                    out_img_ret = torch.cat([out_img_ret, out_img], dim=1)
                    out_fre_ret = torch.cat([out_fre_ret, out_fre], dim=1)
            out_img_ret = torch.nn.functional.max_pool2d(out_img_ret, kernel_size=scale, stride=scale)
            return out_fre_ret, out_img_ret

        elif select == "strategy2":
            tensor = torch.nn.functional.max_pool2d(tensor, kernel_size=scale, stride=scale)
            for index, element in enumerate(filter):
                out_fre, out_img = self.transfourier(tensor, filter_h=element[-2], filter_w=element[-1])
                if index == 0:
                    out_img_ret = out_img
                    out_fre_ret = out_fre
                else:
                    out_img_ret = torch.cat([out_img_ret, out_img], dim=1)
                    out_fre_ret = torch.cat([out_fre_ret, out_fre], dim=1)
            return out_fre_ret, torch.exp(-out_img_ret)

        else:
            return None, None

    def forward(self, x):
        _, _, H, W = x.shape

        if self.ablation_strategy == 'strategy1':
            _, s1_memorybased = self.fastFourierTrans(x.mean(1, keepdim=True), filter=self.filter, select='strategy1',
                                                      scale=self.fftscale)
            memorystream = s1_memorybased
        elif self.ablation_strategy == 'strategy2':
            _, s2_memorybased = self.fastFourierTrans(x.mean(1, keepdim=True), filter=self.downfilter,
                                                      select='strategy2',
                                                      scale=self.fftscale)
            memorystream = s2_memorybased
        else:
            _, s1_memorybased = self.fastFourierTrans(x.mean(1, keepdim=True), filter=self.filter, select='strategy1',
                                                      scale=self.fftscale)
            _, s2_memorybased = self.fastFourierTrans(x.mean(1, keepdim=True), filter=self.downfilter,
                                                      select='strategy2',
                                                      scale=self.fftscale)
            assert s1_memorybased != None, f"fast forier transform error! pleace check fastFourierTrans(fft2sampling) function."
            assert s2_memorybased != None, f"fast forier transform error! pleace check fastFourierTrans(sampling2fft) function."
            memorystream = torch.cat([s1_memorybased, s2_memorybased], dim=1)
            del s1_memorybased, s2_memorybased
        return memorystream


    def extra_repr(self) -> str:
        return f"filter_strategy1={self.filter_strategy1}, filter_strategy2={self.filter_strategy2}," \
               f"ablation_strategy={self.ablation_strategy}, use_sequencefunc={self.use_sequencefunc}," \
               f"fftscale={self.fftscale},"

