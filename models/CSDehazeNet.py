from unittest import skipIf

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.init import _calculate_fan_in_and_fan_out
# from timm.models.layers import to_2tuple,
from timm.layers import DropPath, to_2tuple,trunc_normal_

class AGN(nn.Module):
    r"""Adaptive Global Normalization with Texture Attention"""
    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super(AGN, self).__init__()
        self.eps = eps
        self.detach_grad = detach_grad

        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)

        self.local_attn = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        )

        self.texture_attn = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        )

        # 初始化卷积层
        trunc_normal_(self.meta1.weight, std=.02)
        nn.init.constant_(self.meta1.bias, 1)
        trunc_normal_(self.meta2.weight, std=.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, input):
        # 计算全局均值和标准差
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

        # 归一化输入
        normalized_input = (input - mean) / std

        if self.detach_grad:
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean)

        local_enhanced = self.local_attn(normalized_input)
        texture_enhanced = self.texture_attn(normalized_input)
        dynamic_weight = self.weight * rescale
        dynamic_bias = self.bias + rebias
        out = normalized_input * dynamic_weight + dynamic_bias + local_enhanced + texture_enhanced
        return out, rescale, rebias


class Mlp(nn.Module):
	def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
		super().__init__()
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features

		self.network_depth = network_depth

		self.mlp = nn.Sequential(
			nn.Conv2d(in_features, hidden_features, 1),
			nn.ReLU(True),
			nn.Conv2d(hidden_features, out_features, 1)
		)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Conv2d):
			gain = (8 * self.network_depth) ** (-1/4)
			fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
			std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
			trunc_normal_(m.weight, std=std)
			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		return self.mlp(x)


def window_partition(x, window_size):
	B, H, W, C = x.shape
	x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
	windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size**2, C)
	return windows


def window_reverse(windows, window_size, H, W):
	B = int(windows.shape[0] / (H * W / window_size / window_size))
	x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
	x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
	return x




class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.size(0), 1, 1, 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.1, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.ReLU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x

def get_relative_positions(window_size):
	coords_h = torch.arange(window_size)
	coords_w = torch.arange(window_size)

	coords = torch.stack(torch.meshgrid([coords_h, coords_w],indexing='ij'))  # 2, Wh, Ww
	coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
	relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

	relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
	relative_positions_log  = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())

	return relative_positions_log


class WindowAttention(nn.Module):
	def __init__(self, dim, window_size, num_heads,logit_scale_init=math.log(1 / 0.07)):

		super().__init__()
		self.dim = dim
		self.window_size = window_size  # Wh, Ww
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim ** -0.5
		self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init))
		relative_positions = get_relative_positions(self.window_size)
		self.register_buffer("relative_positions", relative_positions)
		self.meta = nn.Sequential(
			nn.Linear(2, 256, bias=True),
			nn.ReLU(True),
			nn.Linear(256, num_heads, bias=True)
		)

		self.softmax = nn.Softmax(dim=-1)

	def forward(self, qkv,conv_out=None):
		B_q, N_q, _ = qkv.shape
		B_c, N_c, _ = conv_out.shape
		# print("conv_out",conv_out.shape)
		qkv = qkv.reshape(B_q, N_q, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
		conv_out = conv_out.reshape(B_c, N_c, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)
		# print("conv_out.shape", conv_out.shape)
		# print("qkv", qkv.shape)

		# print("qkv.shape", qkv.shape)
		q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
		# print("q.shape", q.shape)
		q = conv_out
		k = k * self.scale
		# Compute scaled cosine similarity
		attn = (q @ k.transpose(-2, -1))
		# print("attn", attn.shape)
		logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
		attn = attn * logit_scale

		relative_position_bias = self.meta(self.relative_positions)
		relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
		attn = attn + relative_position_bias.unsqueeze(0)

		attn = self.softmax(attn)

		x = (attn @ v).transpose(1, 2).reshape(B_q, N_q, self.dim)
		return x


class DSAttention(nn.Module):
	def __init__(self, network_depth, dim, num_heads, window_size, shift_size, use_attn=False, conv_type=None):
		super().__init__()
		self.dim = dim
		self.head_dim = int(dim // num_heads)
		self.num_heads = num_heads

		self.window_size = window_size
		self.shift_size = shift_size

		self.network_depth = network_depth
		self.use_attn = use_attn
		self.conv_type = conv_type

		if self.conv_type == 'DWConv':
			self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')

		if self.conv_type == 'DWConv' or self.use_attn:
			self.Q= nn.Conv2d(dim, dim, 1)
			self.proj = nn.Conv2d(dim, dim, 1)

		if self.use_attn:
			self.KV = nn.Conv2d(dim, dim * 2, 1)
			self.attn = WindowAttention(dim, window_size, num_heads)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Conv2d):
			w_shape = m.weight.shape

			if w_shape[0] == self.dim * 2:  # QK
				fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
				std = math.sqrt(2.0 / float(fan_in + fan_out))
				trunc_normal_(m.weight, std=std)
			else:
				gain = (8 * self.network_depth) ** (-1 / 4)
				fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
				std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
				trunc_normal_(m.weight, std=std)

			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

	def check_size(self, x, shift=False):
		_, _, h, w = x.size()
		mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
		mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

		if shift:
			x = F.pad(x, (self.shift_size, (self.window_size - self.shift_size + mod_pad_w) % self.window_size,
						  self.shift_size, (self.window_size - self.shift_size + mod_pad_h) % self.window_size),
					  mode='reflect')
		else:
			x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
		return x

	def forward(self, X):
		B, C, H, W = X.shape

		if self.conv_type == 'DWConv' or self.use_attn:
			Q = self.Q(X)

		if self.use_attn:
			KV = self.KV(X)
			QKV = torch.cat([Q,KV], dim=1)

			# shift
			shifted_QKV = self.check_size(QKV, self.shift_size > 0)
			Ht, Wt = shifted_QKV.shape[2:]
			shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
			# print("shifted_QKV.shape", shifted_QKV.shape)


			conv_out = self.conv(Q)
			conv_out = self.check_size(conv_out, self.shift_size > 0)
			conv_out = conv_out.permute(0, 2, 3, 1)
			conv_out = window_partition(conv_out, self.window_size)  # nW*B, window_size**2, C


			qkv = window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C

			attn_windows = self.attn(qkv,conv_out)

			# merge windows
			shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)  # B H' W' C

			# reverse cyclic shift
			out = shifted_out[:, self.shift_size:(self.shift_size + H), self.shift_size:(self.shift_size + W), :]
			attn_out = out.permute(0, 3, 1, 2)
			out = self.proj( attn_out)
		else:
			out = self.proj(self.conv(Q))

		return out



class DSDehazeBlock(nn.Module):
	def __init__(self, network_depth, dim, num_heads, mlp_ratio=4.,
				 norm_layer=nn.LayerNorm, mlp_norm=False,
				 window_size=8, shift_size=0, use_attn=True, conv_type=None):
		super().__init__()
		self.use_attn = use_attn
		self.mlp_norm = mlp_norm

		self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
		self.attn = DSAttention(network_depth, dim, num_heads=num_heads, window_size=window_size,
							  shift_size=shift_size, use_attn=use_attn, conv_type=conv_type)

		self.norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()
		self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))
	def forward(self, x):
		identity = x
		if self.use_attn: x, rescale, rebias = self.norm1(x)
		x = self.attn(x)
		if self.use_attn: x = x * rescale + rebias
		x = identity + x
		identity = x
		if self.use_attn and self.mlp_norm: x, rescale, rebias = self.norm2(x)
		x = self.mlp(x)
		if self.use_attn and self.mlp_norm: x = x * rescale + rebias
		x = identity + x
		return x


class BasicLayer(nn.Module):
	def __init__(self, network_depth, dim, depth, num_heads, mlp_ratio=4.,
				 norm_layer=nn.LayerNorm, window_size=8,
				 attn_ratio=0., attn_loc='last', conv_type=None):

		super().__init__()
		self.dim = dim
		self.depth = depth

		attn_depth = attn_ratio * depth

		if attn_loc == 'last':
			use_attns = [i >= depth-attn_depth for i in range(depth)]
		elif attn_loc == 'first':
			use_attns = [i < attn_depth for i in range(depth)]
		elif attn_loc == 'middle':
			use_attns = [i >= (depth-attn_depth)//2 and i < (depth+attn_depth)//2 for i in range(depth)]

		# build blocks
		self.blocks = nn.ModuleList([
			 DSDehazeBlock(network_depth=network_depth,
							 dim=dim,
							 num_heads=num_heads,
							 mlp_ratio=mlp_ratio,
							 norm_layer=norm_layer,
							 window_size=window_size,
							 shift_size=0 if (i % 2 == 0) else window_size // 2,
							 use_attn=use_attns[i], conv_type=conv_type)
			for i in range(depth)])

	def forward(self, x):
		for blk in self.blocks:
			x = blk(x)
		return x


class PatchEmbed(nn.Module):
	def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
		super().__init__()
		self.in_chans = in_chans
		self.embed_dim = embed_dim

		if kernel_size is None:
			kernel_size = patch_size

		self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
							  padding=(kernel_size-patch_size+1)//2, padding_mode='reflect')

	def forward(self, x):
		x = self.proj(x)
		return x


class PatchUnEmbed(nn.Module):
	def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
		super().__init__()
		self.out_chans = out_chans
		self.embed_dim = embed_dim

		if kernel_size is None:
			kernel_size = 1

		self.proj = nn.Sequential(
			nn.Conv2d(embed_dim, out_chans*patch_size**2, kernel_size=kernel_size,
					  padding=kernel_size//2, padding_mode='reflect'),
			nn.PixelShuffle(patch_size)
		)

	def forward(self, x):
		x = self.proj(x)
		return x

class PatchEmbedV2(nn.Module):
	def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
		super().__init__()
		self.in_chans = in_chans
		self.embed_dim = embed_dim
		if kernel_size is None:
			kernel_size = patch_size
		self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
							  padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')
	def forward(self, x):
		B, C, H, W = x.shape
		original_size = (H, W)  # 保存原始尺寸
		pad_h = H % 2 != 0
		pad_w = W % 2 != 0
		if pad_h:
			x = torch.cat([x, x[:, :, -1:, :]], dim=2)
		if pad_w:
			x = torch.cat([x, x[:, :, :, -1:]], dim=3)
		x = self.proj(x)
		return x, original_size


class PatchUnEmbedV2(nn.Module):
	def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
		super().__init__()
		self.out_chans = out_chans
		self.embed_dim = embed_dim
		if kernel_size is None:
			kernel_size = 1
		self.proj = nn.Sequential(
			nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
					  padding=kernel_size // 2, padding_mode='reflect'),
			nn.PixelShuffle(patch_size)
		)
	def forward(self, x, original_size=None):
		x = self.proj(x)
		B, C, H, W = x.shape
		if original_size is not None:
			target_h, target_w = original_size
			return x[:, :, :target_h, :target_w]
		else:
			if H % 2 != 0:
				x = x[:, :, :-1, :]
			if W % 2 != 0:
				x = x[:, :, :, :-1]
		return x




class CSE_Fusion(nn.Module):
	def __init__(self, dim, height=2, reduction=8):
		super().__init__()
		self.height = height
		self.dim = dim

		self.spatial_attn = nn.Sequential(
			nn.Conv2d(1, 1, 3, padding=1, bias=False),
			nn.Sigmoid()
		)

		self.dynamic_ratio = nn.Parameter(torch.tensor(1.0))  # 可学习缩放因子

		d = max(int(dim * self.dynamic_ratio / reduction), 4)
		self.mlp = nn.Sequential(
			nn.Conv2d(dim * height, d, 1, bias=False),
			nn.ReLU(),
			nn.Conv2d(d, dim * height, 1, bias=False)
		)

		self.softmax = nn.Softmax(dim=1)

	def forward(self, in_feats):
		B, C, H, W = in_feats[0].shape
		x = torch.cat(in_feats, dim=1).view(B, self.height, C, H, W)

		# 通道注意力
		feats_avg = x.mean(dim=[-2, -1], keepdim=True)
		attn = self.mlp(feats_avg.view(B, -1, 1, 1))
		attn = self.softmax(attn.view(B, self.height, C, 1, 1))

		# 空间注意力增强
		spatial_mask = self.spatial_attn(x.sum(1).mean(1, keepdim=True))  # [B,1,H,W]
		out = (x * attn).sum(1) * spatial_mask

		return out

class CSDehaze(nn.Module):
	def __init__(self, in_chans=3, out_chans=4, window_size=8,
				 embed_dims=[24, 48, 96, 48, 24],
				 mlp_ratios=[2., 4., 4., 2., 2.],
				 depths=[4, 4, 4, 2, 2],
				 num_heads=[2, 4, 6, 1, 1],
				 attn_ratio=[1/4, 1/2, 1, 0, 0],
				 conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
				 norm_layer=[AGN, AGN, AGN, AGN, AGN]):
		super(CSDehaze, self).__init__()

		# setting
		self.patch_size = 8
		self.window_size = window_size
		self.mlp_ratios = mlp_ratios

		# split image into non-overlapping patches
		self.patch_embed = PatchEmbed(
			patch_size=1, in_chans=in_chans, embed_dim=3, kernel_size=3)

		self.patch_merge_CN1_1 = PatchEmbed(
			patch_size=2, in_chans=3, embed_dim=6)


		#卷积处理
		self.CN1 = ConvNeXtBlock(6)
		self.skip1= nn.Conv2d(6, 6, 1)
		self.patch_merge_CN1 = PatchEmbed(
			patch_size=2, in_chans=6, embed_dim=12)


		self.CN2 = ConvNeXtBlock(12)
		self.patch_merge_CN2= PatchEmbed(
			patch_size=2, in_chans=12, embed_dim=24)
		self.skip2 = nn.Conv2d(12, 12, 1)

		# backbone
		self.layer1 = BasicLayer(network_depth=sum(depths), dim=embed_dims[0], depth=depths[0],
					   			 num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
					   			 norm_layer=norm_layer[0], window_size=window_size,
					   			 attn_ratio=attn_ratio[0], attn_loc='last', conv_type=conv_type[0])

		self.patch_merge1 = PatchEmbedV2(
			patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])

		self.skip3 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

		self.layer2 = BasicLayer(network_depth=sum(depths), dim=embed_dims[1], depth=depths[1],
								 num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
								 norm_layer=norm_layer[1], window_size=window_size,
								 attn_ratio=attn_ratio[1], attn_loc='last', conv_type=conv_type[1])


		self.patch_merge2 = PatchEmbedV2(
			patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

		self.skip4 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

		self.layer3 = BasicLayer(network_depth=sum(depths), dim=embed_dims[2], depth=depths[2],
								 num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
								 norm_layer=norm_layer[2], window_size=window_size,
								 attn_ratio=attn_ratio[2], attn_loc='last', conv_type=conv_type[2])


		self.patch_split1 = PatchUnEmbedV2(
			patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

		assert embed_dims[1] == embed_dims[3]
		self.fusion1 = CSE_Fusion(embed_dims[3])

		self.layer4 = BasicLayer(network_depth=sum(depths), dim=embed_dims[3], depth=depths[3],
								 num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
								 norm_layer=norm_layer[3], window_size=window_size,
								 attn_ratio=attn_ratio[3], attn_loc='last', conv_type=conv_type[3])

		self.patch_split2 = PatchUnEmbedV2(
			patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

		assert embed_dims[0] == embed_dims[4]
		self.fusion2 = CSE_Fusion(embed_dims[4])

		self.layer5 = BasicLayer(network_depth=sum(depths), dim=embed_dims[4], depth=depths[4],
					   			 num_heads=num_heads[4], mlp_ratio=mlp_ratios[4],
					   			 norm_layer=norm_layer[4], window_size=window_size,
					   			 attn_ratio=attn_ratio[4], attn_loc='last', conv_type=conv_type[4])

		self.patch_split_CN3 = PatchUnEmbed(
			patch_size=2, out_chans=12, embed_dim=24)
		self.CN3 = ConvNeXtBlock(12)
		self.fusion3 = CSE_Fusion(12)


		self.patch_split_CN4 = PatchUnEmbed(
			patch_size=2, out_chans=6, embed_dim=12)
		self.CN4 = ConvNeXtBlock(6)
		self.fusion4 = CSE_Fusion(6)

		self.patch_split_CN4_2 = PatchUnEmbed(
			patch_size=2, out_chans=4, embed_dim=6)

		self.patch_unembed = PatchUnEmbed(
			patch_size=1, out_chans=out_chans, embed_dim=4, kernel_size=3)


	def check_image_size(self, x):
		# NOTE: for I2I test
		_, _, h, w = x.size()
		mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
		mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
		x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
		return x

	def forward_features(self, x):

		x = self.patch_embed(x)
        #卷积处理
		x = self.patch_merge_CN1_1(x)
		x_1 = x
		x = self.CN1(x)
		x = self.CN1(x)
		x = x + x_1
		skip1 = x
		x = self.patch_merge_CN1(x)



		x_2 = x
		x = self.CN2(x)
		x = self.CN2(x)
		x = x + x_2
		skip2 = x
		x = self.patch_merge_CN2(x)

		#Transformer处理
		x_3 = x
		x = self.layer1(x)
		x = x + x_3
		skip3 = x
		x, hw1 = self.patch_merge1(x)

		x_4 = x
		x = self.layer2(x)
		x = x + x_4
		skip4 = x
		x, hw = self.patch_merge2(x)

		x = self.layer3(x)
		x = self.patch_split1(x, hw)
		x = self.fusion1([x, self.skip4(skip4)]) + x

		x = self.layer4(x)


		x = self.patch_split2(x,hw1)
		x = self.fusion2([x, self.skip3(skip3)]) + x

		x = self.layer5(x)

		# 卷积处理
		x = self.patch_split_CN3(x)
		x = self.fusion3([x, self.skip2(skip2)]) + x
		x = self.CN3(x)
		# print("x.shape",x.shape)
		x = self.patch_split_CN4(x)
		# x = x + self.skip1(skip1)
		x = self.fusion4([x, self.skip1(skip1)]) + x
		x = self.CN4(x)

		# print("++++++++++++++++++++++++++++++++++++++++++++++++")

		x = self.patch_split_CN4_2(x)

		x = self.patch_unembed(x)
		return x

	def forward(self, x):
		H, W = x.shape[2:]
		x = self.check_image_size(x)

		feat = self.forward_features(x)
		K, B = torch.split(feat, (1, 3), dim=1)

		x = K * x - B + x
		x = x[:, :, :H, :W]
		return x




