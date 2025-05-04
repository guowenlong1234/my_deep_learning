# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):#初始化传入参数中act_layer= nn.GELU
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features          #默认输出维度=输入维度
        hidden_features = hidden_features or in_features    #隐藏层取大的
        self.fc1 = nn.Linear(in_features, hidden_features)  #定义全连接层1
        self.act = act_layer()                              #定义激活函数，采用gelu
        self.fc2 = nn.Linear(hidden_features, out_features) #定义全连接层
        self.drop = nn.Dropout(drop)                        #定义dropout层

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:函数的作用是划分窗口
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape    #x值得是构造好的区域模板，已经经过了滑动之后
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    #[16, 56, 56, 96] -> [16, 8, 7, 8, 7, 96]
    #将x变成窗口的大小

    #[16, 8, 7, 8, 7, 96] ->[16, 8, 8, 7, 7, 96] -> [1024, 7, 7, 96]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C) # window的数量 H/7 * W/7 *batch

    return windows      #返回值[1024, 7, 7, 96] = [batch*win_num_h*win_num_w, patch_size_w,patch_size_h,c]


def window_reverse(windows, window_size, H, W):
    """
    这个函数的作用是将矩阵从窗口形式转化为了patch形式，与window_partition互为反函数
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    #[1024, 7, 7, 96]
    #b = batch = 16
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    #[1024, 7, 7, 96] -> [16, 8, 8 ,7, 7, 96]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)

    #[16, 8, 8, 7, 7, 96] -> [16, 8, 7, 8, 7 ,96] -> [16, 56, 56, 96]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

    return x


class WindowAttention(nn.Module):#窗口内部注意力机制计算类
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads #计算每个头的维度
        self.scale = qk_scale or head_dim ** -0.5   #定义缩放维度，如果没有定义缩放维度，则使用根号下头维度

        # define a parameter table of relative position bias

        '''
        定义一个可学习的相对位置偏差参数表，一个初始化全0的[2*Wh-1 * 2*Ww-1, nH]张量
        在第一个stage中，这是一个[13*13,3]维度的张量，第一个维度记录了每一对位置的相对位置关系，第二个维度记录了不同头之间的相对位置
        在这个表里，可以查询到任意一个窗口与其他任意一个窗口的位置关系
        '''
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        #获取窗口内每个标记的成对相对位置索引
        coords_h = torch.arange(self.window_size[0])    #生成窗口的行索引
        coords_w = torch.arange(self.window_size[1])    #生成窗口的列索引
        #torch.meshgrid([coords_h, coords_w])生成二维坐标网格
        #torch.stack是用于沿着新的维度（默认是第一维）将多个张量合并在一起。
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww

        #将张量展平[2, w, h] ->[2， w*h]
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww

        '''
        coords_flatten[:, :, None]再第三维添加一个维度，结果是一个形状为 (2, H * W, 1) 的张量。
        coords_flatten[:, None, :]再第二维度加上一个维度，得到一个形状为 (2, 1, H * W) 的张量
        coords_flatten[:, :, None] - coords_flatten[:, None, :] 会对这两个张量进行广播（broadcasting），从而计算每一对坐标之间的差值。
        结果就是计算了每对坐标之间的差值，得到一个形状为 (2, H * W, H * W) 的张量，表示每个坐标对之间的相对位置关系。
        '''

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        #调整矩阵形状并且使其再内存中连续Wh*Ww, Wh*Ww, 2
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2

        #
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)#register_buffer注册一个张量用来保存信息，这个张良是不能被学习的，不会更新

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)   #初始化一个kqv矩阵，用来计算kqv矩阵
        self.attn_drop = nn.Dropout(attn_drop)  #初始化dropout率
        self.proj = nn.Linear(dim, dim)     #初始化一个全连接层
        self.proj_drop = nn.Dropout(proj_drop)  #初始化全连接层dropout

        trunc_normal_(self.relative_position_bias_table, std=.02)#对相对位置编码表进行截断正态分布，使其标准差为0.02
        self.softmax = nn.Softmax(dim=-1)#初始化一个softmax层，表示沿着最后一个维度进行softmax操作

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """#传入参数[1024 ,49, 96]
        B_, N, C = x.shape

        #[1024 ,49, 96] -> [1024, 49, 96*3] -> [1024, 49, 3, 3, 32] -> [3, 1024, 3, 49, 32] -> [kqv, win_num, head, patch_num, head_size]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]    #分别取出kqv矩阵，维度为[1024, 3, 49, 32] =  [win_num, head, patch_num, head_size] # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale  #除以根号下特征维度，保证数据的统一性

        #@表示矩阵点积，[1024, 3, 49, 32]·[1024, 3, 32, 49] -> [1024, 3, 49, 49],transpose(-2, -1)表示转置
        attn = (q @ k.transpose(-2, -1))

        '''
        relative_position_bias_table是相对位置编码表[num_relative_positions, num_heads]，其中 num_relative_positions 是相对位置编码的数量，num_heads 是多头注意力的头数。
        self.relative_position_index是相对位置索引矩阵，存储了相对位置对的索引，通过view方法展平为1维矩阵，
        relative_position_index.view(-1) 展平后的索引值会依次去查找
        relative_position_bias_table 中的偏置值，返回的是一个一维的向量。，返回的是每一个位置于其他所有位置的相对位置偏置
        view() 操作将获取到的偏置重新形状化为二维矩阵。新的形状是 [window_size[0] * window_size[1], window_size[0] * window_size[1], -1]
        最终结果为[49, 49, 3]，表示有49个位置，每个位置于其他49个位置都有一个相对位置偏置，共三头注意力机制
        '''
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH



        #[49, 49, 3] -> [3, 49, 49]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        #将注意力得分加上相对位置偏置
        #[1024, 3, 49, 49] + [3, 49, 49]利用矩阵广播机制相加 -> [1024, 3, 49, 49]
        attn = attn + relative_position_bias.unsqueeze(0)


        if mask is not None:    #判断是否需要mask，如果有窗口滑动，则需要mask遮罩
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)   #[1024, 3, 49, 49] -> [16, 64, 3, 49, 49]，再加上mask将不需要的地方的值变得非常小，对mask插入维度，进行广播运算
            attn = attn.view(-1, self.num_heads, N, N)#[16, 64, 3, 49, 49] -> [1024, 3, 49, 49]
            attn = self.softmax(attn)   #经过softmax，将mask的位置的值都变成0
        else:       #不需要的，直接进行softmax
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)     #进行dropout


        #[1024, 3, 49, 49] · [1024, 3, 49, 32] = [1024, 3, 49, 32] -> [1024, 49, 3, 32] -> [1024, 49, 96]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)


        #进行全连接层
        x = self.proj(x)

        #进行dropou
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default:  True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        #如果patch数量小于窗口尺寸，就不需要再准备窗口
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)#窗口尺寸直接设置为最小的pacth数量即可
        '''
        这行代码是一个 断言语句，用来确保某个条件在程序运行时是成立的。如果条件不成立，程序会抛出一个 AssertionError 异常，提示开发者进行修正
        要求移动尺寸必须在0和窗口尺寸之间，如果不成立，就需要抛出异常
        '''
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)    #归一化层，输入参数为维度大小
        self.attn = WindowAttention(    #完成初始化kqv以及相对位置矩阵，是模型中的msa层
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        #根据drop_path参数的值判断是否采用DropPath技术，否则不采用dropout
        '''
        DropPath的基本原理是：在每次前向传播时，随机“丢弃”一些层的输出，以此来防止过拟合并提高模型的泛化能力。不同于传统的 Dropout 操作，DropPath 会直接跳过某一层的计算并返回其输入
        nn.Identity() 是一个 PyTorch 的内置层，它的作用是不做任何变换，直接返回输入。
        '''
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)    #归一化层，输入参数为维度大小
        #计算mlp层特征项向量维度
        mlp_hidden_dim = int(dim * mlp_ratio)
        #实例化MLP层
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:#如果有窗口滑动的话
            # calculate attention mask for SW-MSA执行下列操作计算注意力遮罩
            H, W = self.input_resolution  #每一行patch数量和每一列patch数量
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),    #对于行切片，从0到倒数第七个取出张量的头
                        slice(-self.window_size, -self.shift_size),     #从倒数第7个到倒数第三个取出张量的中间部分
                        slice(-self.shift_size, None))                  #从倒数第三个到最后取出张量的最后部分
            w_slices = (slice(0, -self.window_size),    #对列切片同样定义
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            #经过这三行代码，为一个56*56的矩阵的每一个格子都赋予了一定的值，每种区域内部填充的数字相同为0-8，具体见56*56矩阵表格
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            #mask_windowsx的维度变成（67，7，7，1），表示(windows, patch_h, patch_w, 1)
            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            #将mask_windows维度变成(67，49）：(windows,patch)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:#如果没有窗口滑动操作的话，不设置遮罩
            attn_mask = None
        #在模型中嵌入一个张量，但是不更新。名称为注意力遮罩
        self.register_buffer("attn_mask", attn_mask)    #完成了一个进本的block操作

    def forward(self, x):   #[16,3136,96]
        H, W = self.input_resolution        #取出每一行每一列的patch数量，首轮为56
        B, L, C = x.shape   #取出x的形状维度，
        assert L == H * W, "input feature has wrong size"   #判断维度是否出错

        shortcut = x    #保存一次x
        x = self.norm1(x)       #进行LN层，维度不变
        x = x.view(B, H, W, C)  #[16,3136,96]->[16,56,56,96]
        # cyclic shift
        if self.shift_size > 0: #判断是否需要窗口滑动
            #如果需要shift_siza,就在长和宽两个维度上进行循环位移，得到新的经过移动之后的数据
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) #如果需要窗口滑动
        else:
            shifted_x = x       #不需要窗口滑动，x的值不变，赋给shifted_x 保证了下面代码的一致性

        # partition windows 划分窗口,#返回值[1024, 7, 7, 96] = [batch*win_num_h*win_num_w, patch_size_w,patch_size_h,c]
        x_windows = window_partition(shifted_x, self.window_size)  # 函数的作用是将x划分为窗口，返回值[1024, 7, 7, 96] = [batch*win_num_h*win_num_w, patch_size_w,patch_size_h,c]nW*B, window_size, window_size, C

        #[1024, 7, 7, 96] -> [1024 ,49, 96]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA输入为[1024, 49, 96] ->[1024, 49, 96],加入了多头注意力、相对位置编码等机制，如果需要窗口滑动，则将窗口滑动机制也加入进去
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows[1024, 49, 96] -> [1024, 7, 7, 96]
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        #将数据矩阵从窗口形式重新变成patch形式[1024, 7, 7, 96] -> [16, 56, 56, 96]
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0: #如果需要进行窗口滑动操作，再将之前滑动的反过来，使图像回复原本样子
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        else:   #不需要进行窗口滑动操作
            x = shifted_x #x直接等于shifted_x

        #[16, 56, 56, 96] -> [16, 3136, 96]
        x = x.view(B, H * W, C)
        # FFN   进行残差链接，如果有需要采用drop_path
        x = shortcut + self.drop_path(x)

        #进行mlp链接
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution        #记录输入patch的数量
        self.dim = dim      #记录特征向量维度数量
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)    #全连接层，输入384，输出192，压缩维度
        self.norm = norm_layer(4 * dim) #在396维度上进行LN

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"       #判断输入是否合法，不合法抛出异常
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even." #判断输出是否合法，不合法抛出异常

        x = x.view(B, H, W, C)  #[16, 3136, 96] -> [16, 56, 56, 96]

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C #对x进行下采样，在横竖两个维度隔一个采样一个
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C [16, 56, 56, 96] -> [16, 28, 28, 384]
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C   #[16, 28, 28, 384]->[16, 784, 384],将四个下采样结果展平

        x = self.norm(x)        #针对384维度进行LN归一化处理
        x = self.reduction(x)   #将384维度变成192，通过一个全连接层进行转化 [16, 784, 384] -> [16, 784, 192]

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels. 输入图像尺寸
        input_resolution (tuple[int]): Input resolution. 两个维度上输入patch数量
        depth (int): Number of blocks.  该stage有多少个block
        num_heads (int): Number of attention heads. 头数
        window_size (int): Local window size.   窗口尺寸
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.    #mlp层维度
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True    qkv偏置
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.    qk缩放比例
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm    归一化策略
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None    是否下采样
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False. 是否使用检查点
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim      #特征维度
        self.input_resolution = input_resolution    #两个维度上的patch数量
        self.depth = depth  #该stage有多少个block
        self.use_checkpoint = use_checkpoint

        # build blocks构造block
        self.blocks = nn.ModuleList([ #完成了1个block的初始化，通过shift_size参数判断是否为sw-msa/w-msa
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,    #单数block则shift_size=0，双数block则window_size // 2向下取整
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)]) #重复构造depth个block

        # patch merging layer
        #判断是否需要patch merging，如果需要，执行downsample下采样
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:     #
            if self.use_checkpoint: #判断是否采用了检查点继续
                x = checkpoint.checkpoint(blk, x)
            else:   #没有采用检查点继续
                x = blk(x)  #再第一个stage中，完成了两个block，分别为一个wsma和一个swmsa
        if self.downsample is not None:
            x = self.downsample(x)  #[16, 3136, 96] -> [16, 784, 192]
        return x    #这个函数完成了两个block的运算，并且进行了下采样操作，将特征维度翻倍，窗口数量减小四倍，[16, 784, 192]

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)          #将image_size转化为一个元组(224, 224)
        patch_size = to_2tuple(patch_size)      #将patch size转化为一个元组(4, 4)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]#计算一张图片在两个维度上分别能切出来多少个patch
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]        #计算patch数量

        self.in_chans = in_chans    #图片通道数
        self.embed_dim = embed_dim  #特征向量维度
        #一个卷积层，输入通道数，输出特征向量维度，卷积核尺寸为patch_size，每次移动一个小patch
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:      #如果有embedding归一化层
            self.norm = norm_layer(embed_dim)   #设置一个归一化层
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape    #x[16, 3, 224, 224]，取出各个维度的数值
        # FIXME look at relaxing size constraints
        #判断输入是否合法，如果不合法则抛出异常信息
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # [16, 3, 224, 224]->[16,96,56,56]-> [16, 96, 3136]->[16,3136,96],每个batch有16张图片，每个图片有3136patch，向量长度为96
        x = self.proj(x).flatten(2).transpose(1, 2)

        #判断是否有归一化层
        if self.norm is not None:
            x = self.norm(x)    #如果有进行归一化处理，数据维度不变
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224 输入图像大小，默认224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000 分类数量默认1000
        embed_dim (int): Patch embedding dimension. Default: 96 Patch embedding 向量特征维度
        depths (tuple(int)): Depth of each Swin Transformer layer. 规定了每一层的block数量
        num_heads (tuple(int)): Number of attention heads in different layers.  每一层的头数
        window_size (int): Window size. Default: 7  窗口大小，默认7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4 规定mlp层参数量是特征维度96的几倍
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True 如果为 True，则为qkv添加可学习的偏置。默认值： 为 True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None qk的缩放比例
        drop_rate (float): Dropout rate. Default: 0 dropout的比例
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm. 归一化方法
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False 是否嵌入绝对位置编码
        patch_norm (bool): If True, add normalization after patch embedding. Default: True 是否在embedding后添加归一化层
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False  是否使用chekpoint继续训练
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes  #分类数量
        self.num_layers = len(depths)   #每一层有多少个block2，2，6，2
        self.embed_dim = embed_dim      #特征向量维度
        self.ape = ape                  #是否嵌入绝对位置编码 false
        self.patch_norm = patch_norm    #在embedding后添加归一化层
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1)) #最后一层的特征维度每一个stage使特征向量维度翻倍 96 * 2^3 = 96 * 8 = 768
        self.mlp_ratio = mlp_ratio      #mlp层特征翻倍数

        # split image into non-overlapping patches，分割图像为不重合的patch
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches      #计算总共patch数量3136
        patches_resolution = self.patch_embed.patches_resolution    #在两个维度上分别能切出来多少个patch[56, 56]
        self.patches_resolution = patches_resolution

        # absolute position embedding 是否采用绝对位置编码
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) #构造一个全零的位置编码矩阵[1, 3136, 96]
            trunc_normal_(self.absolute_pos_embed, std=.02)     #采用截断标准差初始化，使绝大多数值位于标准差0.02范围内，防止梯度爆炸

        self.pos_drop = nn.Dropout(p=drop_rate)     #设置dropout

        # stochastic depth随即深度
        #设置随机深度衰减规则
        '''
        torch.linspace(start, end, steps)：生成从 start 到 end 之间均匀分布的 steps 个值
        x.item()：从每个张量 x 中提取单个数值并将其转化为 Python 基本数据类型（如 float）。
        最后生成一个列表，包括一系列不同的drop_path_rate，越往后面层的dropout概率越高
        '''
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()       #构造一个模型容器，存储不同的层
        for i_layer in range(self.num_layers):  #遍历每一个stage，为其中添加一个层
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),   #当前stage的特征向量维度
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),       #在当前stage中，输入图像的尺寸
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],       #当前层的深度，包括多少个block
                               num_heads=num_heads[i_layer],    #当前stage应该有的头数，越往下头数越多
                               window_size=window_size,         #窗口尺寸
                               mlp_ratio=self.mlp_ratio,        #mlp特征维度倍数
                               qkv_bias=qkv_bias, qk_scale=qk_scale,    #是否设置kqv偏置，qk缩放比例
                               drop=drop_rate, attn_drop=attn_drop_rate,    #dropout率
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  #从数列里面去除一个数作为drop_path_rate
                               norm_layer=norm_layer,   #是否进行归一化处理
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,    #下采样策略，不是最后一层都要进行下采样
                               use_checkpoint=use_checkpoint)   #是否使用梯度检查点
            self.layers.append(layer)       #将构造好的layer添加到容器中

        self.norm = norm_layer(self.num_features)   #实例化一个归一化层
        self.avgpool = nn.AdaptiveAvgPool1d(1)      #平均池化，采用平均的方法将特征向量压缩成一维向量
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()     #实例化输出层

        self.apply(self._init_weights)      #初始化模型权重和偏置

    def _init_weights(self, m):
        '''
        :m是传入层的对象，函数会根据传入对象的不同类进行不同的初始化操作
        :param m:
        :return:
        '''
        if isinstance(m, nn.Linear):        #判断传入的对象是否是linear类
            trunc_normal_(m.weight, std=.02)    #采用截断式正态分布的方法初始化权重，正态分布方差为0.2
            if isinstance(m, nn.Linear) and m.bias is not None: #判断这个层是否有偏置
                nn.init.constant_(m.bias, 0)    ##初始偏置初始化为全0
        elif isinstance(m, nn.LayerNorm):   #判断传入对象是否为LN层
            nn.init.constant_(m.bias, 0)        #初始化偏置为0
            nn.init.constant_(m.weight, 1.0)        #初始化权重为1

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore   #这个装饰器的作用是用于标记不应被 JIT 编译的函数或方法。JIT 是 PyTorch 用于加速模型推理的技术，它通过将 Python 代码转换为更高效的底层代码来提升性能。
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):

        x = self.patch_embed(x) #[16, 3, 224, 224]  -> [16,3136,96],经过了一个卷积层，每个batch有16张图片，每个图片有3136patch，向量长度为96


        if self.ape:    #判断是否嵌入绝对位置编码
            x = x + self.absolute_pos_embed     #采用绝对位置编码，初始化为全零的一个张量。
        x = self.pos_drop(x)    #[16,3136,96]不变，使用dropout


        for layer in self.layers:
            x = layer(x)    #在第一次执行这个函数过程中[16, 3136, 96] -> [16, 784, 192],完成了一个stage的计算，并且完成了下采样过程
                            #[16, 3136, 96] -> [16, 784, 192] -> [16, 196, 384] -> [16, 49, 768],经过了四个stage，每次序列长度减少四倍，特征维度翻一倍

        x = self.norm(x)  # B L C   #经过LN归一化处理

        x = self.avgpool(x.transpose(1, 2))  # B C 1 #[16, 49, 768] -> [16, 768, 49] -> [16, 768, 1]

        x = torch.flatten(x, 1)  #[16, 768, 1] -> [16, 768]

        return x

    def forward(self, x):
        x = self.forward_features(x)    #x[16,3,224,224] ->[16, 768]将每一张图片提取为一个特征向量
        x = self.head(x)    #根据这个特征向量对图片进行分类  [16, 768] -> [16, 10]
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
