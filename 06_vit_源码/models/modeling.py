# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import models.configs as configs

from .modeling_resnet import ResNetV2


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):#初始化传入参数config, vis
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]      #设置注意力机制的头数，在当前为12
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)   #计算每个头应该有的特征维度
        self.all_head_size = self.num_attention_heads * self.attention_head_size        #计算所有头合并起来的特征维度

        self.query = Linear(config.hidden_size, self.all_head_size)     #初始化三个个计算Q、K、V矩阵的全连接层
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)       #初始化一个输出层，
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])   #初始化两个dropout层
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)      #初始化激活函数

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        print(new_x_shape)
        x = x.view(*new_x_shape)
        print(x.shape)
        print(x.permute(0, 2, 1, 3).shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        print(hidden_states.shape)
        mixed_query_layer = self.query(hidden_states)
        print(mixed_query_layer.shape)
        mixed_key_layer = self.key(hidden_states)
        print(mixed_key_layer.shape)
        mixed_value_layer = self.value(hidden_states)
        print(mixed_value_layer.shape)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        print(query_layer.shape)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        print(key_layer.shape)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        print(value_layer.shape)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        print(attention_scores.shape)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        print(attention_scores.shape)
        attention_probs = self.softmax(attention_scores)
        print(attention_probs.shape)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)
        print(attention_probs.shape)

        context_layer = torch.matmul(attention_probs, value_layer)
        print(context_layer.shape)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        print(context_layer.shape)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        print(context_layer.shape)
        attention_output = self.out(context_layer)
        print(attention_output.shape)
        attention_output = self.proj_dropout(attention_output)
        print(attention_output.shape)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):             #初始化传入参数config
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])        #线性层
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)        #线性层
        self.act_fn = ACT2FN["gelu"]                                                #激活函数层，选择激活函数gelu
        self.dropout = Dropout(config.transformer["dropout_rate"])                  #设置dropout参数

        self._init_weights()                            #初始化各个权重项

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)        #采用Xavier 初始化权重参数，它试图保持每一层的激活值和梯度的方差一致
        nn.init.xavier_uniform_(self.fc2.weight)        #采用Xavier 初始化，它试图保持每一层的激活值和梯度的方差一致
        nn.init.normal_(self.fc1.bias, std=1e-6)        #采用正态分布初始化偏置参数
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()#初始化函数是，传入参数config, img_size=img_size
        self.hybrid = None#设置是否为混合模式
        img_size = _pair(img_size)#调用的pytorch内置函数，这个函数通常会检查输入的 img_size 类型并进行相应的转换。例如：如果 img_size 是一个单一的整数，如 224，则 _pair(224) 会返回 (224, 224)，即假设图像的高度和宽度相等。如果 img_size 已经是一个二元组，例如 (224, 224)，则 _pair((224, 224)) 会直接返回 (224, 224)。

        if config.patches.get("grid") is not None:#在所有的配置中并没有找到这个参数。暂时不清楚这个参数的作用
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            #获取patch_size并且经过pair函数处理为元组类型
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            #计算出获取patch的数量，将其存储在n_patches变量中
            self.hybrid = False#不是混合模式，只有单独的transformer模型

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        #设置一个卷积层，卷积输入为in_channels=3，输出为模型的hidden_size，卷积核尺寸是与patch尺寸相同，滑动距离也是一个patch的尺寸，就是对每一个patch块做一下卷积操作

        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        #设置位置编码，初始化为全0张量，（batch，patch数量加1（因为cls_token），hidden_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        #初始化cls_token，初始化为全0向量，初始为全batch共享，后面通过self.cls_token.expand为每一个图片赋予自己的token。

        self.dropout = Dropout(config.transformer["dropout_rate"])
        #初始化dropout参数

    def forward(self, x):
        print(x.shape)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        print(cls_tokens.shape)
        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        print(x.shape)
        x = x.flatten(2)
        print(x.shape)
        x = x.transpose(-1, -2)
        print(x.shape)
        x = torch.cat((cls_tokens, x), dim=1)
        print(x.shape)

        embeddings = x + self.position_embeddings
        print(embeddings.shape)
        embeddings = self.dropout(embeddings)
        print(embeddings.shape)
        return embeddings


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()       #初始化传入参数config, vis
        self.hidden_size = config.hidden_size                           #初始化hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)   #初始化一个归一化层，用于注意力机制的归一化
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)         #初始化一个归一化层，用于ffn
        self.ffn = Mlp(config)           #初始化一个ffn层，其中包括两个线性层，一个激活函数，设置有dropout和初始化参数方法。
        self.attn = Attention(config, vis)  #初始化一个attn注意力机制，包括头数，每个头的尺寸，所有头的尺寸和，计算QKV三个矩阵的全连接层，两个dropout层和激活函数

    def forward(self, x):
        print(x.shape)
        h = x
        x = self.attention_norm(x)
        print(x.shape)
        x, weights = self.attn(x)
        x = x + h
        print(x.shape)

        h = x
        x = self.ffn_norm(x)
        print(x.shape)
        x = self.ffn(x)
        print(x.shape)
        x = x + h
        print(x.shape)
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            """ 
            linux下路径按照这个
            
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))
            """
            query_weight = np2th(weights[ROOT + "/" + ATTENTION_Q + "/" + "kernel"]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[ROOT + "/" +  ATTENTION_K+ "/" + "kernel"]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[ROOT + "/" +  ATTENTION_V+"/" + "kernel"]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[ROOT + "/" + ATTENTION_OUT+"/" + "kernel"]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[ROOT + "/" +  ATTENTION_Q+"/" + "bias"]).view(-1)
            key_bias = np2th(weights[ROOT + "/" +  ATTENTION_K+"/" + "bias"]).view(-1)
            value_bias = np2th(weights[ROOT + "/" +  ATTENTION_V+"/" + "bias"]).view(-1)
            out_bias = np2th(weights[ROOT + "/" +  ATTENTION_OUT+"/" + "bias"]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[ROOT + "/" +  FC_0+"/" + "kernel"]).t()
            mlp_weight_1 = np2th(weights[ROOT + "/" +  FC_1+"/" + "kernel"]).t()
            mlp_bias_0 = np2th(weights[ROOT + "/" +  FC_0+"/" +"bias"]).t()
            mlp_bias_1 = np2th(weights[ROOT + "/" +  FC_1+"/" +"bias"]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[ROOT + "/" +  ATTENTION_NORM+"/" + "scale"]))
            self.attention_norm.bias.copy_(np2th(weights[ROOT + "/" + ATTENTION_NORM+"/" +  "bias"]))
            self.ffn_norm.weight.copy_(np2th(weights[ROOT + "/" + MLP_NORM+"/" +  "scale"]))
            self.ffn_norm.bias.copy_(np2th(weights[ROOT + "/" + MLP_NORM+"/" +  "bias"]))


class Encoder(nn.Module):
    def __init__(self, config, vis):        #初始化encoder，初始化传入参数config, vis
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()        #初始化一个layer作为模型容器，存储对应的层，可以像访问列表一样访问对应的层
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)#初始化一个归一化层，在采用的是LayerNorm归一化方法，即为对每个样本的特征归一化，与之对应的 Batch是对整个批次中的的每个特征进行归一化操作
        for _ in range(config.transformer["num_layers"]):#"num_layers"有几层就在模型容器layer中添加几层Block
            layer = Block(config, vis)      #初始化两个归一化层，一个ffn层和一个注意力机制层
            self.layer.append(copy.deepcopy(layer))     #需要多少层，就在模型容器中存储多少层block

    def forward(self, hidden_states):
        print(hidden_states.shape)
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    #在VisionTransformer中调用
    #输入数据为x
    #返回值为x，x, attn_weights
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)     #完成Embeddings的初始化操作，主要有窗口卷积层，位置编码层，cls_token的初始化
        self.encoder = Encoder(config, vis)     #初始化Encoder，包括8个block层，一个归一化层

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class VisionTransformer(nn.Module):
    #调用时的传入参数：config模型类型, args.img_size图像尺寸大小, zero_head=True, num_classes=num_classes数据集类型，默认10
    #返回值model
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes      #分类的任务数量，默认为使用cifer10数据集的10分类任务
        self.zero_head = zero_head          #？
        self.classifier = config.classifier #token？

        self.transformer = Transformer(config, img_size, vis)       #完成transform的主要初始化，包括Embeddings和Encoder
        self.head = Linear(config.hidden_size, num_classes)         #初始化输出层，为一个全链接层

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        print(x.shape)
        logits = self.head(x[:, 0])
        print(logits.shape)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits, attn_weights

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
}
