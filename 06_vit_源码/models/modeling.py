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

    def transpose_for_scores(self, x):#三次调用分别输入通过全连接层得到的qkv三个向量
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)#将x的最后一个维度去除，更换为（头数，头尺寸维度）

        x = x.view(*new_x_shape)#将x向量重组为多头

        return x.permute(0, 2, 1, 3)    #交换x向量的维度

    def forward(self, hidden_states):#调用传入参数，经过归一化操作的x

        mixed_query_layer = self.query(hidden_states)   #通过q全连接层生成q向量
                #通过k全连接层生成k向量
        mixed_key_layer = self.key(hidden_states)

        mixed_value_layer = self.value(hidden_states)   #通过v全连接层生成v向量


        query_layer = self.transpose_for_scores(mixed_query_layer)#生成q向量的多头形式（batch，头数，patch，头尺寸）

        key_layer = self.transpose_for_scores(mixed_key_layer)

        value_layer = self.transpose_for_scores(mixed_value_layer)


        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))#用q向量取乘以每一个k向量，得到注意力矩阵

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)#除以根号下的头尺寸，排除因为头尺寸带来的数据大小波动

        attention_probs = self.softmax(attention_scores)    #通过一个softmax激活函数

        weights = attention_probs if self.vis else None     #可视化内容，暂时不管
        attention_probs = self.attn_dropout(attention_probs)#将最后attention输出结果dropout一下


        context_layer = torch.matmul(attention_probs, value_layer)  #最后得到的结果再与v向量相乘

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()#交换矩阵维度，会导致再内存中不是连续的矩阵存储，通过contiguous()方法，将矩阵调整为连续存储空间的矩阵

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)#构建一个新的矩阵形状，取出原来矩阵的前两维度，加上总头尺寸的维度，方便后续将多头合并
        context_layer = context_layer.view(*new_context_layer_shape)#重新调整矩阵维度，将多头合并

        attention_output = self.out(context_layer)#再通过一个全连接层

        attention_output = self.proj_dropout(attention_output)#最终结果dropout一下

        return attention_output, weights#返回通过qkv计算得到的注意力结果。weights与可视化相关，暂时不管


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

    def forward(self, x):#调用传入参数，经过归一化attention操作的x
        x = self.fc1(x) #一个全连接层，两个链接层之间的隐藏层与输入输出维度不相同，为3072
        x = self.act_fn(x)#激活层
        x = self.dropout(x)
        x = self.fc2(x)#再经过一个全连接层，维度重新变回hidden_size
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
        #调用时传入参数x(batch,channel,size,size)
        B = x.shape[0]      #取出batch尺寸
        cls_tokens = self.cls_token.expand(B, -1, -1)   #将cls扩充到每一张图片，其他维度保持不变

        if self.hybrid:     #决定是否使用混合模型，是否经过CNN处理
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)#对x进行一个卷积层操作，分割为每个patch，卷积核大小为patch的尺寸(batch,channel,size,size)->（batch，hiddensize，patch_num,patch_num)

        x = x.flatten(2)#从第二维度开始，后面的维度进行展开，展开为1维（batch，hiddensize，patch_num,patch_num)->（batch，hiddensize，patch_num*patch_num）

        x = x.transpose(-1, -2)#交换最后两个维度不改变数据，（batch，hiddensize，patch_num*patch_num)->（batch，patch_num^2,hiddensize)
        x = torch.cat((cls_tokens, x), dim=1)#将cls_token拼接在x上，拼接维度为第1维度，应保证除了拼接维度以外其他维度相同（batch，patch_num^2,hiddensize)->（batch，patch_num^2+1,hiddensize)

        embeddings = x + self.position_embeddings       #加上位置编码信息，每次输入的图像都可以获取对应的位置编码
        embeddings = self.dropout(embeddings)           #随机杀死些神经元
        return embeddings


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()       #初始化传入参数config, vis
        self.hidden_size = config.hidden_size                           #初始化hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)   #初始化一个归一化层，用于注意力机制的归一化
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)         #初始化一个归一化层，用于ffn
        self.ffn = Mlp(config)           #初始化一个ffn层，其中包括两个线性层，一个激活函数，设置有dropout和初始化参数方法。
        self.attn = Attention(config, vis)  #初始化一个attn注意力机制，包括头数，每个头的尺寸，所有头的尺寸和，计算QKV三个矩阵的全连接层，两个dropout层和激活函数

    def forward(self, x):#调用传入参数为hidden_states
        h = x           #保存当前x用于一会进行残差链接
        x = self.attention_norm(x)      #对x进行一次归一化操作
        x, weights = self.attn(x)#返回通过qkv计算得到的注意力结果，weights与可视化有关，暂时不管
        x = x + h       #进行残差链接

        h = x       #备份当前x，进行残差链接
        x = self.ffn_norm(x)    #经过归一化操作
        x = self.ffn(x) #经过ffn层，其中有两个全连接层，中间夹着一个隐藏层，隐藏层尺寸为3017，有激活函数与dropout
        x = x + h       #进行残差链接
        return x, weights#weight与可视化相关

    def load_from(self, weights, n_block):#这个函数完成了对所有的block权重项的加载
        ROOT = f"Transformer/encoderblock_{n_block}"#一个动态字符串，会被替换成n_block
        with torch.no_grad():       #禁止梯度更新
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
            """#加载kqv与输出层3个矩阵的权重参数
            query_weight = np2th(weights[ROOT + "/" + ATTENTION_Q + "/" + "kernel"]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[ROOT + "/" +  ATTENTION_K+ "/" + "kernel"]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[ROOT + "/" +  ATTENTION_V+"/" + "kernel"]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[ROOT + "/" + ATTENTION_OUT+"/" + "kernel"]).view(self.hidden_size, self.hidden_size).t()
            #加载kqv与输出层三层的偏置参数
            query_bias = np2th(weights[ROOT + "/" +  ATTENTION_Q+"/" + "bias"]).view(-1)
            key_bias = np2th(weights[ROOT + "/" +  ATTENTION_K+"/" + "bias"]).view(-1)
            value_bias = np2th(weights[ROOT + "/" +  ATTENTION_V+"/" + "bias"]).view(-1)
            out_bias = np2th(weights[ROOT + "/" +  ATTENTION_OUT+"/" + "bias"]).view(-1)
            #将上面得到的权重与偏置加载到当前模型中
            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)
            #加载mlp权重参数与偏置参数
            mlp_weight_0 = np2th(weights[ROOT + "/" +  FC_0+"/" + "kernel"]).t()
            mlp_weight_1 = np2th(weights[ROOT + "/" +  FC_1+"/" + "kernel"]).t()
            mlp_bias_0 = np2th(weights[ROOT + "/" +  FC_0+"/" +"bias"]).t()
            mlp_bias_1 = np2th(weights[ROOT + "/" +  FC_1+"/" +"bias"]).t()
            #加载ffn偏置参数与权重参数
            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)
            #加载两个归一化层的权重和偏置参数
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

    def forward(self, hidden_states):#调用传入信息为embedding_output，包括cls和位置编码的序列

        attn_weights = []   #用于可视化展示的参数，暂时不管
        for layer_block in self.layer:#模型容器中有几层block就执行几层
            hidden_states, weights = layer_block(hidden_states)#输出为每个block输出，输入-归一化-attention-归一化-ffn执行一次
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)#最后的encoded归一化操作
        return encoded, attn_weights    #attn_weights用于可视化展示，暂时不管最终返回经过多个block之后的encoded


class Transformer(nn.Module):
    #在VisionTransformer中调用
    #输入数据为x
    #返回值为x，x, attn_weights
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)     #完成Embeddings的初始化操作，主要有窗口卷积层，位置编码层，cls_token的初始化
        self.encoder = Encoder(config, vis)     #初始化Encoder，包括8个block层，一个归一化层

    def forward(self, input_ids):#调用时传入参数x
        embedding_output = self.embeddings(input_ids)       #返回已经完成编码的信息序列（batch，patch_num+1,hiddensize）其中包括cls和位置编码信息
        encoded, attn_weights = self.encoder(embedding_output)  #最终返回经过多个block的encoded，attn_weights为可视化参数，暂时为0
        return encoded, attn_weights


class VisionTransformer(nn.Module):
    #调用时的传入参数：config模型类型, args.img_size图像尺寸大小, zero_head=True, num_classes=num_classes数据集类型，默认10
    #返回值model
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes      #分类的任务数量，默认为使用cifer10数据集的10分类任务
        self.zero_head = zero_head          #控制是否加载头部层权重，也就是输出层的权重
        self.classifier = config.classifier #token？，最终的分类头是token吗，表示最终分类的依据

        self.transformer = Transformer(config, img_size, vis)       #完成transform的主要初始化，包括Embeddings和Encoder
        self.head = Linear(config.hidden_size, num_classes)         #初始化输出层，为一个全链接层

    def forward(self, x, labels=None):  #调用时传入两个参数（x，y）
        x, attn_weights = self.transformer(x)   #最终返回经过多个block的encoded，attn_weights为可视化参数，暂时为0
        logits = self.head(x[:, 0])         #取出cls_token，经过全连接层进行十分类

        if labels is not None:#如果没有输入标签值，说明是在验证阶段，直接返回十分类结果即可，有标签值说明在训练模式
            loss_fct = CrossEntropyLoss()   #计算损失函数
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))#logits.view(-1, self.num_classes)将logits变成一个二维的张量，第二个维度为num_classes，其余维度通过计算获取
            return loss     #返回损失值
        else:
            return logits, attn_weights     #在训练集，直接返回预测概率值

    def load_from(self, weights):
        with torch.no_grad():       #禁用梯度计算
            if self.zero_head:      #判断是否加载需要头部层的权重参数
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)#将输出层的所有权重和偏置设置为0
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())       #np2th从 numpy 数组转换为 PyTorch 张量，通过copy_方法赋值给self.head.weight..t()表示矩阵转置，确保维度相同。加载输出层权重
                self.head.bias.copy_(np2th(weights["head/bias"]).t())           #加载输出层偏置

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))    #加载embedding层权重
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))#加载cls权重
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))    #加载归一化层权重
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])     #加载位置编码权重
            posemb_new = self.transformer.embeddings.position_embeddings            #取出当前模型中的位置编码
            if posemb.size() == posemb_new.size():      #如果位置编码维度相同，直接复制，维度不同，需要做一些调整操作
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))#打印加载的原始位置嵌入（posemb）和调整后的位置嵌入（posemb_new）的尺寸
                ntok_new = posemb_new.size(1)       #从位置编码信息中取出第1个维度，代表了token的数量，在当前模型中是197

                if self.classifier == "token":      #如果分类头是token，要取出cls_token存在posemb_tok，剩余部分存在posemb_grid
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]#切片操作[start:end:step]
                    ntok_new -= 1                   #取出来一个，维度少1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]#取出第一个维度的元素，分类头不是token的话，只是用posemb_grid

                gs_old = int(np.sqrt(len(posemb_grid)))     #通过token维度计算网格尺寸，表明原来图像每行每列有多少个patch
                gs_new = int(np.sqrt(ntok_new))             #现在的每行每列应该有14个patch
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)#将原来的矩阵维度变更成24*24*hiddensize的维度

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)#需要对网格进行缩放，计算每个维度的缩放因子
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)#使用ndimage.zoom函数对posemb_grid进行的缩放（缩放矩阵，缩放因子，插值方法）order=1表示插值方式为线性插值）
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)#重新将矩阵patch维度变回1维
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))#重新将头部cls拼接上去，拼接维度为1维度，在将数据载入模型

            for bname, block in self.transformer.encoder.named_children():  #named_children()它会返回该模块下所有子模块的名称和模块本身的元组
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)  #对每一个block加载权重项

            if self.transformer.embeddings.hybrid:#以下是混合模型采用的方法
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
