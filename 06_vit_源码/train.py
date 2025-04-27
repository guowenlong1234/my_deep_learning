# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
#from apex import amp
#from apex.parallel import DistributedDataParallel as DDP

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size


logger = logging.getLogger(__name__)


"""
训练参数：
--name cifar10-100_500 
--dataset cifar10 
--model_type ViT-B_16 
--pretrained_dir checkpoint/ViT-B_16.npz
"""
class AverageMeter(object):
    """Computes and stores the average and current value"""
    #计算并存储平均值和当前值
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0    #当前批次的最新值
        self.avg = 0    #当前所有更新值的加权平均值
        self.sum = 0    #所有更新值的总和
        self.count = 0  #更新值的计数

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean() #检查标签值是否相等，.mean() 是对布尔数组求平均值操作，返回为准确率平均值


def save_model(args, model):#调用时传入参数args, model
    model_to_save = model.module if hasattr(model, 'module') else model     #如果模型进行了分布式训练，可能会被包装，这句意义在于无论模式是否被包装，都可以保存正确的模型
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)   #构建一个模型保存路径
    torch.save(model_to_save.state_dict(), model_checkpoint)#.state_dict()返回模型的所有可学习参数，保存到对应的路径之中
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)     #打印日志消息，提示保存路径


def setup(args):
    # Prepare model
    #取出使用的那个模型
    config = CONFIGS[args.model_type]
    #取出使用的数据集，默认cifar10有10分类，因此cifar10就是10
    num_classes = 10 if args.dataset == "cifar10" else 100

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)#完成了vit模型的初始化，包括输出层，encoder、decoder等内容
    model.load_from(np.load(args.pretrained_dir))       #加载预训练模型
    model.to(args.device)           #将模型装载进GPU中
    num_params = count_parameters(model)#对模型所有的训练参数个数进行计数


    logger.info("{}".format(config))        #写入日志文件
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)

    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    #设置随机初始化种子，保证在每次运行过程中的结果相同。
    random.seed(args.seed)      #random模块随机种子确定
    np.random.seed(args.seed)   #np.random模块随机初始化种子确定
    torch.manual_seed(args.seed)#torch随机初始化种子确定
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step):#调用传入参数args, model, writer, test_loader, global_step
    # Validation!
    eval_losses = AverageMeter()        #实例化一个验证集的损失记录器
    #进行日志操作
    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    #进入验证模式
    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,      #设置一个验证集进度条迭代对象
                          desc="Validating... (loss=X.X)",  #设置内容
                          bar_format="{l_bar}{r_bar}",      #设置布局
                          dynamic_ncols=True,               #允许根据设备调节宽度
                          disable=args.local_rank not in [-1, 0])#非主进程禁用
    loss_fct = torch.nn.CrossEntropyLoss()  #实例化一个损失函数，采用交叉熵函数计算损失
    for step, batch in enumerate(epoch_iterator):   #从迭代器中取出迭代对象的数据
        batch = tuple(t.to(args.device) for t in batch) #将内容装载进gpu，batch中有两个数列，一个存储数据，另外一个存储标签值
        x, y = batch        #x存储数据，y存储标签
        with torch.no_grad():   #进入推理模式，禁用梯度计算
            logits = model(x)[0]    #将x作为输入，输入模型进行推理，返回值为logits, attn_weights，取出第一个（batch，num_classes），表示每个样本数据所属各个类别的概率

            eval_loss = loss_fct(logits, y)     #将预测值与标签值输入损失函数，计算模型损失
            eval_losses.update(eval_loss.item())#更新验证集结果存储

            preds = torch.argmax(logits, dim=-1)    #在最后一个维度上返回最大值所处的索引，方便后续计算准确率

        if len(all_preds) == 0: #判断是不是第一次进入验证阶段，如果是第一次进入验证阶段
            all_preds.append(preds.detach().cpu().numpy())  #将输出结果与计算图断开连接，装载进入cpu，转化成numpy数组格式，再加入到all_preds
            all_label.append(y.detach().cpu().numpy())      #同样的操作，将标签也放入all_label
        else:       #如果不是第一次进入验证阶段
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )       #将本次训练的结果追加到all_label的第一个元素中，如果不是使用追加，会再新建一个列表来存储
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )       #将本次训练的标签追加到all_label的第一个元素中
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)
                    #更新进度条迭代器，显示信息
    all_preds, all_label = all_preds[0], all_label[0]       #取出训练的结果和对应的标签
    accuracy = simple_accuracy(all_preds, all_label)#计算出准确率，结果存储在accuracy中
    #操作日志
    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy #返回验证准确率


def train(args, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:          #判断当前进程是否为主进程，主进程则执行以下代码
        os.makedirs(args.output_dir, exist_ok=True)         #创建输出路径，如果路径已经存在，不会报错
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))     #写入日志文件

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps       #对每个patch进行梯度累计，将每个patch分成很多的微patch，累计后进行反向传播，总体效果等同于每个patch都反向传播，降低了存储压力与运算次数

    # Prepare dataset
    train_loader, test_loader = get_loader(args)        #获取数据集，获得两个dataloader

    # Prepare optimizer and scheduler
    #设置优化器
    optimizer = torch.optim.SGD(model.parameters(),#采用SGD优化器，更新模型的所有参数
                                lr=args.learning_rate,      #设置学习率
                                momentum=0.9,               #设置优化器动量，防止陷入局部最小值，0.9为经验值
                                weight_decay=args.weight_decay)#设置权重衰减值，避免模型权重值过大
    t_total = args.num_steps    #总体步数epoch
    #设置学习率调度器，选择学习率的变化策略
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)#采用余弦衰减策略，有预热过程
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)#采用线性衰减策略，有预热过程
    """
    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())
    """
    # Train!
    #开始训练
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    #日志操作，保存训练相关信息
    model.zero_grad()   #累计梯度清零，开始训练
    set_seed(args)  #固定随机初始化的种子，保证实验的可重复性# Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0    #总步数、最好的准确率初始化为0
    while True:
        model.train()#进入训练模式
        #设置进度条显示，显示当前训练进度等信息
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,       #允许根据终端调整进度条长度
                              disable=args.local_rank not in [-1, 0])#不在主进程则禁用进度条
        for step, batch in enumerate(epoch_iterator):#epoch_iterator是前面创建的进度条迭代器，提供batch，获取batch的数据与索引
            batch = tuple(t.to(args.device) for t in batch)#将当前数据转载金对应的设备中
            x, y = batch    #从batch中取出数据与标签
            loss = model(x, y)  #调用模型，计算损失值，返回损失值

            if args.gradient_accumulation_steps > 1:        #检测是否到达反向传播点
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:       #检测是否采用16位精度
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()     #进行反向传播更新参数

            if (step + 1) % args.gradient_accumulation_steps == 0:#检测是否到达了累计批次数量，如果到达了就执行一此更新
                losses.update(loss.item()*args.gradient_accumulation_steps)#将当前的损失值等信息都更新一下
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  #进行梯度裁剪，防止出现梯度爆炸，参见内容为model.parameters()模型所有参数，参见范围为args.max_grad_norm，超出这个范围进行裁剪
                scheduler.step()        #学习率调度器进行一步
                optimizer.step()        #优化器执行一步
                optimizer.zero_grad()   #优化器累计梯度清零
                global_step += 1        #步数计数

                epoch_iterator.set_description(     #.set_description设置一个可以动态更新的文本标题，显示现在的训练信息，包括损失以及训练进程
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if args.local_rank in [-1, 0]:      #如果是主进程，开始进行日志操作，记录当前损失值，全局步数，学习率等信息
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:   #检查训练轮数是否到达验证集次数，
                    accuracy = valid(args, model, writer, test_loader, global_step) #调用验证函数，运行验证集，返回验证准确率
                    if best_acc < accuracy:     #如果本轮验证集效果更好
                        save_model(args, model) #保存模型
                        best_acc = accuracy     #更新最优模型的准确率
                    model.train()               #恢复训练模式

                if global_step % t_total == 0:  #如果完成了训练，跳出循环
                    break
        losses.reset()  #损失存储器置零
        if global_step % t_total == 0:
            break   #继续跳出，完成训练

    if args.local_rank in [-1, 0]:  #在主进程进行操作
        writer.close()  #关闭日志，保存有所信息
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    #用于设置当前训练运行的名称。
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10",
                        help="Which downstream task.")
    #指定训练数据集
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
    #指出使用的模型
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    #指定预训练模型路径
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    #指定模型输出路径
    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    #指定输入图像尺寸大小
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Total batch size for training.")
    #指定训练集batch大小
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    #指定验证集batch大小
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")
    #指定多少步进行一次验证集验证
    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    #指定学习率大小
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    #设置权重衰减策略
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    #设置epochs
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    #设置学习率衰减方式
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    #设置学习率预热的步数
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    #设置最大梯度范围，超过这个最大梯度范围，将会被裁剪，防止梯度爆炸
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    #指定当前GPU的排序编号
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    #设置随机种子，确保实验的可以复现性
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    #指定梯度累积的步数。每积累一定的步数后，再进行一次梯度更新。
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    #指定是否使用 16 位浮点数（FP16）来代替 32 位浮点数（FP32）。
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    #指定在使用 FP16 时，Apex AMP（Automatic Mixed Precision）优化的级别。不同级别代表不同的混合精度策略。
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    #在使用 FP16 时，用于设置损失缩放的大小，以避免数值不稳定的问题。
    args = parser.parse_args()
    #载入参数

    # Setup CUDA, GPU & distributed training
    #设置多卡训练相关信息
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    #设置日志输出内容与要求
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    #设置随机初始化的种子，保证每次的实验结果相同
    set_seed(args)

    # Model & Tokenizer Setup
    #setup函数生成模型和参数组
    args, model = setup(args)

    # Training
    train(args, model)      #模型训练函数


if __name__ == "__main__":
    main()
