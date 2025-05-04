# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor







'''
--cfg configs/swin_tiny_patch4_window7_224.yaml
--data-path imagenet
--local_rank 0
--model name: swin_base_patch4_window7_224
--local_rank 0
--batch-size 16
'''

def parse_option():     #设置允许接受的命令行参数
    #实例化一个命令行解析器对象，程序描述，add_help=False不允许自动添加帮助信息
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    #指定配置文件路径，必须的添加的
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    #指定是否使用opt来添加键值对，默认为None，nargs='+'表示可以跟一个或者多个值，+表示至少需要一个值，传递的值将被解析为一个列表
    parser.add_argument("--opts",help="Modify config options by adding 'KEY VALUE' pairs. ",default=None,nargs='+',)

    # easy config modification
    #设置batch_size
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    #设置数据集路径
    parser.add_argument('--data-path', type=str, help='path to dataset')
    #指定是否使用了压缩文件zip
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    #指定数据集如何缓存。默认part: 将数据集分割成不重叠的部分，仅缓存其中一部分。
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    #从一个特定文件初恢复训练
    parser.add_argument('--resume', help='resume from checkpoint')
    #累计多少步更新一次权重
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    #是否使用梯度检查点，布尔类型
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    #混合精度训练的优化级别
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    #指定输出文件的根目录
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    #标记实验的标签
    parser.add_argument('--tag', help='tag of experiment')
    #只执行评估，不执行训练
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    #只进行吞吐量测试
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    #用于分布式训练时的本地进程编号
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    #解析命令行参数，将未定义的参数返回一个列表
    args, unparsed = parser.parse_known_args()

    config = get_config(args)   #将aegs中的参数全部装入congfig对象中，并且config文件锁定冻结，防止后续运行中被修改

    return args, config


def main(config):
    #调用build_loader函数，生成训练数据和验证数据的dataset与data_loader
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)#mixup_fn是一种数据增强策略

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)     #实例化一个模型
    model.cuda()        #将模型加载到GPU上
    logger.info(str(model))         #输出日志关于模型的信息

    optimizer = build_optimizer(config, model)  #实例化一个优化器，传入参数为配置文件和模型

    # if config.AMP_OPT_LEVEL != "O0":
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)

    #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    #model_without_ddp = model.module

    #统计需要学习的参数量并输出
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")


    """
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")
    """
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train)) #设置学习率衰减的策略

    if config.AUG.MIXUP > 0.:   #判断是否采用maxup的数据增强策略
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()    #使用了maxup数据增强就采用SoftTargetCrossEntropy()作为损失函数
    elif config.MODEL.LABEL_SMOOTHING > 0.: #判断是否采用了标签平滑损失函数，目的是避免模型过度自信地预测某一类别，从而提升模型的泛化能力
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)  #LabelSmoothingCrossEntropy 来计算损失。该损失函数的作用是将每个标签的目标概率分布略微平滑（例如，将 1 转为 0.9，将 0 转为 0.1），从而减少模型的过拟合风险。
    else:
        criterion = torch.nn.CrossEntropyLoss()     #如果两个都没有采用，就是用普通的交叉熵函数

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        '''
        这段代码的目的是在训练过程中自动恢复模型的状态。如果配置中启用了 AUTO_RESUME 选项，
        它将通过 auto_resume_helper 函数自动查找并加载最近的检查点（checkpoint），从而恢复模型的训练进度。
        '''
        resume_file = auto_resume_helper(config.OUTPUT)     #如果有最近的检查点，则返回检查点文件，如果没有，则返回none
        if resume_file: #如果有检查点文件
            if config.MODEL.RESUME: #如果允许使用检查点文件
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
                #日志输出采用的检查点文件
            config.defrost()    #解冻配置文件
            config.MODEL.RESUME = resume_file   #将检查点文件写入配置文件中
            config.freeze()     #冻结配置文件
            logger.info(f'auto resuming from {resume_file}')    #输出一条信息，从某个检查点开始训练
        else:   #没有检查点文件
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')    #日志输出没有检查点文件

    if config.MODEL.RESUME: #如果采用检查点配置文件
        max_accuracy = load_checkpoint(config, model, optimizer, lr_scheduler, logger)  #加载检查点函数，加载模型数据，优化器，学习率调度器等信息，返回检查点文件中保存的最大准确率数据

        #return acc1_meter.avg, acc5_meter.avg, loss_meter.avg
        acc1, acc5, loss = validate(config, data_loader_val, model)     #使用验证集验证模型，返回损失以及准确率
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")      #输出日志，记录断点最初的损失和准确率
        if config.EVAL_MODE:    #是否开启了评估模式，如果开启了，只验证不训练，main函数直接返回
            return

    if config.THROUGHPUT_MODE:      #判断是否处于判断吞吐量模式
        throughput(data_loader_val, model, logger)
        return

    #输出日志，开始训练
    logger.info("Start training")

    #记录训练开始时间
    start_time = time.time()

    #从配置文件中读取一共需要多少轮迭代，并执行
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):

        #这行代码用于在分布式训练中更新数据加载器（DataLoader）的 Sampler，以确保每个 epoch 都能正确地从数据集采样数据。
        data_loader_train.sampler.set_epoch(epoch)

        #执行函数，训练1个epoch
        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler)
        #判断是否需要保存模型，dist.get_rank() == 0 当前进程为主进程，config.SAVE_FREQ控制保存频率，，或者最后一个epoch
        if  (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger)

        #执行验证函数，返回验证结果参数
        acc1, acc5, loss = validate(config, data_loader_val, model)

        #通过日志输出当前验证结果
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")

        #判断当前验证结果是否好于所有的验证结果，并更新输出
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    #计算模型总共训练的时间
    total_time = time.time() - start_time

    #输出总共的执行时间
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler):
    model.train()   #进入训练模式
    optimizer.zero_grad()   #梯度累计清零

    num_steps = len(data_loader)    #计算总共需要的步数
    batch_time = AverageMeter()     #实例化三个数据统计类用于统计本轮训练过程的参数
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()             #记录开始时间
    end = time.time()               #记录结束时间
    for idx, (samples, targets) in enumerate(data_loader):  #对dataloader进行迭代
        samples = samples.cuda(non_blocking=True)   #将数据装载如gpu中
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:        #判断是否需要mixup数据增强
            samples, targets = mixup_fn(samples, targets)       #如果需要则进行数据增强

        outputs = model(samples)        #计算模型输出，返回值为一个张量[16,3,224,224] ->[16, 768] -> [16, 1000]

        if config.TRAIN.ACCUMULATION_STEPS > 1:     #检查是否使用了梯度累计，默认不使用，即为每次训练之后都进行梯度更新
            loss = criterion(outputs, targets)      #计算损失函数
            loss = loss / config.TRAIN.ACCUMULATION_STEPS   #计算平均损失
        else:
            loss = criterion(outputs, targets)      #计算损失函数
            optimizer.zero_grad()       #梯度清零

            loss.backward() #反向传播

            if config.TRAIN.CLIP_GRAD:  #判断是否有梯度裁剪参数
                #进行梯度裁剪防止梯度爆炸
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)

            else:

                grad_norm = get_grad_norm(model.parameters())
                #计算梯度范围的函数

            optimizer.step()    #优化器进行迭代
            lr_scheduler.step_update(epoch * num_steps + idx)       #调整学习率

        torch.cuda.synchronize()        #同步各个进程时间
        #更新统计数据
        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()   #记录结束时间

        if idx % config.PRINT_FREQ == 0:    #判断是否达到输出步骤，如果达到进行输出
            lr = optimizer.param_groups[0]['lr']   #计算当前lr
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) #计算总共使用的显存
            etas = batch_time.avg * (num_steps - idx)   #计算预计剩余时间
            logger.info(#输出日志信息
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
            break
    epoch_time = time.time() - start    #计算一轮训练总时间
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")  #打印日志信息，输出结果


@torch.no_grad()    #装饰器，用于禁止模型进行梯度更新
def validate(config, data_loader, model):
    '''
    这是模型的验证函数
    '''
    criterion = torch.nn.CrossEntropyLoss()     #定义损失函数为交叉熵函数
    model.eval()    #模型进入验证模式

    #AverageMeter 是一个常用的工具，用于在训练或验证过程中计算并存储各种统计数据（如时间、损失、准确率等）的平均值。在每个批次结束时，都会更新这些统计数据。
    #将AverageMeter类实例化四个
    batch_time = AverageMeter() #记录每个批次的时间
    loss_meter = AverageMeter() #记录损失值
    acc1_meter = AverageMeter() #记录top-1 准确率（Top-1 准确率指的是模型预测的类别与实际类别完全匹配的比例。）
    acc5_meter = AverageMeter() #记录top-5 准确率（Top-5 准确率指的是模型预测的前 5 个类别中包含实际类别的比例，常用于多类别分类任务的评估。）

    end = time.time()   #记录当前时间

    for idx, (images, target) in enumerate(data_loader):    #对data_loader迭代，每次返回索引，数据，以及标签
        images = images.cuda(non_blocking=True) #将数据移动到GPU上，不阻塞代码运行
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)  #计算模型的输出

        # measure accuracy and record loss  计算精度以及计算损失
        loss = criterion(output, target)    #计算损失函数
        acc1, acc5 = accuracy(output, target, topk=(1, 5))  #计算acc1, acc5的准确率
        #accuracy(output, target, topk=(1, 5))，（输出值，实际值，top几的准确率）

        #对 acc1 进行归约操作。通常，这种操作是在分布式训练中使用的，用来在多 GPU 或多机器的情况下，合并来自各个设备的计算结果。
        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        #更新模型损失，acc1、acc5准确率
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        #计算这轮batch验证消耗的时间

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:    #是否到了指定轮数输出结果
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)     #计算存储量使用情况
            #更新日志，输出对应的训练信息
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')

    #返回本次训练的平均准确率，返回平均损失
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):     #输入参数data_loader_val, model, logger
    model.eval()        #验证模式

    for idx, (images, _) in enumerate(data_loader):     #迭代dataloader，取出索引与图像数据
        images = images.cuda(non_blocking=True)     #图像数据并行载入gpu
        batch_size = images.shape[0]
        for i in range(50):
            model(images)   #执行50次模型获取输出
        torch.cuda.synchronize()        #同步gpu的操作，确保所有进程都完成运算
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):         #在执行30次
            model(images)
        torch.cuda.synchronize()        #同步gpu的操作，确保所有进程都完成运算
        tic2 = time.time()              #
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        #输出执行30次所需要的平均时间，写入日志文件
        return


if __name__ == '__main__':
    _, config = parse_option()      #加载命令行参数，返回参数args, config，config中已经加载有args中的全部参数，且已经被冻结防止修改
    """
    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"
    """

    # 用于整理多机多卡进程os.environ访问环境变量，Rank是当前进程的标号，WORLD_SIZE是总进程数和节点数
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:     #如果存在这两个值，说明有多机多线程训练
        rank = int(os.environ["RANK"])          #提取rank，转化为int
        world_size = int(os.environ['WORLD_SIZE'])  #提取WORLD_SIZE，转化为int
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")   #输出当前进程排名和总进程数
    else:       #不存在这两个值，说明是单机单卡训练
        rank = -1   #将这两个值设置为-1，表示只有主进程一个进程
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)    #设置每个进程使用哪个设备
    # torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    # torch.distributed.barrier()

    seed = config.SEED #获取随机初始化的种子，保证每次实验随机初始化的值是相同的#+ dist.get_rank()
    torch.manual_seed(seed) #设置pythoch种子
    np.random.seed(seed)    #设置numpy种子
    cudnn.benchmark = True  #一种优化算法，在输入尺寸固定的情况下可以优化卷积计算性能，当尺寸不固定时，不要用

    # linear scale the learning rate according to total batch size, may not be optimal
    #根据总批次大小线性调整学习率，可能不是最优解
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE #catch*基础lr作为总lr。      * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE #设置启动步数       * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE #根据patch设置最小lr      * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    #根据梯度累计更新学习率
    if config.TRAIN.ACCUMULATION_STEPS > 1:#如果设置了梯度累计
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS       #基础学习率、启动步数、最小学习率均根据累计步数进行扩大
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()    #配置文件解冻
    config.TRAIN.BASE_LR = linear_scaled_lr     #将学习率相关的三个参数写入配置文件中
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze() #重新冻结配置文件

    os.makedirs(config.OUTPUT, exist_ok=True)
    #logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")
    #创建一个日志记录器，保存日志操作
    """
    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    """
    main(config)    #传入配置文件，调用主函数

