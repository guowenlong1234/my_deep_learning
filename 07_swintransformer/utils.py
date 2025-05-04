# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    '''
    这段代码定义了一个名为 load_checkpoint 的函数，用于从保存的检查点（checkpoint）文件加载模型、
    优化器、学习率调度器等状态，并恢复训练的过程。
    '''
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    # 检查检查点文件是否以https开头，如果是则是通过url下载的，调用torch.hub.load_state_dict_from_url来加载检查点
    if config.MODEL.RESUME.startswith('https'):
        #map_location='cpu' 表示加载到 CPU 上
        #check_hash=True检查验证文件的哈希值
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:   #如果不是通过url下载的检查点
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')    #使用torch.load函数加载检查点

    #checkpoint['model'] 是从检查点中获取的模型权重（通常是一个字典，包含模型的参数）
    #model.load_state_dict() 方法将加载这些参数到模型中。strict=False 允许加载时忽略一些缺失的键（例如，某些层的权重或优化器状态等）
    msg = model.load_state_dict(checkpoint['model'], strict=False)

    logger.info(msg)

    #记录模型最大准确率的变量
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        #如果不处于评估模式（config.EVAL_MODE 是 False），并且检查点中包含优化器（optimizer）、学习率调度器（lr_scheduler）和训练的当前轮次（epoch）信息，则会恢复这些训练状态。
        optimizer.load_state_dict(checkpoint['optimizer'])  #加载优化器
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])    #加载学习率调度器
        config.defrost()    #解锁配置文件
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1  #将模型中的epoch设置为检查点epoch+1
        config.freeze() #冻结配置文件

        #如果启用了混合精度，则回复混合精度相关参数
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        #输出一条日志，记录恢复成功并且输出还需要训练的轮数

        #如果模型中保存了最高准确率，则将其值读出来
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    # 删除加载的检查点数据，清理缓存，防止内存泄漏
    return max_accuracy #返回最大准确率


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),      #构造一个字典，用于存储需要保存的内容
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if config.AMP_OPT_LEVEL != "O0":
        #是否采用了混合精度模式
        save_state['amp'] = amp.state_dict()

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')  #构造保存路径
    logger.info(f"{save_path} saving......")    #打印日志信息
    torch.save(save_state, save_path)           #保存模型
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    '''
    这个函数的作用是计算给定参数的梯度范数
    '''
    #如果输入的是单个张量，则将其转化为列表进行统一处理
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    #过滤掉所有没有梯度的参数
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    #确保参数为浮点数
    norm_type = float(norm_type)

    #初始化总范数
    total_norm = 0

    #遍历所有的参数
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)        #计算当前参数的范数
        total_norm += param_norm.item() ** norm_type    #计算范数类型幂

    #返回结果
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    '''
    这是一个辅助函数，它的作用是帮助找到最近的检查点文件，并返回该文件的路径。config.OUTPUT
    是保存输出文件的根目录，通常包含训练过程中的检查点文件（checkpoint），日志文件等。
    '''
    checkpoints = os.listdir(output_dir)
    #该函数返回指定目录 (output_dir) 中的所有文件和子目录的名称，返回的是一个字符串列表。
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    #这是一个列表推导式，目的是筛选出所有以 .pth 后缀结尾的文件。ckpt.endswith('pth') 会检查每个文件名是否以 .pth 结尾。
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:    #如果找到了检查点的pth文件
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)   #从中找到最新的，时间最晚的检查点文件
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None      #如果没有找到检查点文件，则返回none
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt
