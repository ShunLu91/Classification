import sys
import argparse
from utils import *
import torch.nn as nn
import pretrainedmodels
from thop import profile
from datetime import datetime
from torchvision import models
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from model import MobileNetV2

# 解析命令行参数
parser = argparse.ArgumentParser('Train model')
parser.add_argument('--exp_name', type=str, required=True, help='search model name')
parser.add_argument('--classes', type=int, default=2, help='num of MB_layers')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='num of epochs')
parser.add_argument('--seed', type=int, default=2020, help='seed')
parser.add_argument('--learning_rate', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--learning_rate_min', type=float, default=1e-8, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--train_interval', type=int, default=1, help='train to print frequency')
parser.add_argument('--val_interval', type=int, default=5, help='evaluate and save frequency')
parser.add_argument('--dropout', type=float, default=0.5, help='drop out rate')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--gpu', type=int, default=7, help='gpu id')
parser.add_argument('--resume', type=bool, default=False, help='resume')
parser.add_argument('--pretrained', type=str, default=None, help='pretrained')
# ******************************* dataset *******************************#
parser.add_argument('--dataset', type=str, default='imagenet', help='[cifar10, imagenet]')
parser.add_argument('--data_dir', type=str, default=None, help='dataset dir')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--resize', action='store_true', default=False, help='use resize')

args = parser.parse_args()
print(args)
# tensorboard文档记录
train_writer = SummaryWriter(log_dir='./writer/' + args.exp_name + '/Train')
val_writer = SummaryWriter(log_dir='./writer/' + args.exp_name + '/Val')


# 训练代码
def train(args, epoch, train_data, device, model, criterion, optimizer, scheduler):
    model.train() # 模型设置为训练模式
    train_loss = 0.0
    top1 = AvgrageMeter() # 定义两个精度记录类
    for step, (inputs, targets) in enumerate(train_data):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad() # 优化器梯度清空
        if args.exp_name == 'inception_v3':
            (outputs, aux) = model(inputs) # 计算模型输出
        else:
            outputs = model(inputs)
        loss = criterion(outputs, targets)  # 计算损失
        loss.backward() # 梯度反传
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) # 梯度裁剪
        prec1 = accuracy(outputs, targets, topk=(1, )) # 计算准确率
        n = inputs.size(0) # 输出数据尺寸
        top1.update(prec1[0], n) # 更新准确率
        optimizer.step() # 调整模型参数
        train_loss += loss.item()
        dt = datetime.now()
        # 输出显示
        sys.stdout.write("\r{0} {1}  epoch:{2}/{3}, batch:{4}/{5}, lr:{6}, loss:{7}, top1:{8}".format(
            dt.strftime('%x'), dt.strftime('%X'), '%.4d' % (epoch + 1), '%.4d' % args.epochs,
                                                  '%.4d' % (step + 1), '%.4d' % len(train_data),
                                                  '%.5f' % scheduler.get_lr()[0],
                                                  '%.6f' % loss, '%.3f' % top1.avg))
        sys.stdout.flush()
    train_writer.add_scalar('Loss', train_loss / (step + 1), epoch) # tensorboard文档写入
    train_writer.add_scalar('Acc', top1.avg, epoch)

    return train_loss / (step + 1), top1.avg


# 验证代码，运行流程同上
def validate(epoch, val_data, device, model):
    model.eval()
    val_loss = 0.0
    val_top1 = AvgrageMeter()
    criterion = nn.CrossEntropyLoss().to(device)

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_data):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            prec1 = accuracy(outputs, targets, topk=(1, ))
            n = inputs.size(0)
            val_top1.update(prec1[0], n)
    val_writer.add_scalar('Loss', val_loss / (step + 1), epoch)
    val_writer.add_scalar('Acc', val_top1.avg, epoch)

    return val_top1.avg, val_loss / (step + 1)


def main():
    # 定义设备
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        device = torch.device("cuda")

    # 设置随机数种子保证实验可复现
    set_seed(args.seed)

    # 定义损失函数和模型
    criterion = nn.CrossEntropyLoss().to(device)
    if args.exp_name == 'mobilenetv2':
        model = MobileNetV2(num_classes=args.classes)
    elif args.exp_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        # 改写全连接类别的输出
        fc_inputs = model.fc.in_features
        model.fc = nn.Linear(fc_inputs, args.classes)
    elif args.exp_name == 'inception_v3':
        model = models.inception_v3(pretrained=True)
        # 改写全连接类别的输出
        model.fc = nn.Linear(2048, args.classes)
        model.AuxLogits.fc = nn.Linear(768, args.classes)
    elif args.exp_name == 'xception':
        model = pretrainedmodels.__dict__['xception'](pretrained='imagenet')
        fc_inputs = 2048
        model.fc = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(fc_inputs, args.classes)
        )

    # pretrained
    # 如果有预训练模型可以直接加载
    if args.pretrained:
        pretrained_dict = torch.load(args.pretrained, map_location=device)
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    # 模型放到GPU/cpu设备
    model = model.to(device)
    # 计算模型的参数和计算量
    flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224).to(device),), verbose=False)
    print('Model: FLOPs={}M, Params={}M'.format(flops / 1e6, params / 1e6))

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters())
    # 定义学习率衰减策略
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min, last_epoch=-1)
    # 定义训练集和验证集的图片变换方法，以及加载图片（pytorch固定操作）
    train_transform, valid_transform = data_transforms(args)
    trainset = dset.ImageFolder(root=os.path.join(args.data_dir, 'train'), transform=train_transform)
    print('INFO:', trainset.class_to_idx)
    train_queue = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=8, pin_memory=True)
    valset = dset.ImageFolder(root=os.path.join(args.data_dir, 'test'), transform=valid_transform)
    valid_queue = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=8, pin_memory=True)
    print('Dataset: Train={}, Val={}'.format(len(trainset), len(valset)))

    # 如果要继续上次的训练，可以加载上次训练保存的模型（类似加载预训练模型）
    if args.resume:
        resume_path = './snapshots/{}_train_states.pt.tar'.format(args.exp_name)
        if os.path.isfile(resume_path):
            print("Loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            model.load_state_dict(checkpoint['supernet_state'])
            scheduler.laod_state_dict(checkpoint['scheduler_state'])
        else:
            raise ValueError("No checkpoint found at '{}'".format(resume_path))
    else:
        start_epoch = 0

    best_acc = 0.0
    path = './snapshots/{}_train_states.pt.tar'.format(args.exp_name)
    if not os.path.exists('snapshots'):
        os.mkdir('snapshots')
    # 开始迭代训练和验证
    for epoch in range(start_epoch, args.epochs):
        t1 = time.time()

        # train
        train(args, epoch, train_queue, device, model, criterion=criterion, optimizer=optimizer, scheduler=scheduler)
        scheduler.step()

        # validate
        val_top1, val_obj = validate(epoch, val_data=valid_queue, device=device, model=model)
        elapse = time.time() - t1
        h, m, s = eta_time(elapse, args.epochs - epoch - 1)

        # save best model
        if val_top1 > best_acc:
            best_acc = val_top1
            # save the states of this epoch
            state = {
                'epoch': epoch,
                'args': args,
                'optimizer_state': optimizer.state_dict(),
                'supernet_state': model.state_dict(),
                'scheduler_state': scheduler.state_dict()
            }
            torch.save(state, path)
            # print('\n best val acc: {:.6}'.format(best_acc))
        print('\nval: loss={:.6}, top1={:.6}, best={:.6}, elapse={:.0f}s, eta={:.0f}h {:.0f}m {:.0f}s\n'
              .format(val_obj, val_top1, best_acc, elapse, h, m, s))

    print('Best Top1 Acc: {:.6}'.format(best_acc))


if __name__ == '__main__':
    # 程序从这里开始运行
    start = time.time() # 记录时间
    main() # 主函数
    # tensorboard文档关闭
    train_writer.close()
    val_writer.close()
    # 记录并汇报时间
    time_record(start)
