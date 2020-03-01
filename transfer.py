import os
import sys
import time
import torch
import argparse
import torch.nn as nn
from datetime import datetime
from torchsummary import summary
from thop import profile
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from utils import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser('Train signal model')
parser.add_argument('--exp_name', type=str, default='transfer', help='search model name')
parser.add_argument('--classes', type=int, default=9, help='num of MB_layers')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--epochs', type=int, default=600, help='num of epochs')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--learning_rate_min', type=float, default=1e-8, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--train_interval', type=int, default=1, help='train to print frequency')
parser.add_argument('--val_interval', type=int, default=5, help='evaluate and save frequency')
parser.add_argument('--smooth', type=float, default=0.0, help='label smoothing')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='drop out rate')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop_path_prob')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--gpu', type=int, default=4, help='gpu id')
parser.add_argument('--resume', type=bool, default=False, help='resume')
# ******************************* dataset *******************************#
parser.add_argument('--dataset', type=str, default='imagenet')
parser.add_argument('--data_dir', type=str, default='/home/lushun/dataset/mushroom/valAndTrain')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')

args = parser.parse_args()
print(args)


def pretrained_model(name, classes):
    if name == 'resnet50':
        network = models.resnet50(pretrained=True)  # 调用预训练好的RestNet模型
    elif name == 'xception':
        network = models.xception(pretrained=True)  # 调用预训练好的RestNet模型

    # freeze params
    # for param in network.parameters():
    #     param.requires_grad = False
    fc_inputs = network.fc.in_features
    network.fc = nn.Sequential(
        nn.Dropout(args.dropout_rate),
        nn.Linear(fc_inputs, classes)
    )

    return network

def train(args, epoch, train_data, device, model, criterion, optimizer, scheduler):
    model.train()
    train_loss = 0.0
    drop_path_prob = 0.0
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    if args.drop_path_prob > 0.0:
        drop_path_prob = args.drop_path_prob * (epoch + 1) / args.epochs

    for step, (inputs, targets) in enumerate(train_data):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        n = inputs.size(0)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        optimizer.step()
        train_loss += loss.item()
        dt = datetime.now()
        sys.stdout.write("\r{0} {1}  epoch:{2}/{3}, batch:{4}/{5}, lr:{6}, loss:{7}, top1:{8}, top5:{9}".format(
            dt.strftime('%x'), dt.strftime('%X'), '%.4d' % (epoch+1), '%.4d' % args.epochs,
            '%.4d' % (step + 1), '%.4d' % len(train_data), '%.5f' % scheduler.get_lr()[0],
            '%.6f' % loss, '%.3f' % top1.avg, '%.3f' % top5.avg))
        sys.stdout.flush()

    return train_loss/(step+1), top1.avg, top5.avg


def validate(epoch, val_data, device, model):
    model.eval()
    val_loss = 0.0
    val_top1 = AvgrageMeter()
    val_top5 = AvgrageMeter()
    criterion = nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_data):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            val_top1.update(prec1.item(), n)
            val_top5.update(prec5.item(), n)

    return val_top1.avg, val_top5.avg, val_loss / (step + 1)


if __name__ == '__main__':
    # seed
    set_seed(args.seed)

    # prepare dir
    if not os.path.exists('./snapshots'.format(args.exp_name)):
        os.mkdir('./snapshots'.format(args.exp_name))

    # device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        device = torch.device("cuda")


    image_transforms = {
        'train30n': transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
            # transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            # transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'valid30': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_transform, valid_transform = data_transforms(args)
    train_data = datasets.ImageFolder(root=os.path.join(args.data_dir, 'train'), transform=image_transforms['train30n'])
    val_data = datasets.ImageFolder(root=os.path.join(args.data_dir, 'val'), transform=image_transforms['train30n'])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                              shuffle=True, num_workers=8, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
                                              shuffle=False, num_workers=8, pin_memory=True)
    print('train_data:{}, val_data:{}'.format(len(train_data), len(val_data)))

    # train_directory = os.path.join(args.data_dir, 'train')
    # valid_directory = os.path.join(args.data_dir, 'val')
    #
    # data = {
    #     'train30n': datasets.ImageFolder(root=train_directory, transform=image_transforms['train30n']),
    #     'valid30': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid30'])}
    #
    # train_data_size = len(data['train30n'])
    # valid_data_size = len(data['valid30'])
    #
    # train_loader = DataLoader(data['train30n'], batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    # valid_loader = DataLoader(data['valid30'], batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    model = pretrained_model('resnet50', classes=args.classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    summary(model, (3, 224, 224))
    flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224).to(device),), verbose=False)
    print('FLOPs: {}, params: {}'.format(flops / 1e6, params / 1e6))

    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     args.learning_rate,
    #     momentum=args.momentum,
    #     weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min, last_epoch=-1)


    best_acc = 0.0
    for epoch in range(0, args.epochs):
        t1 = time.time()

        # train
        train(args, epoch, train_loader, device, model, criterion=criterion, optimizer=optimizer, scheduler=scheduler)
        scheduler.step()

        # validate
        val_top1, val_top5, val_obj = validate(epoch, val_data=valid_loader, device=device, model=model)
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
            path = './snapshots/{}_transfer_states.pt.tar'.format(args.exp_name)
            torch.save(state, path)
            # print('\n best val acc: {:.6}'.format(best_acc))
        print('\nval: loss={:.6}, top1={:.6}, top5={:.6}, best={:.6}, elapse={:.0f}s, eta={:.0f}h {:.0f}m {:.0f}s\n'
              .format(val_obj, val_top1, val_top5, best_acc, elapse, h, m, s))
    print('Best Top1 Acc: {:.6}'.format(best_acc))
