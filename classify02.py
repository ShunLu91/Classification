import sys
import argparse
from utils import *
import torch.nn as nn
import scipy.io as scio
from thop import profile
from datetime import datetime
import torchvision.datasets as dset
from torchvision import models
from torchsummary import summary
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser('Train model')
parser.add_argument('--exp_name', type=str, required=True, help='search model name')
parser.add_argument('--classes', type=int, default=4, help='num of MB_layers')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
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
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
parser.add_argument('--resume', type=bool, default=False, help='resume')
parser.add_argument('--pretrained', type=str, default=None, help='pretrained')
# ******************************* dataset *******************************#
parser.add_argument('--dataset', type=str, default='imagenet', help='[cifar10, imagenet]')
parser.add_argument('--data_dir', type=str, default='/Users/lushun/Documents/dataset/brain_wave/brain_wave.mat'
                    , help='dataset dir')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--resize', action='store_true', default=False, help='use resize')

args = parser.parse_args()
print(args)
train_writer = SummaryWriter(log_dir='./writer/' + args.exp_name + '/Train')
val_writer = SummaryWriter(log_dir='./writer/' + args.exp_name + '/Val')


class BrainWave_Set(Dataset):
    """
    BrainWave_Set
    """

    def __init__(self, args, training):
        super(BrainWave_Set, self).__init__()

        # self.train_transform, self.valid_transform = data_transforms(args)

        self.training = training
        self.train_data, self.val_data, self.train_label, self.val_label= self.load_data(args.data_dir)

    def load_data(self, data_dir):
        data = scio.loadmat(data_dir)
        train_data = np.dstack((data['color1'][:, :, :168], data['color2'][:, :, :168],
                                data['color3'][:, :, :168], data['color4'][:, :, :168]))
        val_data = np.dstack((data['color1'][:, :, 168:], data['color2'][:, :, 168:],
                              data['color3'][:, :, 168:], data['color4'][:, :, 168:]))
        train_label = np.hstack((np.ones(168) * 0, np.ones(168) * 1,
                                 np.ones(168) * 2, np.ones(168) * 3))
        val_label = np.hstack((np.ones(72) * 0, np.ones(72) * 1,
                               np.ones(72) * 2, np.ones(72) * 3))
        return train_data, val_data, train_label, val_label

    def __getitem__(self, index):
        # train
        if self.training:
            data = self.train_data
            label = self.train_label
        # val
        else:
            data = self.val_data
            label = self.val_label
        image = data[:, :, index]
        label = label[index]
        image = np.array(image/255, dtype='double').reshape((1, 6, 250))
        label = np.array(label, dtype='double')
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        # image = transforms.ToPILImage()(image).convert('RGB')
        # image = transforms.ToTensor()(image)

        return image, label

    def __len__(self):
        if not self.training:
            return 288
        return 672


def train(args, epoch, train_data, device, model, criterion, optimizer, scheduler):
    model.train()
    train_loss = 0.0
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    for step, (inputs, targets) in enumerate(train_data):
        inputs, targets = inputs.to(device), targets.to(device)
        # print(np.array(inputs).shape)
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs, targets.long())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        prec1 = accuracy(outputs, targets, topk=(1,))
        n = inputs.size(0)
        top1.update(prec1[0], n)
        # top5.update(prec5.item(), n)
        optimizer.step()
        train_loss += loss.item()
        dt = datetime.now()
        sys.stdout.write("\r{0} {1}  epoch:{2}/{3}, batch:{4}/{5}, lr:{6}, loss:{7}, top1:{8}, top5:{9}".format(
            dt.strftime('%x'), dt.strftime('%X'), '%.4d' % (epoch + 1), '%.4d' % args.epochs,
                                                  '%.4d' % (step + 1), '%.4d' % len(train_data),
                                                  '%.5f' % scheduler.get_lr()[0],
                                                  '%.6f' % loss, '%.3f' % top1.avg, '%.3f' % 1.0))
        sys.stdout.flush()
    train_writer.add_scalar('Loss', train_loss / (step + 1), epoch)
    train_writer.add_scalar('Acc', top1.avg, epoch)

    return train_loss / (step + 1), top1.avg


def validate(epoch, val_data, device, model):
    model.eval()
    val_loss = 0.0
    val_top1 = AvgrageMeter()
    val_top5 = AvgrageMeter()
    criterion = nn.CrossEntropyLoss().to(device)

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_data):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.float())
            loss = criterion(outputs, targets.long())
            val_loss += loss.item()
            prec1 = accuracy(outputs, targets, topk=(1,))
            n = inputs.size(0)
            val_top1.update(prec1[0], n)
            # val_top5.update(prec5.item(), n)
    val_writer.add_scalar('Loss', val_loss / (step + 1), epoch)
    val_writer.add_scalar('Acc', val_top1.avg, epoch)

    return val_top1.avg, 1.0, val_loss / (step + 1)


class Network(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Network, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = torch.reshape(x, (-1, 1500))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


def main():
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        device = torch.device("cuda")

    criterion = nn.CrossEntropyLoss().to(device)

    # Network
    model = Network(6*250, 300, 100, 4)

    # pretrained
    if args.pretrained:
        pretrained_dict = torch.load(args.pretrained, map_location=device)
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model = model.to(device)
    # summary(model, (3, 224, 224))
    # flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).to(device),), verbose=False)
    # print('Model: FLOPs={}M, Params={}M'.format(flops / 1e6, params / 1e6))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min, last_epoch=-1)

    train_transform, valid_transform = data_transforms(args)
    trainset = BrainWave_Set(args, training=True)
    valset = BrainWave_Set(args, training=False)
    train_queue = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=8, pin_memory=True)
    valid_queue = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=8, pin_memory=True)
    print('Dataset: Train={}, Val={}'.format(len(trainset), len(valset)))

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
    if not os.path.exists('./snapshots'):
        os.mkdir('./snapshots')
    for epoch in range(start_epoch, args.epochs):
        t1 = time.time()

        # train
        train(args, epoch, train_queue, device, model, criterion=criterion, optimizer=optimizer, scheduler=scheduler)
        scheduler.step()

        # validate
        val_top1, val_top5, val_obj = validate(epoch, val_data=valid_queue, device=device, model=model)
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
        print('\nval: loss={:.6}, top1={:.6}, top5={:.6}, best={:.6}, elapse={:.0f}s, eta={:.0f}h {:.0f}m {:.0f}s\n'
              .format(val_obj, val_top1, val_top5, best_acc, elapse, h, m, s))

    print('Best Top1 Acc: {:.6}'.format(best_acc))


if __name__ == '__main__':
    start = time.time()
    main()
    train_writer.close()
    val_writer.close()
    time_record(start)
