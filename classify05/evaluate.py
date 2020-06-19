from utils import *
import torch.nn as nn
from classify_lung import args
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import pretrainedmodels
from model import MobileNetV2
from torchvision import models


if __name__ == '__main__':
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        device = torch.device("cuda")

    train_transform, valid_transform = data_transforms(args)
    valset = dset.ImageFolder(root=os.path.join(args.data_dir, 'val'), transform=valid_transform)
    valid_queue = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                              shuffle=False, pin_memory=True)

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

    # load weights
    resume_path = './snapshots/{}_train_states.pt.tar'.format(args.exp_name)
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint['supernet_state'])

    model = model.to(device)

    model.eval()
    val_loss = 0.0
    val_top1 = AvgrageMeter()
    criterion = nn.CrossEntropyLoss().to(device)

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(valid_queue):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            prec1 = accuracy(outputs, targets, topk=(1, ))
            n = inputs.size(0)
            val_top1.update(prec1[0], n)
    print('Validation: Top1_Acc={}'.format(val_top1.avg))
