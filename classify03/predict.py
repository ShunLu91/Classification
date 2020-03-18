from utils import *
from PIL import Image
import torch.nn as nn
from classify_lung import args
from torchvision import models
import torch.backends.cudnn as cudnn
import pretrainedmodels
from model import MobileNetV2


if __name__ == '__main__':
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        device = torch.device("cuda")

    train_transform, valid_transform = data_transforms(args)

    # load img
    img = Image.open('person1946_bacteria_4874.jpeg').convert("RGB")
    input = valid_transform(img)
    input = input.unsqueeze(0)


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

    # predict
    predict = model(input.to(device))
    print('predict:{}'.format(predict))
    print('Class:{}'.format(torch.argmax(predict)))
