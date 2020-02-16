from utils import *
from PIL import Image
import torch.nn as nn
from classify01 import args
from torchvision import models
import torch.backends.cudnn as cudnn


if __name__ == '__main__':
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        device = torch.device("cuda")

    # transform
    predict_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # load img
    img = Image.open('/Users/lushun/Documents/dataset/wavelet/val/4/4-2-36.jpg')
    input = predict_transform(img)
    input = input.unsqueeze(0)

    # load weights
    model = models.resnet50(pretrained=True)
    fc_inputs = model.fc.in_features
    model.fc = nn.Linear(fc_inputs, args.classes)
    resume_path = './snapshots/{}_train_states.pt.tar'.format(args.exp_name)
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint['supernet_state'])

    model = model.to(device)
    model.eval()

    # predict
    predict = model(input)
    print('predict:{}'.format(predict))
    print('Class:{}'.format(torch.argmax(predict)))
