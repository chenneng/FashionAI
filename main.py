import os
import sys
import os.path as osp
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from serialization import Logger
from model import creat_model
from dataset import fashionData
from train import train, evaluate
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='fashion classification')
parser.add_argument('--data_root', type = str, default = '../datasets/fashionAI_attributes_train1', help = 'data root path')
parser.add_argument('--batch', type = int, default = 128, help = 'batch size for training (default: 128)')
parser.add_argument('--test_batch', type = int, default = 64, help = 'batch size for testing (default: 64)')
parser.add_argument('--attr_epochs', type = int, default = 40, help = 'number of each attributes epochs')
parser.add_argument('--lr', type = float, default = 0.001, help = 'learning rate (default: 0.001)')
parser.add_argument('--adjust_lr', type = bool, default = True, help = 'adjust learning rate')
parser.add_argument('--momentum', type = float, default = 0.9, help = 'SGD momentum (default: 0.9)')
parser.add_argument('--gpu', type = bool, default = True, help = 'GPU training')
parser.add_argument('--shuffle', type = bool, default = True, help = 'data shuffle')
parser.add_argument('--resume', type = str, default = None, help = 'resume model path')
parser.add_argument('--height', type = int, default = 224, help = 'height')
parser.add_argument('--width', type = int, default = 224, help = 'width')
parser.add_argument('--evaluate_interval', type = int, default = 1, help = 'epochs before evaluate model')
parser.add_argument('--save_interval', type = int, default = 20, help = 'epochs before save model')
parser.add_argument('--save_dir', type = str, default = './models', help = 'log and model save dir')
parser.add_argument('--test_only', type = bool, default = False, help = 'only evaluate the model')
args = parser.parse_args()

attr = None
data_transform = transforms.Compose(transforms = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((args.height, args.width), interpolation = 3),
    transforms.ColorJitter(brightness = 0.5, contrast = 0.5, hue = 0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.647, 0.609, 0.596], std=[0.089, 0.093, 0.094])
])
kwargs = {'num_workers': 6, 'pin_memory': True} if args.gpu else {}

model = creat_model()
if args.resume:
    model.load_state_dict(torch.load(args.resume))

if args.gpu:
    model = nn.DataParallel(model).cuda()

optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum)

def adjust_lr(optimizer, attr_epoch):
    if attr_epoch < 30:
        lr = 0.001
    elif attr_epoch < 40:
        lr = 0.0002
    else:
        lr = 0.0001
    for para in optimizer.param_groups:
        para['lr'] = lr

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))

for i in range(8):
    if i == 0:
        attr = 'coat_length_labels'
    elif i == 1:
        attr = 'pant_length_labels'
    elif i == 2:
        attr = 'skirt_length_labels'
    elif i == 3:
        attr = 'sleeve_length_labels'
    elif i == 4:
        attr = 'collar_design_labels'
    elif i == 5:
        attr = 'lapel_design_labels'
    elif i == 6:
        attr = 'neck_design_labels'
    elif i == 7:
        attr = 'neckline_design_labels'

    trainset = fashionData(args.data_root, attr, split = 0.8, data_type = 'train', transform = data_transform)
    testset = fashionData(args.data_root, attr, split = 0.8, data_type = 'test', transform = data_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size = args.batch, shuffle = True, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size = args.test_batch, shuffle = True, **kwargs)

    if args.test_only and args.resume is not None:
        evaluate(i, model, test_loader)

    if args.test_only is False:
        start_epoch = 0
	for epoch in range(start_epoch, args.attr_epochs):
	    if args.adjust_lr:
		adjust_lr(optimizer, epoch)

            train(i, epoch, model, optimizer, train_loader)

	    if epoch % args.evaluate_interval == 0 or epoch == args.attr_epochs - 1:
		evaluate(i, model, test_loader)

	    if epoch % args.save_interval == 0 or epoch == args.attr_epochs - 1:
		torch.save(model.state_dict(),  osp.join(args.save_dir, ('model_{}.pth'.format(i * args.attr_epochs + epoch))))

