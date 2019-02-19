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
parser.add_argument('--epochs', type = int, default = 30, help = 'number of epochs')
parser.add_argument('--lr', type = float, default = 0.0001, help = 'learning rate (default: 0.001)')
parser.add_argument('--adjust_lr', type = bool, default = True, help = 'adjust learning rate')
parser.add_argument('--momentum', type = float, default = 0.9, help = 'SGD momentum (default: 0.9)')
parser.add_argument('--gpu', type = bool, default = True, help = 'GPU training')
parser.add_argument('--shuffle', type = bool, default = True, help = 'data shuffle')
parser.add_argument('--resume', type = str, default = None, help = 'resume model path')
parser.add_argument('--height', type = int, default = 224, help = 'height')
parser.add_argument('--width', type = int, default = 224, help = 'width')
parser.add_argument('--evaluate_interval', type = int, default = 1, help = 'epochs before evaluate model')
parser.add_argument('--save_interval', type = int, default = 5, help = 'epochs before save model')
parser.add_argument('--save_dir', type = str, default = './models', help = 'log and model save dir')
parser.add_argument('--test_only', type = bool, default = False, help = 'only evaluate the model')
args = parser.parse_args()

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
    #model.load_state_dict(torch.load(args.resume))
    model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.resume).items()})

if args.gpu:
    model = nn.DataParallel(model).cuda()

#optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum)
optimizer = optim.Adam(model.parameters(), lr = args.lr)

def adjust_lr(base_lr, optimizer, epoch):
    lr = base_lr * (0.1 ** (epoch // 10))
    
    for para in optimizer.param_groups:
        para['lr'] = lr

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))

print('let us begin:')

trainset = fashionData(args.data_root, split = 0.8, data_type = 'train', transform = data_transform)
testset = fashionData(args.data_root, split = 0.8, data_type = 'test', transform = data_transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size = args.batch, shuffle = True, **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size = args.test_batch, shuffle = True, **kwargs)

if args.test_only and args.resume is not None:
    evaluate(model, test_loader)

if args.test_only is False:
    start_epoch = 0
for epoch in range(start_epoch, args.epochs):
    if args.adjust_lr:
	    adjust_lr(args.lr, optimizer, epoch)

    train(epoch, model, optimizer, train_loader)

    if epoch % args.evaluate_interval == 0 or epoch == args.epochs - 1:
        evaluate(model, test_loader)

    if epoch % args.save_interval == 0 or epoch == args.epochs - 1:
        print('saving model..')
        torch.save(model.state_dict(),  osp.join(args.save_dir, ('model_{}.pth'.format(epoch))))

