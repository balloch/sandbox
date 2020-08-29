import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision 
from torchvision import transforms

import time
import argparse
import os

from PIL import Image


def train(model, criterion, trainloader, testloader, args):
    start_epoch = 0
    global best_acc
    best_acc = 0
    if args.resume_checkpoint:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
            momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.045, 
            epochs=args.epochs, steps_per_epoch=len(trainloader), 
            anneal_strategy='linear', div_factor=10.0, final_div_factor=200.0)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.98)  # paper numbers init_LR=0.045
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.sched_step)
    
    #print(torch.nn.functional.softmax(output[0], dim=0))
    for epoch in range(start_epoch, args.epochs-start_epoch):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
#            print('lr: ', scheduler.get_last_lr())
            scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Train:\t', 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(len(trainloader)+1), 100.*correct/total, correct, total))
            #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        evaluate(model, criterion, testloader, epoch)
        #scheduler.step()
    return best_acc

def evaluate(model, criterion, testloader, epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Val:\t', 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(len(testloader)+1), 100.*correct/total, correct, total))
            #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--epochs', default=100, type=int, help='total training epochs')
parser.add_argument('--sched_step', default=25, type=int, help='when scheduler reduces lr')
parser.add_argument('--batch_size',default=240, type=int, help='test and train batch_size')
parser.add_argument('--val_batch', default=0, type=int, help='val batch size if greater than 0')
parser.add_argument('--resume_checkpoint', default=None, type=str, help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=False).to(device)
criterion = nn.CrossEntropyLoss()

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
if args.val_batch > 0:
    testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.val_batch, shuffle=False, num_workers=2)
else:
    testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Multi GPU
#if device == 'cuda':
#    model = torch.nn.DataParallel(model)
#    cudnn.benchmark = True

start=time.process_time()
best_acc = train(model, criterion, trainloader, testloader, args)
end = time.process_time()
print('Best Accuracy: ', best_acc)
print('Total Time: ', end-start)
