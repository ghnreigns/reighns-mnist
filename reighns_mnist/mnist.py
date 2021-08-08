from __future__ import print_function

import mlflow.pytorch
import mlflow
# Trains using PyTorch and logs training metrics and weights in TensorFlow event format to the MLflow run's artifact directory.
# This stores the TensorFlow events in MLflow for later access using TensorBoard.
#
# Code based on https://github.com/mlflow/mlflow/blob/master/example/tutorial/pytorch_tensorboard.py.
#


import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple

from chardet.universaldetector import UniversalDetector

# from tensorboardX import SummaryWriter
# import tensorflow as tf
# import tensorflow.summary
# from tensorflow.summary import scalar
# from tensorflow.summary import histogram
# Create Params dictionary

"""
    1. mlflow ui -> go to ui
    2. Use tensorboard to visualize histograms etc
    3. Use MLFLOW to visualize diagrams too - connect 2 and 3
    """


class Params(object):
    def __init__(self, batch_size, test_batch_size, epochs, lr, momentum, seed, cuda, log_interval):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.seed = seed
        self.cuda = cuda
        self.log_interval = log_interval


# Configure args
args = Params(64, 1000, 10, 0.01, 0.5, 1, True, 200)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

kwargs = {'num_workers': 1, 'pin_memory': True} if device else {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='/data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='/data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=0)

    def log_weights(self, step):
        writer.add_histogram(tag='conv1_weight',
                             values=model.conv1.weight.data, global_step=step)
        # writer.add_summary(writer.add_histogram('weights/conv1/bias',
        #                                         model.conv1.bias.data).eval(), step)
        # writer.add_summary(writer.add_histogram('weights/conv2/weight',
        #                                         model.conv2.weight.data).eval(), step)
        # writer.add_summary(writer.add_histogram('weights/conv2/bias',
        #                                                  model.conv2.bias.data).eval(), step)
        # writer.add_summary(writer.add_histogram('weights/fc1/weight',
        #                                         model.fc1.weight.data).eval(), step)
        # writer.add_summary(writer.add_histogram('weights/fc1/bias',
        #                                         model.fc1.bias.data).eval(), step)
        # writer.add_summary(writer.add_histogram('weights/fc2/weight',
        #                                         model.fc2.weight.data).eval(), step)
        # writer.add_summary(writer.add_histogram('weights/fc2/bias',
        #                                         model.fc2.bias.data).eval(), step)


model = Model()
if device:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

writer = None  # Will be used to write TensorBoard events


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if device:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar(tag='train_loss',
                              scalar_value=loss.data.item(), global_step=step)
            model.log_weights(step)


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if device:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target,
                                    reduction='sum').data.item()
            # get the index of the max log-probability
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100.0 * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_accuracy))
    step = (epoch + 1) * len(train_loader)
    log_scalar('test_loss', test_loss, step)
    log_scalar('test_accuracy', test_accuracy, step)


def log_scalar(name, value, step):
    """Log a scalar value to both MLflow and TensorBoard"""
    writer.add_scalar(tag=name, scalar_value=value, global_step=step)
    mlflow.log_metric(name, value, step=step)
