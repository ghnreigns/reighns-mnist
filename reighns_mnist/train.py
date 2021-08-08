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
from reighns_mnist import config, mnist
from argparse import Namespace


class Trainer:
    """Object used to facilitate training."""

    def __init__(
        self,
        params: Namespace,
        model,
        device=torch.device("cpu"),
        loss_fn=None,
        optimizer=None,
        scheduler=None,
        trial=None,
        writer=None
    ):
        # Set params
        self.params = params
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.trial = trial
        self.writer = writer

    def train(self, train_loader):
        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if self.device:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self.model(data)
            # loss = F.nll_loss(output, target)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.params.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.params.epochs, batch_idx *
                    len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data.item()))
                step = self.params.epochs * len(train_loader) + batch_idx
                self.writer.add_scalar(tag='train_loss',
                                       scalar_value=loss.data.item(), global_step=step)
                self.log_weights(step)

                # Logging
                config.logger.info(
                    f"Epoch: {self.params.epochs} | "
                    f"train_loss: {loss.data.item():.5f}, "
                    # f"val_loss: {val_loss:.5f}, "
                    f"lr: {self.optimizer.param_groups[0]['lr']:.2E}, "
                    # f"_patience: {_patience}"
                )

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                if self.device:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                output = self.model(data)
                # sum up batch loss
                test_loss += self.loss_fn(output,
                                          target).data.item()
                # test_loss += F.nll_loss(output, target,
                #                         reduction='sum').data.item()
                # get the index of the max log-probability
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).cpu().sum().item()

        test_loss /= len(test_loader.dataset)
        test_accuracy = 100.0 * correct / len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), test_accuracy))
        # TODO: Correct the below code into a new method.
        # step = (epoch + 1) * len(self.train_loader)
        # self.log_scalar('test_loss', test_loss, step)
        # self.log_scalar('test_accuracy', test_accuracy, step)

    def log_scalar(self, name, value, step):
        """Log a scalar value to both MLflow and TensorBoard"""
        self.writer.add_scalar(tag=name, scalar_value=value, global_step=step)
        mlflow.log_metric(name, value, step=step)

    def log_weights(self, step):
        self.writer.add_histogram(tag='conv1_weight',
                                  values=self.model.conv1.weight.data, global_step=step)
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
