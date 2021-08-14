from __future__ import print_function
import enum

import mlflow.pytorch
import mlflow
# Trains using PyTorch and logs training metrics and weights in TensorFlow event format to the MLflow run's artifact directory.
# This stores the TensorFlow events in MLflow for later access using TensorBoard.
#
# Code based on https://github.com/mlflow/mlflow/blob/master/example/tutorial/pytorch_tensorboard.py.
#

import pytz
import os
import numpy as np
import tempfile
import datetime
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple

from chardet.universaldetector import UniversalDetector
from reighns_mnist import config, mnist, callbacks, models
from argparse import Namespace
from typing import *


class Trainer:
    """Object used to facilitate training."""

    def __init__(
        self,
        params: Namespace,
        model: models.Model,
        device=torch.device("cpu"),
        train_loss_fn=None,
        val_loss_fn=None,
        optimizer=None,
        scheduler=None,
        trial=None,
        writer=None,
        early_stopping: callbacks.EarlyStopping = None
    ):
        # Set params
        self.params = params
        self.model = model
        self.device = device
        # Note that we can evaluate train and validation fold with different loss functions.
        self.train_loss_fn = train_loss_fn
        self.val_loss_fn = val_loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.trial = trial
        self.writer = writer
        self.early_stopping = early_stopping

        # list to contain various train metrics
        self.train_loss_history: List = []
        self.train_metric_history: List = []
        self.val_loss_history: List = []
        self.val_acc_history: List = []

    def get_lr(self, optimizer) -> float:
        """Get the learning rate of the current epoch.

        Note learning rate can be different for different layers, hence the for loop.

        Args:
            self.optimizer (torch.optim): [description]

        Returns:
            float: [description]
        """
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def run_train(self, train_loader: torch.utils.data.DataLoader) -> Dict:
        """[summary]

        Args:
            train_loader (torch.utils.data.DataLoader): [description]

        Returns:
            Dict: [description]
        """

        # train start time
        train_start_time = time.time()

        # get avg train loss for this epoch
        avg_train_loss = self.train_one_epoch(train_loader)
        self.train_loss_history.append(avg_train_loss)

        # train end time
        train_end_time = time.time()

        # total time elapsed for this epoch

        train_elapsed_time = time.strftime(
            "%H:%M:%S", time.gmtime(train_end_time - train_start_time)
        )

        return {"train_loss": self.train_loss_history, "time_elapsed": train_elapsed_time}

    def run_val(self, val_loader: torch.utils.data.DataLoader) -> Dict:
        """[summary]

        Args:
            val_loader (torch.utils.data.DataLoader): [description]

        Returns:
            Dict: [description]
        """

        # train start time
        val_start_time = time.time()

        # get avg train loss for this epoch
        self.avg_val_loss, self.val_acc = self.valid_one_epoch(val_loader)
        self.val_loss_history.append(self.avg_val_loss)
        self.val_acc_history.append(self.val_acc)

        # train end time
        val_end_time = time.time()

        # total time elapsed for this epoch

        val_elapsed_time = time.strftime(
            "%H:%M:%S", time.gmtime(val_start_time - val_end_time)
        )

        return {"val_loss": self.val_loss_history, "val_acc": self.val_acc_history, "time_elapsed": val_elapsed_time}

    def fit(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, fold: int = None):
        if fold is None:
            fold = "No Fold"

        config.logger.info(
            f"Training on Fold {fold} and using {self.params.model['model_name']}")

        for _epoch in range(1, self.params.epochs+1):
            # get current epoch's learning rate
            curr_lr = self.get_lr(self.optimizer)
            # get current time
            timestamp = datetime.datetime.now(pytz.timezone(
                "Asia/Singapore")).strftime("%Y-%m-%d %H-%M-%S")
            # print current time and lr
            config.logger.info("\nTime: {}\nLR: {}".format(timestamp, curr_lr))

            train_dict: Dict = self.run_train(train_loader)
            print(train_dict)

            # Note that train_dict['train_loss'] returns a list of loss [0.3, 0.2, 0.1 etc] and since _epoch starts from 1, we therefore
            # index this list by _epoch - 1 to get the current epoch loss.

            config.logger.info(f"[RESULT]: Train. Epoch {_epoch} | Avg Train Summary Loss: {train_dict['train_loss'][_epoch-1]:.3f} | "
                               f"Time Elapsed: {train_dict['time_elapsed']}")

            val_dict: Dict = self.run_val(val_loader)

            # Note that train_dict['train_loss'] returns a list of loss [0.3, 0.2, 0.1 etc] and since _epoch starts from 1, we therefore
            # index this list by _epoch - 1 to get the current epoch loss.

            config.logger.info(f"[RESULT]: Validation. Epoch {_epoch} | Avg Val Summary Loss: {val_dict['val_loss'][_epoch-1]:.3f} | "
                               f"Val Acc: {val_dict['val_acc'][_epoch-1]:.3f} | "
                               f"Time Elapsed: {val_dict['time_elapsed']}")

            # Early Stopping code block
            if self.early_stopping is not None:

                best_score, early_stop = self.early_stopping.should_stop(
                    curr_epoch_score=val_dict['val_loss'][_epoch-1]
                )
                self.best_loss = best_score
                # TODO: SAVE MODEL
                # self.save(
                #     "{self.param.model['model_name']}_best_loss_fold_{fold}.pt")
                if early_stop:
                    config.logger.info("Stopping Early!")
                    break

            # Scheduler Step code block
            # Note the special case for ReduceLROnplateau.
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_dict['val_acc'])
                else:
                    self.scheduler.step()

    def train_one_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        """Train one epoch of the model."""
        # set to train mode
        self.model.train()
        cumulative_train_loss: float = 0.0

        # Iterate over train batches
        for step, data in enumerate(train_loader, start=1):
            # convert everythinbg inside data to device
            data = [item.to(self.device) for item in data]

            # unpack
            inputs, targets = data[0], data[1]

            self.optimizer.zero_grad()  # reset gradients

            logits = self.model(inputs)  # Forward pass logits

            curr_batch_train_loss = self.train_loss_fn(logits, targets)

            curr_batch_train_loss.backward()  # Backward pass

            self.optimizer.step()  # Update weights using the optimizer

            # Cumulative Loss
            # Batch/Step 1: curr_batch_train_loss = 10 -> cumulative_train_loss = (10-0)/1 = 10
            # Batch/Step 2: curr_batch_train_loss = 12 -> cumulative_train_loss = 10 + (12-10)/2 = 11
            # Essentially, cumulative train loss = loss over all batches / batches
            cumulative_train_loss += (curr_batch_train_loss.detach().item() -
                                      cumulative_train_loss) / (step)

            self.writer.add_scalar(tag='train_loss',
                                   scalar_value=curr_batch_train_loss.data.item(), global_step=step)
            self.log_weights(step)

        return cumulative_train_loss

    # @torch.no_grad
    def valid_one_epoch(self, val_loader):
        """Validate one training epoch."""
        # set to eval mode
        self.model.eval()

        cumulative_val_loss: float = 0.0

        VAL_LOGITS, Y_TRUES, Y_PROBS = [], [], []

        with torch.no_grad():
            for step, data in enumerate(val_loader, start=1):
                # convert everythinbg inside data to device
                data = [item.to(self.device) for item in data]
                # unpack
                inputs, y_true = data[0], data[1]

                self.optimizer.zero_grad()  # reset gradients

                logits = self.model(inputs)  # Forward pass logits

                curr_batch_val_loss = self.val_loss_fn(logits, y_true)
                cumulative_val_loss += (curr_batch_val_loss.item() -
                                        cumulative_val_loss) / (step)

                # Store outputs that are needed to compute various metrics
                softmax_preds = torch.nn.Softmax(dim=1)(
                    input=logits).to("cpu").numpy()
                # for OOF score and other computation
                Y_PROBS.extend(softmax_preds)
                Y_TRUES.extend(y_true.cpu().numpy())
                VAL_LOGITS.extend(logits.cpu().numpy())

            argmax = np.argmax(Y_PROBS, axis=1)
            correct = np.equal(argmax, np.asarray(Y_TRUES))
            total = correct.shape[0]
            # argmax = [1, 2, 1] Y_TRUES = [1, 1, 2] -> correct = [True, False, False] -> num_correct = 1 and total = 3 -> acc = 1/3
            num_correct = np.sum(correct)
            accuracy = (num_correct/total) * 100

        return cumulative_val_loss, accuracy

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
