from __future__ import print_function
from reighns_mnist import models_test, models, seed, utils, train

from argparse import Namespace
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
from typing import *
from pathlib import Path
from chardet.universaldetector import UniversalDetector
from reighns_mnist import config, mnist
import typer

# Typer CLI app
app = typer.Typer()


@app.command()
def download_data():
    """Load data from URL and save to local drive."""
    # Download data, pre-caching.
    datasets.MNIST(root=config.DATA_DIR.absolute(), train=True, download=True)
    datasets.MNIST(root=config.DATA_DIR.absolute(), train=False, download=True)
    # Save data

    config.logger.info("Data downloaded!")


@app.command()
def train_model(params_fp: Path = Path(config.CONFIG_DIR, "params.json"),
                model_dir: Optional[Path] = Path(config.MODEL_DIR),
                experiment_name: Optional[str] = "baselines",
                run_name: Optional[str] = "model",
                ) -> None:
    """[summary]

    Args:
        params_fp (Path, optional): [description]. Defaults to Path(config.CONFIG_DIR, "params.json").
        model_dir (Optional[Path], optional): [description]. Defaults to Path(config.MODEL_DIR).
        experiment_name (Optional[str], optional): [description]. Defaults to "baselines".
        run_name (Optional[str], optional): [description]. Defaults to "model".
    """

    # Parameters
    params = Namespace(**utils.load_dict(filepath=params_fp))

    mlflow.set_experiment(experiment_name=experiment_name)

    with mlflow.start_run() as run:
        run_id: str = run.info.run_uuid

        # Log our parameters into mlflow
        # vars turns params into a dict
        for key, value in vars(params).items():
            mlflow.log_param(key, value)

        # this ensures that each run is uniquely stored
        output_dir = Path(config.TENSORBOARD, run_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Writing TensorFlow events locally to %s\n" %
              output_dir.absolute())

        writer = SummaryWriter(
            log_dir=output_dir.absolute(), comment='MNIST_LOG')

        # create loader
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=config.DATA_DIR.absolute(), train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=params.train_bs, shuffle=True, **params.dataloader)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=config.DATA_DIR.absolute(), train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=params.test_bs, shuffle=True, **params.dataloader)

        # construct Trainer class
        model = models.Model().to(device=config.DEVICE)
        loss_fn: torch.nn = torch.nn.CrossEntropyLoss()
        optimizer: torch.optim = torch.optim.SGD(
            model.parameters(), lr=params.base_lr, momentum=params.momentum)
        scheduler: torch.nn = None
        trial = None
        trainer = train.Trainer(params=params, model=model, device=config.DEVICE,
                                loss_fn=loss_fn, optimizer=optimizer,
                                scheduler=scheduler, trial=trial, writer=writer)

        for epoch in range(1, params.epochs + 1):
            # print out active_run
            print("Active Run ID: %s, Epoch: %s \n" %
                  (run_id, epoch))

            trainer.train(train_loader)
            trainer.test(test_loader)

        print("Uploading TensorFlow events as a run artifact.")
        mlflow.log_artifacts(output_dir, artifact_path="events")
