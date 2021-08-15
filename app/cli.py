from __future__ import print_function

from torch._C import device
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
from reighns_mnist import config, callbacks, plot, seed
import typer
import torchvision


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

    seed.seed_all(1992)  # seed

    # Unpack Parameters from config/params.json file.
    params = Namespace(**utils.load_dict(filepath=params_fp))

    mlflow.set_experiment(experiment_name=experiment_name)

    with mlflow.start_run() as run:
        # Each run/experiment will yield an unique run_id.
        # We conveniently use run_id to name our experiment for both mlflow and tensorboard
        run_id: str = run.info.run_uuid

        # this ensures that each run is uniquely stored. We store our artifacts to the directory created here.
        # T1 = Tensorboard Code 1
        output_dir = Path(config.TENSORBOARD, run_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        config.logger.info(
            f"Creating MLflow and Tensorboard with run id {run_id}.")

        # T2: Create Tensorboard writer and specify the log_dir where the artifacts from Tensorboard is going to be stored.

        writer = SummaryWriter(
            log_dir=output_dir.absolute(), comment='MNIST_LOG')

        # Log our parameters into mlflow and tensorboard - we loop through params, note that vars turns params into a dict.
        # Limitation of tensorboard here, somehow it cannot log nested dictionary.

        mlflow.log_params(vars(params))
        # T3
        writer.add_hparams(hparam_dict={}, metric_dict={}, run_name=run_id)

        # create loader
        train_dataset = datasets.MNIST(root=config.DATA_DIR.absolute(), train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ]))
        test_dataset = datasets.MNIST(root=config.DATA_DIR.absolute(), train=False,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              (0.1307,), (0.3081,))
                                      ]))
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=params.train_bs, shuffle=True, **params.dataloader)

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=params.test_bs, shuffle=True, **params.dataloader)

        # T4: get some random training images and write to Tensorboard
        dataiter = iter(train_loader)
        one_batch_images, _ = dataiter.next()

        # create grid of images
        img_grid = torchvision.utils.make_grid(one_batch_images)

        # show images
        plot.matplotlib_imshow(img_grid, one_channel=True)

        # write to tensorboard
        writer.add_image('four_fashion_mnist_images', img_grid)
        # end of T4, quite useful if you want visualize your images.

        # construct Trainer class
        model = models.Model().to(device=config.DEVICE)
        # T5: Visualize Graph of model: Remember to double click on the model in the graph to visualize granular layers.
        writer.add_graph(
            model=model, input_to_model=one_batch_images.to(device=config.DEVICE))
        writer.close()
        # End of T5

        # select random images and their target indices
        images, labels = plot.select_n_random(
            train_dataset.data, train_dataset.targets, n=1000)
        # Get mapping for classes
        classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        # get the class labels for each image
        class_labels = [classes[lab] for lab in labels]

        # log embeddings
        features = images.view(-1, 28 * 28)
        writer.add_embedding(features,
                             metadata=class_labels,
                             label_img=images.unsqueeze(1))

        config.logger.info(
            "You can click on T-SNE to see a nice 3D visualization of Embeddings.")
        writer.close()

        train_loss_fn: torch.nn.modules.loss = torch.nn.CrossEntropyLoss()
        val_loss_fn: torch.nn.modules.loss = torch.nn.CrossEntropyLoss()
        optimizer: torch.optim = torch.optim.SGD(
            model.parameters(), lr=params.base_lr, momentum=params.momentum)

        scheduler: torch.nn = None
        trial = None
        trainer: train.Trainer = train.Trainer(params=params, model=model, device=config.DEVICE,
                                               train_loss_fn=train_loss_fn, val_loss_fn=val_loss_fn, optimizer=optimizer,
                                               scheduler=scheduler, trial=trial, writer=writer,
                                               early_stopping=callbacks.EarlyStopping(patience=3,
                                                                                      mode=callbacks.Mode.MIN,
                                                                                      min_delta=1e-5))

        trainer.fit(train_loader, test_loader)

        print("Uploading TensorFlow events as a run artifact.")
        mlflow.log_artifacts(output_dir, artifact_path="events")
        print(f"Command Line: mlflow ui {run_id}")
        print(
            f"Run Tensorboard @ tensorboard --logdir ./stores/tensorboard/{run_id}")
