# Configurations

import logging
import warnings
import sys
from pathlib import Path

import mlflow
import pretty_errors  # NOQA: F401 (imported but unused)
from rich.logging import RichHandler
import torch

# Repository's Names
AUTHOR = "Hongnan G."
REPO = "reighns_mnist"

# device global config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directories:
# This will create all the folders needed. Which begs the question on when should I execute this script?
BASE_DIR = Path(__file__).parent.parent.absolute()  # C:\Users\reigHns\mnist

CONFIG_DIR = Path(BASE_DIR, "config")
LOGS_DIR = Path(BASE_DIR, "logs")
DATA_DIR = Path(BASE_DIR, "data")
MODEL_DIR = Path(BASE_DIR, "model")
STORES_DIR = Path(BASE_DIR, "stores")


# Local stores
BLOB_STORE = Path(STORES_DIR, "blob")
FEATURE_STORE = Path(STORES_DIR, "feature")
MODEL_REGISTRY = Path(STORES_DIR, "model")
TENSORBOARD = Path(STORES_DIR, "tensorboard")

# Create dirs
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
STORES_DIR.mkdir(parents=True, exist_ok=True)
BLOB_STORE.mkdir(parents=True, exist_ok=True)
FEATURE_STORE.mkdir(parents=True, exist_ok=True)
MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)
# new folder
TENSORBOARD.mkdir(parents=True, exist_ok=True)


# MLFlow model registry: Note the filepath is file:////C:\Users\reigHns\mnist\stores\model

#mlflow.set_tracking_uri(uri="file://" + str(MODEL_REGISTRY.absolute()))
# workaround for windows at the moment
# TODO : Switch to Linux.
mlflow.set_tracking_uri(uri="file://" + "C:/Users/reigHns/mnist/stores/model")

# suppress User Warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Logger
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "loggers": {
        "root": {
            "handlers": ["console", "info", "error"],
            "level": logging.INFO,
            "propagate": True,
        },
    },
}
# call this function and set a global variable to set the logger to the configuration dictionary that you have.
logging.config.dictConfig(logging_config)
# logger -> creates a logger class which corresponds to the settings in the logging_config[loggers[root]]
logger = logging.getLogger("root")
# set markup to true for console
logger.handlers[0] = RichHandler(markup=True)

logger.warning(
    msg="CE Loss combines LogSoftmax and NLLLoss in one single class. If your head output layer has LogSoftmax, then use NLLLoss for similar effects of CE loss, but note on the numerical instability that it may cause.")
