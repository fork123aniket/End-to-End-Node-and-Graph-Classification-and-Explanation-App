import os
import numpy as np
import random
import shutil
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from urllib.parse import urlparse

import torch
import mlflow

from src.constants.config import *
from src.log.logger import logger


def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        verbose (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


def read_yaml() -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(ParamConfig.path_to_yaml.value) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"params.yaml file loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("params.yaml file is empty")
    except Exception as e:
        raise e


def set_seeds(seed: int = 42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    eval("setattr(torch.backends.cudnn, 'deterministic', True)")
    eval("setattr(torch.backends.cudnn, 'benchmark', False)")
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"


def search_best_run_and_save_model(exp_name: str, task: str = 'node'):
    sorted_runs = mlflow.search_runs(
        experiment_names=[exp_name], order_by=["metrics.test_accuracy DESC"],
        search_all_experiments=True
    )

    logger.info(f'sorted_runs: {sorted_runs}')

    artifact_dir = urlparse(mlflow.get_run(str(sorted_runs['run_id'][0])).info.artifact_uri).path
    path = Path(artifact_dir[1:], 'model')

    # os.mkdir(TrainModel.model_save_dir.value)

    if task == 'node':
        shutil.copytree(path, TrainModel.model_save_dir.value)
    else:
        shutil.copytree(path, TrainModel.graph_model_save_dir.value)
    logger.info(">>>>>> Best Model Saved Successfully <<<<<<")
