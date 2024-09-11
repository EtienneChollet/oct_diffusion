import os
import re
import shutil
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import cornucopia as cc


def log_hist(model: torch.nn.Module,
             epoch: int,
             writer: SummaryWriter
             ) -> None:
    """
    Log histograms of model parameters and gradients for TensorBoard.

    Parameters
    ----------
    model : Module
        Model whose parameters are to be logged.
    epoch : int
        Current epoch.
    writer : SummaryWriter
        TensorBoard writer object.
    """
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)
        if param.grad is not None:
            writer.add_histogram(f'{name}.grad', param.grad, epoch)


def log_metrics(writer: SummaryWriter,
                phase: str,
                metrics: dict,
                step: int
                ) -> None:
    """
    Logs training or validation metrics to TensorBoard.

    Parameters
    ----------
    writer : SummaryWriter
        TensorBoard writer object.
    phase : str
        Phase of training (e.g., 'training', 'validation', 'epoch').
    metrics : dict
        Dictionary of metrics to log.
    step : int
        Step (iteration or epoch) at which metrics are logged.
    """
    for key, value in metrics.items():
        writer.add_scalar(f'{phase}_{key}', value, step)


def log_model_graph(model_dir: str, model: torch.nn.Module,
                    train_loader: DataLoader) -> SummaryWriter:
    """
    Logs the model graph to TensorBoard.

    Parameters
    ----------
    model_dir : str
        Directory where TensorBoard logs will be saved.
    model : torch.nn.Module
        Model to be logged.
    train_loader : DataLoader
        DataLoader for sample input.

    Returns
    -------
    SummaryWriter
        TensorBoard SummaryWriter object.
    """
    # Initialize writer
    writer = SummaryWriter(model_dir)
    sample_inputs, _ = next(iter(train_loader))
    sample_inputs = sample_inputs[0]
    writer.add_graph(model, sample_inputs.to(next(model.parameters()).device))
    return writer


def save_checkpoint(state, filename: str = "checkpoint.pth.tar"):
    """Save checkpoint to disk."""
    torch.save(state, filename)


def check_and_recreate_directory(directory_path: str, verbose: bool = False):
    """
    Checks if a directory exists. If yes, deletes it and its contents.
    If not, create the directory.

    Parameters
    ----------
    directory_path : str
        Path of the directory to check.
    """
    if os.path.exists(directory_path):
        # If the directory exists, delete it and all its contents
        shutil.rmtree(directory_path)
        if verbose:
            print(f"Directory '{directory_path}' existed and was deleted.")

    # Create the directory (whether it was deleted or didn't exist)
    os.makedirs(directory_path)
    if verbose:
        print(f"Directory '{directory_path}' is created.")


def get_latest_checkpoint(directory_path: str,
                          grep_pattern: str = r"checkpoint_epoch_(\d+)") -> str:
    """
    Finds most recent file matching grep_pattern within specified directory.

    Parameters
    ----------
    directory_path : str
        The path to the directory where checkpoint files are stored.
    grep_pattern : str
        Grep pattern defining files to consider.

    Returns
    -------
    checkpoint_path : str
        Most recent checkpoint path.
    """

    # Regular expression to match files in the form *checkpoint_step_n_*
    pattern = re.compile(grep_pattern)

    max_step = -1
    latest_file = None

    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        match = pattern.search(filename)

        if match:
            # Extract the step number `n`
            step = int(match.group(1))

            # Track the file with the highest step number
            if step > max_step:
                max_step = step
                latest_file = filename

    if latest_file:
        return os.path.join(directory_path, latest_file)
    else:
        return None


def reassign_intensities(x0):
    x0 = x0.float()
    unique_labels = torch.unique(x0)
    for i in unique_labels:
        x0.masked_fill_(x0 == i, cc.Uniform(-1, 1)())
    return x0
