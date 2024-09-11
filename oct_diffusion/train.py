import torch
import torch.multiprocessing.spawn
import torch.utils
from torch.utils.data import DataLoader, random_split
import torch.utils.data
from torch import nn

from oct_diffusion.models import ResUNet
from oct_diffusion.datasets import PatchDirectoryDataset
from oct_diffusion.noise import RandomMarkovChainSampler
from utils import (
    log_model_graph,
    log_hist,
    log_metrics,
    save_checkpoint,
    check_and_recreate_directory
    )


class Trainer:
    """
    Initializes the trainer object.

    Parameters
    ----------
    model : nn.Module
        The model to be trained.
    train_loader : DataLoader
        DataLoader for the training dataset.
    val_loader : DataLoader
        DataLoader for the validation dataset.
    criterion : nn.Module
        Loss function.
    optimizer : torch.optim
        Optimizer for updating model weights.
    device : str
        Device to run the training on ('cpu' or 'cuda').
    """
    def __init__(self,
                 model_dir: str,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: torch.optim,
                 device: str = 'cuda'
                 ):
        """
        Initializes the trainer object.

        Parameters
        ----------
        model : nn.Module
            The model to be trained.
        train_loader : DataLoader
            DataLoader for the training dataset.
        val_loader : DataLoader
            DataLoader for the validation dataset.
        criterion : nn.Module
            Loss function.
        optimizer : torch.optim
            Optimizer for updating model weights.
        device : str
            Device to run the training on ('cpu' or 'cuda').
        """
        self.model_dir = model_dir
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.device = device
        # Initializing states of tracking attributes
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = 1

    def train(self, num_epochs: int = 100):
        """
        Train the model over num_epochs epochs.

        Parameters
        ----------
        num_epochs : int
            Number of epochs to train the model.
        """
        # Refreshing output directories (making them if they don't exist)
        check_and_recreate_directory(self.model_dir)
        check_and_recreate_directory(f'{self.model_dir}/checkpoints')
        # Log the connectivity graph of the model by passing a tensor through
        # and tracking it
        self.writer = log_model_graph(
            self.model_dir,
            self.model,
            self.train_loader)
        # Iterate across epochs
        for epoch in range(num_epochs):
            train_loss = self._train_one_epoch()
            val_loss = self._validate()
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Validation Loss: {val_loss:.4f}")
            if self.current_epoch % 5 == 0:
                # Log parameter and gradient frequency distributions
                log_hist(self.model, self.current_epoch, self.writer)
            # Increment epoch
            self.current_epoch += 1

    def _train_one_epoch(self):
        """
        Perform a single epoch of training.

        Parameters
        ----------
        epoch : int
            Curent epoch.
        num_epochs : int
            Total number of epochs.

        Returns
        -------
        epoch_train_loss : float
            Average loss over epoch.
        """
        self.model.train()  # Set the model to training mode
        epoch_train_loss = 0.0
        for i, (x0, zt) in enumerate(self.train_loader):
            # Get data
            x0, zt = x0.to(self.device)[0], zt.to(self.device)[0]

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(zt)
            loss = self.criterion(x0, outputs)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            epoch_train_loss += loss.item()

            # Log training metrics every 10 steps
            if i % 10 == 0:
                # Log instantaneous metrics
                log_metrics(
                    writer=self.writer,
                    phase='training',
                    metrics={
                        'loss': loss.item(),
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                        },
                    step=self.current_step
                    )
            # Increment step
            self.current_step += 1

        epoch_train_loss /= len(self.train_loader)
        # Log average metrics over epoch
        log_metrics(self.writer,
                    'epoch',
                    {'loss': epoch_train_loss},
                    self.current_epoch)
        return epoch_train_loss

    def _validate(self):
        """
        Evaluate model on validation set.

        Returns
        -------
        val_loss : float
            The loss on the validation set.
        """
        # Set the model to evaluation mode
        self.model.eval()
        val_loss = 0.0

        # Make predictions without tracking gradients
        with torch.no_grad():
            for x0, labels in self.val_loader:
                x0, labels = (
                    x0.to(self.device)[0],
                    labels.to(self.device)[0]
                    )

                outputs = self.model(labels)
                loss = self.criterion(x0, outputs)
                val_loss += loss.item()
        val_loss /= len(self.val_loader)
        # Logging validation loss to tensorboard
        self.writer.add_scalar('val_loss', val_loss, self.current_epoch)
        # Saving checkpoint if it's the best!
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            print(f"New best val_loss: {self.best_val_loss}")
            save_checkpoint({
                'epoch': self.current_epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, filename=(
                f'{self.model_dir}/checkpoints/'
                f'checkpoint_epoch_{self.current_epoch}_val-'
                f'{val_loss}.pth.tar')
                )
        return val_loss


if __name__ == '__main__':
    batch_size = 1
    train_split = 0.8
    patch_directory = ('/autofs/cluster/octdata2/users/epc28/oct_diffusion/'
                       'data/x0')
    model_dir = '/autofs/cluster/octdata2/users/epc28/oct_diffusion/models'

    transform = RandomMarkovChainSampler(200)
    dataset = PatchDirectoryDataset(patch_directory, transform)

    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
        )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = ResUNet().cuda()

    trainer = Trainer(
        model_dir=model_dir,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=torch.nn.MSELoss(),
        # lr=1e-6, weight_decay=1e-9
        optimizer=torch.optim.Adam(
            model.parameters(),
            lr=1e-5,
            weight_decay=1e-5),
        )
    trainer.train(1000)
