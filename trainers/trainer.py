"""trainer.py

Contains scripts for training of U-Net type models
"""

from pathlib import Path
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import torchvision.transforms

import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt
import sklearn.utils

from tqdm import tqdm


class Trainer:
    """Performs data loading and training of the models.
    
    Trains a network to map input images X to outputs Y. During training, outputted predictions are compared with 
    desired outputs (Y) for each input image (X) and the network is iteratively adjusted to make the the output 
    closer to Y.   
    """

    def __init__(self, network: nn.Module, train_images: np.ndarray, train_labels: np.ndarray, test_images: np.ndarray,
    test_labels: np.ndarray, criterion, optimiser: torch.optim.Optimizer, device: torch.device, batch_size: int, 
    training_cycles: int):

        self.network = network
        self.criterion = criterion
        self.optimiser = optimiser
        self.device = device
        self.batch_size = batch_size if batch_size < test_images.shape[0] else 1
        self.training_cycles = training_cycles

        self.losses = {'train': [], 'test': []}  # History of test and train losses

        self.network.to(self.device)

        to_torch = lambda nparr: torch.from_numpy(nparr)

        self.X_train, self.X_test = to_torch(train_images), to_torch(test_images)
        self.Y_train, self.Y_test = to_torch(train_labels), to_torch(test_labels)
        self.batch_data()  # Should give dimensions (batch_id, frame_num, channel, x_px, y_px)

        # Define the batches to process across
        get_batch_indices = lambda tensor: np.arange(tensor.shape[0]) \
                                             .repeat(self.training_cycles//tensor.shape[0] + 1)[:self.training_cycles]
        self.train_batch_indices = get_batch_indices(self.X_train)
        self.test_batch_indices = get_batch_indices(self.X_test)
        self.train_batch_indices = sklearn.utils.shuffle(self.train_batch_indices)
        self.test_batch_indices = sklearn.utils.shuffle(self.test_batch_indices)


    def train_step(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """Computes a single backpropagation optimisation iteration for a batch. Expects the tensors to be loaded in 
        the device already.
        """
        self.network.train()  # Training mode
        self.optimiser.zero_grad()
        output = self.network(X)
        Y = crop_to_fit(Y, output.shape[-2:])
        loss = self.criterion(output, Y)
        loss.backward()
        self.optimiser.step()
        return loss.item()

    def test_step(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """Computes an evaluation of the network on a batch. Expects the tensors to already be loaded onto the 
        device.
        """
        self.network.eval()
        with torch.no_grad():
            output = self.network(X)
            Y = crop_to_fit(Y, output.shape[-2:])
            loss = self.criterion(output, Y)
        return loss.item()

    def batch_data(self, send_to_device=False):
        """Chunks the test and train data into batches. Ignores any remainder"""

        def chunk_tensor(tensor: torch.Tensor, batch_size: int = self.batch_size) -> torch.Tensor:
            num_batches = tensor.shape[0] // batch_size
            trimmed_tensor = tensor[:num_batches*batch_size]
            return torch.stack(torch.tensor_split(trimmed_tensor, num_batches))

        self.X_train = chunk_tensor(self.X_train)
        self.X_test = chunk_tensor(self.X_test)
        self.Y_train = chunk_tensor(self.Y_train)
        self.Y_test = chunk_tensor(self.Y_test)

        if send_to_device:
            self.X_train = self.X_train.to(self.device)
            self.X_test = self.X_test.to(self.device)
            self.Y_train = self.Y_train.to(self.device)
            self.Y_test = self.Y_test.to(self.device)

    def step(self, training_cycle: int):
        """Train-test step for a mini-batch"""

        train_batch_index = self.train_batch_indices[training_cycle]
        test_batch_index = self.test_batch_indices[training_cycle]
        
        X_train_batch = self.X_train[train_batch_index].to(self.device)
        Y_train_batch = self.Y_train[train_batch_index].to(self.device)
        X_test_batch = self.X_test[test_batch_index].to(self.device, non_blocking=True)
        Y_test_batch = self.Y_test[test_batch_index].to(self.device, non_blocking=True)

        # Training step
        loss = self.train_step(X_train_batch, Y_train_batch)
        self.losses['train'].append(loss / self.batch_size)

        # Test step
        loss = self.test_step(X_test_batch, Y_test_batch)
        self.losses['test'].append(loss / self.batch_size)

    def run(self):
        """Iteratively trains the network through steps"""

        for training_cycle in tqdm(range(self.training_cycles)):
            self.step(training_cycle)

    def plot_losses(self, plot=True) -> matplotlib.figure.Figure:
        """Plots the training and test loss during training (non-blocking)"""
        fig, ax = plt.subplots()
        ax.plot(self.losses['train'], label='Training Loss')
        ax.plot(self.losses['test'], label='Testing Losses')
        ax.set_xlabel("Training Cycle")
        ax.set_ylabel("Loss")
        ax.set_title("Gradient Descent During Training")
        ax.legend()
        if plot:
            plt.show()
        return fig


class CrossEntropyLoss(nn.Module):
    """A Cross Entropy Loss with cropping"""

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):

        if targets.shape[-2:] != inputs.shape[-2:]:
            targets = crop_to_fit(targets, inputs.shape[-2:])

        ce_loss = F.cross_entropy(inputs, targets)

        return ce_loss


class DiceLoss(nn.Module):
    """A custom DICE Loss. I calculate DICE for the classes seperately and then add them together...
    
    The weights should be a tensor of shape (C,) for C channels and on the same device as the model.
    """
    def __init__(self, log_loss: bool=False, weights: Union[torch.Tensor, None]=None):
        super(DiceLoss, self).__init__()
        self.log_loss = log_loss
        self.weights = weights
        self.smooth = 0.1
        

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """Expects inputs and targets of shape (B, C, H, W) for batch, channel, height and width."""

        if targets.shape[-2:] != inputs.shape[-2:]:
            targets = crop_to_fit(targets, inputs.shape[-2:])

        inputs = flatten(inputs)  # To shape (C, B*H*W)
        targets = flatten(targets)

        intersection = (inputs * targets).sum(-1)
        denominator = (inputs*inputs).sum(-1) + (targets*targets).sum(-1)

        dice = (2. * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1 - dice if not self.log_loss else -torch.log(dice)  # Loss for each channel
        if self.weights is None:
            dice_loss = dice_loss.sum() / dice_loss.shape[0]
        else:
            dice_loss = (dice_loss * self.weights).sum()

        return dice_loss

    def __str__(self):
        name = "Dice Loss"
        if self.log_loss: name += f" with log_loss"
        return name


class DiceBCELoss(nn.Module):
    """Combination of Dice and BCE loss. Their ratio can be set by weights (Dice, BCE).
    
    Simply adds a Dice and Binary Cross-Entropy loss together with scale factors defined by ratio. If specified, 
    log-loss is applied to the Dice loss. Weights correspond to class weightings and should be a shape (C,) tensor.
    """
    def __init__(self, log_loss: bool=False, ratio: Union[Tuple[float, float], None]=None, 
    weights: Union[torch.Tensor, None]=None):
        super(DiceBCELoss, self).__init__()
        self.log_loss = log_loss
        if not ratio:
            ratio = (0.5, 0.5)
        elif sum(ratio) > 1:
            raise ValueError("The ratio of DiceBCELoss must sum to 1.")
        self.ratio = ratio
        self.weights = weights

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):

        if targets.shape[-2:] != inputs.shape[-2:]:
            targets = crop_to_fit(targets, inputs.shape[-2:])

        # Dice loss
        inputs = flatten(inputs)  # To shape (C, B*H*W)
        targets = flatten(targets)

        intersection = (inputs * targets).sum(-1)
        denominator = (inputs*inputs).sum(-1) + (targets*targets).sum(-1)

        dice = (2. * intersection + 1.) / (denominator + 1.)
        dice_loss = 1 - dice if not self.log_loss else -torch.log(dice)  # Loss for each channel
        if self.weights is None:
            dice_loss = dice_loss.sum() / dice_loss.shape[0]
        else:
            dice_loss = (dice_loss * self.weights).sum()
        
        # BCE loss
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')

        # Combination
        Dice_BCE_loss = self.ratio[1] * BCE + self.ratio[0] * dice_loss

        if self.weights is not None:
            Dice_BCE_loss = Dice_BCE_loss * self.weights

        return Dice_BCE_loss.sum()

    def __str__(self):
        name = f"({self.ratio[0] :.1f}Dice + {self.ratio[1] :.1f}BCE) Loss"
        if self.log_loss: name += " with log_loss"
        return name


class IoULoss(nn.Module):
    """Jaccard i.e. IoU loss"""

    def __init__(self) -> None:
        super().__init__()

        self.smooth = 1.

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):

        inputs = flatten(inputs)  # (C, B*H*W)
        targets = flatten(targets)

        intersection = (inputs * targets).sum(-1)
        total = (inputs + targets).sum(-1)
        union = total - intersection

        IoU = (intersection + self.smooth) / (union + self.smooth)
        IoU_loss = 1 - IoU  # This is still per channel

        return IoU_loss.sum()


class FocalLoss(nn.Module):
    """Focal Loss, wights are for class weightings, a shape (C,) torch tensor or None."""
    def __init__(self, alpha=0.8, gamma=2., weights: Union[torch.Tensor, None]=None) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weights = weights
    
    def forward(self, inputs, targets):

        if targets.shape[-2:] != inputs.shape[-2:]:
            targets = crop_to_fit(targets, inputs.shape[-2:])

        inputs = flatten(inputs)  # To shape (C, B*H*W)
        targets = flatten(targets)

        BCE = F.binary_cross_entropy(inputs, targets)
        BCE_exp = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_exp)**self.gamma * BCE

        if self.weights is not None:
            focal_loss = focal_loss * self.weights

        focal_loss = focal_loss.sum()
        
        return focal_loss

    def __str__(self):
        name = f"Focal Loss (α={self.alpha :.1f}, γ={self.gamma :.1f})"
        return name


class TverskyLoss(nn.Module):
    """Implements Tversky Loss
    
    Alpha penalises false positives more and beta penalises false negatives more. With alpha=beta=0.5 this becomes DICE
    loss.
    """
    def __init__(self, alpha: float, beta: float, log_loss: bool=False):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.log_loss = log_loss

    def forward(self, inputs, targets, smooth=1):

        if targets.shape[-2:] != inputs.shape[-2:]:
            targets = crop_to_fit(targets, inputs.shape[-2:])

        inputs = torch.reshape(inputs, (-1,))
        targets = torch.reshape(targets, (-1,))

        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()

        Tversky = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)
        return 1 - Tversky if not self.log_loss else -torch.log(Tversky)

    def __str__(self):
        name = f"Tversky Loss (α={self.alpha :.1f}, β={self.beta :.1f})"
        return name


def flatten(tensor: torch.Tensor) -> torch.Tensor:
    "Flattens the tensor along all except the channel dimension i.e. (B, C, H, W) -> (C, B*H*W)"

    new_axis_order = (1, 0, 2, 3)  # Put the channel first, then flatten to shape (C, B*H*W)
    tensor = tensor.permute(new_axis_order).flatten(start_dim=1)
    return tensor


def crop_to_fit(crop_tensor: torch.Tensor, crop_size: Tuple[int, int]) -> torch.Tensor:
    """Crops a tensor to a specified size using centre-crop"""

    cropped_tensor = torchvision.transforms.CenterCrop(crop_size)(crop_tensor)
    return cropped_tensor
