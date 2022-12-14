"""train_script.py

Trains the models using the trainer.
"""

import torch
import torch.nn as nn
import numpy as np
import numba
import matplotlib.pyplot as plt

from trainers.deeper_unet import UNet
from trainers.trainer import *

######################
# Load training data #
######################

print("Loading training data")

training_data = np.load("synthesisers/au_on_graphene_scrambled_gaussian_training_data.npz")

train_images, test_images = training_data['X_train'].astype(np.single)[:450], training_data['X_test'].astype(np.single)[:150]
train_labels, test_labels = training_data['y_train'].astype(np.single)[:450], training_data['y_test'].astype(np.single)[:150]

print("Training data loaded")

# PyTorch expects channel in axis 1
train_images, test_images = np.moveaxis(train_images, -1, 1), np.moveaxis(test_images, -1, 1)
train_labels, test_labels = np.moveaxis(train_labels, -1, 1), np.moveaxis(test_labels, -1, 1)

@numba.njit(parallel=True)  # Only 30% speedup, mainly for practice
def generate_background(train_images, test_images, train_labels, test_labels):
    # Add a background channel as NOT the other channels
    train_background = (train_labels.sum(axis=1) < 0.5).astype(np.single)
    train_background = np.expand_dims(train_background, axis=1)
    train_labels = np.append(train_labels, train_background, axis=1)
    test_background = (test_images.sum(axis=1) < 0.5).astype(np.single)
    test_background = np.expand_dims(test_background, axis=1)
    test_labels = np.append(test_labels, test_background, axis=1)
    return train_images, test_images, train_labels, test_labels

train_images, test_images, train_labels, test_labels = generate_background(train_images, test_images, train_labels, test_labels)

# Collapse lattice and adatom classes into one
# train_labels = np.clip(train_labels[:, 0] + train_labels[:, 1], 0, 1)[:, None]
# test_labels = np.clip(test_labels[:, 0] + test_labels[:, 1], 0, 1)[:, None]
# train_labels = np.concatenate((train_labels, np.logical_not(train_labels)), axis=1)
# test_labels = np.concatenate((test_labels, np.logical_not(test_labels)), axis=1)

print("Training data configured")

################
# Define model #
################

device = torch.device('cuda:1')

initial_kernels = 32
layers = 3
model = UNet(1, 3, initial_kernels, layers, train_images.shape[-2:])

criterion = nn.BCELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)

trainer = Trainer(model, train_images, train_labels, test_images, test_labels, criterion, optimiser, device, 
batch_size=250, training_cycles=3000)

###############
# Train model #
###############

trainer.run()

##################
# Evaluate model #
##################

loss_figure = trainer.plot_losses()


def test_model(plot_test_images, plot_test_labels, plot_outputs, title):
    """Applies the model to the test images and compares to ground truth in a figure. Expects data 
    in (frame_num, channel, X, Y).
    """

    # Pad the outputs to match the input size in a visibly clear way
    plot_test_image_size = plot_test_images.shape[-2:]
    plot_pad = [(image_ax_size - output_ax_size) // 2 for image_ax_size, output_ax_size in zip(plot_test_image_size, plot_outputs.shape[-2:])]
    plot_pad = [(0, 0), (0, 0)] + [(plot_ax_pad, plot_ax_pad) for plot_ax_pad in plot_pad]  # Pad equally to lower & upper bounds of final two axes
    channel_means = np.max(plot_outputs, axis=(0, 2, 3))
    plot_outputs = np.pad(plot_outputs, plot_pad, mode='constant', constant_values=-1)  # It should be impossible for -1 to be present in the U-Net's output due to sigmoid layer, so can be used as a label
    for channel, channel_mean in zip(range(plot_outputs.shape[1]), channel_means):
        channel_output = plot_outputs[:, channel]
        channel_output[channel_output == -1] = channel_mean

    fig, axes = plt.subplots(len(plot_test_images), 6, sharex=True, sharey=True)
    img_min, img_max = plot_test_images.min(), plot_test_images.max()
    for ax_row, plot_test_image, plot_test_label, prediction in zip(axes, plot_test_images, plot_test_labels, plot_outputs):
        ax_row[0].imshow(plot_test_image[0]**0.5, vmin=img_min, vmax=img_max, interpolation=None)
        ax_row[1].imshow(plot_test_label[0], vmin=0, vmax=1, interpolation=None)
        ax_row[2].imshow(plot_test_label[1], vmin=0, vmax=1, interpolation=None)
        ax_row[3].imshow(prediction[0], interpolation=None)
        ax_row[4].imshow(prediction[1], interpolation=None)
        ax_row[5].imshow(prediction[2], interpolation=None)
    axes[0, 0].set_title("Input Image (sqrt)")
    axes[0, 1].set_title("Lattice Atom GT Layer")
    axes[0, 2].set_title("Adatom GT Layer")
    axes[0, 3].set_title("Lattice Prediction Layer")
    axes[0, 4].set_title("Adatom Prediction Layer")
    axes[0, 5].set_title("Background Prediction Layer")
    for ax in axes.flatten(): 
        ax.set_axis_off()
    fig.suptitle(title)
    plt.show()


# Eval
model = model.to('cpu')

# Plot train images
plot_test_images = train_images[:3]
plot_test_labels = train_labels[:3]
plot_outputs = model.predict(plot_test_images)
plot_title = f"U-Net ({str(criterion)}) applied to own training data"
test_model(plot_test_images, plot_test_labels, plot_outputs, plot_title)

# Plot test images
plot_test_images = test_images[:3]
plot_test_labels = test_labels[:3]
plot_outputs = model.predict(plot_test_images)
plot_title = f"U-Net ({str(criterion)}) applied to own testing data"
test_model(plot_test_images, plot_test_labels, plot_outputs, plot_title)

# Plot completely unseen images
plot_test_images = np.moveaxis(training_data['X_test'][-3:].astype(np.float32), -1, 1)
plot_test_labels = np.moveaxis(training_data['y_test'][-3:].astype(np.float32), -1, 1)
plot_outputs = model.predict(plot_test_images)
plot_title = f"U-Net ({str(criterion)}) applied to unseen simulated data"
test_model(plot_test_images, plot_test_labels, plot_outputs, plot_title)

