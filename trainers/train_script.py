"""train_script.py

Trains the models using the trainer.
"""

from trainers.deeper_unet import UNet
from trainers.trainer import *

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


training_data = np.load("synthesisers/au_on_graphene_gaussian_training_data.npz")

train_images, test_images = training_data['X_train'].astype(np.single)[:20], training_data['X_test'].astype(np.single)[:5]
train_labels, test_labels = training_data['y_train'].astype(np.single)[:20], training_data['y_test'].astype(np.single)[:5]

# PyTorch expects channel in axis 1
train_images, test_images = np.moveaxis(train_images, -1, 1), np.moveaxis(test_images, -1, 1)
train_labels, test_labels = np.moveaxis(train_labels, -1, 1), np.moveaxis(test_labels, -1, 1)

device = torch.device('cuda')

initial_kernels = 32
layers = 4
model = UNet(1, 2, initial_kernels, layers, train_images.shape[-2:])

criterion = TverskyLoss(0.8, 2.2)
optimiser = torch.optim.Adam(model.parameters(), lr=1e-6)

trainer = Trainer(model, train_images, train_labels, test_images, test_labels, criterion, optimiser, device, 
batch_size=1, training_cycles=3000)

trainer.run()

loss_figure = trainer.plot_losses()

# Eval
model = model.to('cpu')
test_images = test_images[:3]
test_labels = test_labels[:3]
outputs = model.predict(test_images)

fig, axes = plt.subplots(len(test_images), 5, sharex=True, sharey=True)
img_min, img_max = test_images.min(), test_images.max()
for ax_row, test_image, test_label, prediction in zip(axes, test_images, test_labels, outputs):
    ax_row[0].imshow(test_image[0]**0.5, vmin=img_min, vmax=img_max, interpolation=None)
    ax_row[1].imshow(test_label[0], vmin=0, vmax=1, interpolation=None)
    ax_row[2].imshow(test_label[1], vmin=0, vmax=1, interpolation=None)
    ax_row[3].imshow(prediction[0], interpolation=None)
    ax_row[4].imshow(prediction[1], interpolation=None)
axes[0, 0].set_title("Input Image (sqrt)")
axes[0, 1].set_title("Lattice Atom GT Layer")
axes[0, 2].set_title("Adatom GT Layer")
axes[0, 3].set_title("Lattice Prediction Layer")
axes[0, 4].set_title("Adatom Prediction Layer")
for ax in axes.flatten(): 
    ax.set_axis_off()
fig.suptitle("U-Net with " + str(criterion))

plt.show()