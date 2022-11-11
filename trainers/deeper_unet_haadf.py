"""deeper_unet_haadf.py

A U-Net trainer for training atom and adatom finders for gold on graphene. Uses a custom U-Net, deeper than O.G. and 
only uses the HAADF channels.

THIS DOES NOT WORK!
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

import atomai
import atomai.nets


training_data = np.load("synthesisers/au_on_graphene_scrambled_gaussian_training_data.npz")

train_images, test_images = training_data['X_train'].astype(np.single), training_data['X_test'].astype(np.single)
train_labels, test_labels = training_data['y_train'].astype(np.int8), training_data['y_test'].astype(np.int8)

# PyTorch expects channel in axis 1
# train_images, test_images = np.moveaxis(train_images, -1, 1), np.moveaxis(test_images, -1, 1)
# train_labels, test_labels = np.moveaxis(train_labels, -1, 1), np.moveaxis(test_labels, -1, 1)

# train_labels_temp = train_labels[:, 0]; train_labels_temp[train_labels[:, 1]>0.5] = 2
# test_labels_temp = test_labels[:, 0]; test_labels_temp[test_labels[:, 1]>0.5] = 2
# train_labels = train_labels_temp[:, None, ...]
# test_labels = test_labels_temp[:, None, ...]


# Plot a sample
# fig, axes = plt.subplots(3, 3, sharex=True, sharey=True)
# for row_num, row_ax in enumerate(axes):
#     row_ax[0].imshow(train_images[row_num, 0])
#     row_ax[1].imshow(train_labels[row_num, 0])
#     row_ax[2].imshow(train_labels[row_num, 1])
#     row_ax[0].set_ylabel(f"Training image #{row_num}")
# for col_ax, col_title in zip(axes.T, ("Image", "Carbon", "Gold")):
#     col_ax[0].set_title(col_title)
#     for ax in col_ax:
#         ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)
# plt.show(block=True)


# Define U-Net Model
# torch_encoder = torch.nn.Sequential(
#     atomai.nets.ConvBlock(2, 1, 1, 8, batch_norm=True),
#     torch.nn.MaxPool2d(2, 2),
#     atomai.nets.ConvBlock(2, 2, 8, 16, batch_norm=False),
#     torch.nn.MaxPool2d(2, 2),
#     atomai.nets.ConvBlock(2, 2, 16, 32, batch_norm=False),
#     torch.nn.MaxPool2d(2, 2),
#     atomai.nets.ConvBlock(2, 2, 32, 64, batch_norm=True),
#     torch.nn.MaxPool2d(2, 2),
#     atomai.nets.ConvBlock(2, 2, 64, 128, batch_norm=False)
# )

# torch_decoder = torch.nn.Sequential(
#     atomai.nets.UpsampleBlock(2, 128, 128, mode='nearest'),
#     atomai.nets.ConvBlock(2, 2, 128, 64, batch_norm=False),
#     atomai.nets.UpsampleBlock(2, 64, 64, mode='nearest'),
#     atomai.nets.ConvBlock(2, 2, 64, 32, batch_norm=False),
#     atomai.nets.UpsampleBlock(2, 32, 32, mode='nearest'),
#     atomai.nets.ConvBlock(2, 2, 32, 16, batch_norm=False),
#     atomai.nets.UpsampleBlock(2, 16, 16, mode='nearest'),
#     atomai.nets.ConvBlock(2, 2, 16, 8, batch_norm=False),
#     atomai.nets.UpsampleBlock(2, 8, 8, mode='nearest'),
#     atomai.nets.ConvBlock(2, 1, 8, 2, batch_norm=False)  # Output 2 layers!
# )

# torch_unet = torch.nn.Sequential(torch_encoder, torch_decoder)
# trainer = atomai.trainers.BaseTrainer()
# trainer.set_model(torch_unet, nb_classes=2)

# trainer._reset_weights()
# trainer._reset_training_history()

# trainer.compile_trainer(
#     (train_images, train_labels, test_images, test_labels), loss='dice', training_cycles=1000, swa=False
# )
# trained_model = trainer.run()

# predictor = atomai.predictors.BasePredictor(trained_model, use_gpu=True)

print(np.unique(train_labels))

model = atomai.models.Segmentor("SegResNet", nb_classes=3)
model.fit(train_images, train_labels, test_images, test_labels, loss='dice', training_cycles=1000, swa=True, compute_accuracy=True)

nn_output, _ = model.predict(test_images[0])