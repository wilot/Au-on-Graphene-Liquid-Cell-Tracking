# Hyperparameter Tuning

Found that, using the a full set of ~10k 256x256 synthetic HAADF STEM training images and labels, the network would 
fail to learn. Morover, the loss would never approach zero. The network couldn't overfit either.

It was suspected that the network might possibly be failing because:
* Network doesn't have enough capacity
    * Try with much smaller training set, see if it can overfit on ~25 images
    * Increase the networ's capacity
* There is a bug
    * If it cannot be made to overfit then this is a possibility?
* There is a mistake in network architecture
* Loss function might be inadequate? 
    * Try CE instead of BCE?
* There is no background channel
    * Unlikely the cause, but could help

Attempts to remedy the problem are recorded here.

## Small Data Low lr

Tried reducing the training image set to just 25 images (20 train, 5 test) and reduced the learning rate to 1e-6. The
network was set to have 4 layers, 32 initial kernels. The prediction figures are not plotted
from 0->1, but instead from min->max. All the predictions are spread out over only 0.05.