# imports
import matplotlib.pyplot as plt
import numpy as np
import torch

# helper function to show an image
# (used in the `plot_classes_preds` function below)
# TODO: Make this function more generic as now unnormalize is hard coded.


def matplotlib_imshow(img, one_channel=False) -> plt.imshow:
    """Take a batch of image from the dataloader and plot it out. 

    Unnormalize formula: z = \frac{x-\mu}{\sigma}

    One channel: For our case, mean = (0.1307,) | std = (0.3081,)
    Thus, given an input image matrix x, we normalize to z = (x-0.1307)/0.3081
    To unnormalize: x = z*0.3081 + 0.1307

    Args:
        img ([type]): [description]
        one_channel (bool, optional): [description]. Defaults to False.
    """
    if one_channel:
        img = img.mean(dim=0)
    img = img * 0.3081 + 0.1307     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


# helper function
def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]
