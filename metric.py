import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import pandas as pd
import imageio
from pathlib import Path

# Get image
im_id = '01d44a26f6680c42ba94c9bc6339228579a95d0e2695b149b7cc0c9592b21baf'
im_dir = Path('/home/vajira/Desktop/Data mining Bowl challenge/stage1_train/{}'.format(im_id))
im_path = im_dir / 'images' / '{}.png'.format(im_id)
im = imageio.imread(im_path.as_posix())


# Get masks
targ_masks = []
for mask_path in im_dir.glob('masks/*.png'):
    targ = imageio.imread(mask_path.as_posix())
    targ_masks.append(targ)
targ_masks = np.stack(targ_masks)



# Make messed up masks
pred_masks = np.zeros(targ_masks.shape)
for ind, orig_mask in enumerate(targ_masks):
    aug_mask = ndimage.rotate(orig_mask, ind*1.5, 
                              mode='constant', reshape=False, order=0)
    pred_masks[ind] = ndimage.binary_dilation(aug_mask, iterations=1)

# Plot the objects
fig, axes = plt.subplots(1,3, figsize=(16,9))
axes[0].imshow(im)
axes[1].imshow(targ_masks.sum(axis=0),cmap='hot')
axes[2].imshow(pred_masks.sum(axis=0), cmap='hot')

labels = ['Original', '"GroundTruth" Masks', '"Predicted" Masks']
for ind, ax in enumerate(axes):
    ax.set_title(labels[ind], fontsize=18)
    ax.axis('off')

