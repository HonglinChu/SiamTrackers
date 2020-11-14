from __future__ import absolute_import

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from PIL import Image


fig_dict = {}
patch_dict = {}


def show_frame(image, boxes=None, fig_n=1, pause=0.001,
               linewidth=3, cmap=None, colors=None, legends=None):
    r"""Visualize an image w/o drawing rectangle(s).
    
    Args:
        image (numpy.ndarray or PIL.Image): Image to show.
        boxes (numpy.array or a list of numpy.ndarray, optional): A 4 dimensional array
            specifying rectangle [left, top, width, height] to draw, or a list of arrays
            representing multiple rectangles. Default is ``None``.
        fig_n (integer, optional): Figure ID. Default is 1.
        pause (float, optional): Time delay for the plot. Default is 0.001 second.
        linewidth (int, optional): Thickness for drawing the rectangle. Default is 3 pixels.
        cmap (string): Color map. Default is None.
        color (tuple): Color of drawed rectanlge. Default is None.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image[..., ::-1])

    if not fig_n in fig_dict or \
        fig_dict[fig_n].get_size() != image.size[::-1]:
        fig = plt.figure(fig_n)
        plt.axis('off')
        fig.tight_layout()
        fig_dict[fig_n] = plt.imshow(image, cmap=cmap)
    else:
        fig_dict[fig_n].set_data(image)

    if boxes is not None:
        if not isinstance(boxes, (list, tuple)):
            boxes = [boxes]
        
        if colors is None:
            colors = ['r', 'g', 'b', 'c', 'm', 'y'] + \
                list(mcolors.CSS4_COLORS.keys())
        elif isinstance(colors, str):
            colors = [colors]

        if not fig_n in patch_dict:
            patch_dict[fig_n] = []
            for i, box in enumerate(boxes):
                patch_dict[fig_n].append(patches.Rectangle(
                    (box[0], box[1]), box[2], box[3], linewidth=linewidth,
                    edgecolor=colors[i % len(colors)], facecolor='none',
                    alpha=0.7 if len(boxes) > 1 else 1.0))
            for patch in patch_dict[fig_n]:
                fig_dict[fig_n].axes.add_patch(patch)
        else:
            for patch, box in zip(patch_dict[fig_n], boxes):
                patch.set_xy((box[0], box[1]))
                patch.set_width(box[2])
                patch.set_height(box[3])
        
        if legends is not None:
            fig_dict[fig_n].axes.legend(
                patch_dict[fig_n], legends, loc=1,
                prop={'size': 8}, fancybox=True, framealpha=0.5)

    plt.pause(pause)
    plt.draw()
