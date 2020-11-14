import visdom, pdb
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import matplotlib.patches as patches


class visual():
    def __init__(self, port=8097):
        self.vis = visdom.Visdom(port=port)
        self.counter = 0
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        self.var = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    def denormalize(self, img):
        img = img.cpu().detach().numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def plot_error(self, errors, win=0, id_val=1):
        if not hasattr(self, 'plot_data'):
            self.plot_data = [{'X': [], 'Y': [], 'legend': list(errors.keys())}]
        elif len(self.plot_data) != id_val:
            self.plot_data.append({'X': [], 'Y': [], 'legend': list(errors.keys())})
        id_val -= 1
        self.plot_data[id_val]['X'].append(self.counter)
        self.plot_data[id_val]['Y'].append([errors[k] for k in self.plot_data[id_val]['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data[id_val]['X'])] * len(self.plot_data[id_val]['legend']), 1),
            Y=np.array(self.plot_data[id_val]['Y']),
            opts={
                'legend': self.plot_data[id_val]['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'}, win=win)
        self.counter += 1

    def plot_img(self, img, win=1, name='img'):
        self.vis.image(img, win=win, opts={'title': name})

    def plot_img_list(self, img, name='img', win=1):
        fig = plt.figure()
        for i in range(len(img)):
            ax = fig.add_subplot(1, len(img), i + 1)
            ax.imshow(img[i])
        self.vis.matplot(fig, win=win, opts={'title': name})
        plt.clf()

    def plot_box(self, im1, gt_box1, im2, gt_box2, box, name='img', win=1):
        im1 = self.denormalize(im1)
        im2 = self.denormalize(im2)
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.imshow(im1)
        p = patches.Rectangle(
            (gt_box1[0], gt_box1[1]), gt_box1[2] - gt_box1[0], gt_box1[3] - gt_box1[1],
            fill=False, clip_on=False, color='r'
        )
        ax.add_patch(p)

        ax = fig.add_subplot(122)
        ax.imshow(im2)
        p = patches.Rectangle(
            (gt_box2[0], gt_box2[1]), gt_box2[2] - gt_box2[0], gt_box2[3] - gt_box2[1],
            fill=False, clip_on=False, color='r'
        )
        ax.add_patch(p)
        box = box.copy()
        box[:, 2] -= box[:, 0]
        box[:, 3] -= box[:, 1]
        for i in range(box.shape[0]):
            p = patches.Rectangle(
                (box[i, 0], box[i, 1]), box[i, 2], box[i, 3],
                fill=False, clip_on=False, color='b'
            )
            ax.add_patch(p)
        self.vis.matplot(fig, win=win, opts={'title': name})
        plt.clf()
