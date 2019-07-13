import numpy as np
from sklearn.manifold import TSNE
import os

from time import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import offsetbox


# Scale and visualize the embedding vectors
def plot_embedding3d(X, y, title=None):
    NUM_COLORS = np.max(y)
    cm = plt.get_cmap('hsv')

    x_min, x_max = np.min(X, 0), np.max(X, 0)
    print(x_min, x_max)
    X = (X - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    marker_array = [r"${}$".format(y[i]) for i in range(y.shape[0])]
    repr(marker_array).replace("$", r"\$")

    plt.xticks([]), plt.yticks([])

    if title is not None:
        plt.title(title)
    # plt.show()

    for i in range(X.shape[0]):
        ax.scatter(xs=X[i, 0], ys=X[i, 1], zs=X[i, 2], c=[cm(1. * y[i] / (1 + NUM_COLORS))], marker=marker_array[i],
                   alpha=0.6, s=4)

    def init():
        return fig,

    def animate(i):
        ax.view_init(elev=i, azim=i)
        return fig,

    # Animate
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=120, interval=20, blit=True)
    # Save
    anim.save('{}.mp4'.format(title), fps=30, extra_args=['-vcodec', 'libx264'], bitrate=2000, dpi=600)

# Scale and visualize the embedding vectors
def plot_embedding(X, y, title=None):
    NUM_COLORS = np.max(y)
    cm = plt.get_cmap('hsv')

    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    # ax.set_color_cycle([cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=cm(1. * y[i] / (1+NUM_COLORS)),
                 fontdict={'weight': 'bold', 'size': 9})

    # if hasattr(offsetbox, 'AnnotationBbox'):
    #     # only print thumbnails with matplotlib > 1.0
    #     shown_images = np.array([[1., 1.]])  # just something big
    #     for i in range(X.shape[0]):
    #         dist = np.sum((X[i] - shown_images) ** 2, 1)
    #         if np.min(dist) < 4e-3:
    #             # don't show points that are too close
    #             continue
    #         shown_images = np.r_[shown_images, [X[i]]]
    #         imagebox = offsetbox.AnnotationBbox(
    #             offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
    #             X[i])
    #         ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()



def visualize_tsne(path_to_dir):
    embeddings = np.load(os.path.join(path_to_dir, 'test_representations.npy'))
    level_labels = np.load(os.path.join(path_to_dir, 'test_level_labels.npy'))
    print(embeddings.shape)
    print(level_labels.shape)

    X = embeddings
    level_labels = level_labels
    t0 = time()
    X_embedded = TSNE(n_components=3, random_state=0).fit_transform(X)
    print(X_embedded.shape)
    for i in range(4):
        plot_embedding3d(X_embedded, level_labels[:, i], "t-SNE embedding for level {}".format(i))
        print('completed animation for level {}'.format(i))
        break


visualize_tsne('')