import numpy as np
import os

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt


def func(X, Y, vec, thresh=0.0):
    X = X/np.sqrt(X**2+Y**2)
    Y = Y/np.sqrt(X**2+Y**2)
    # vec = vec/np.sqrt(vec[0]**2+vec[1]**2)
    dot_p = X*vec[0] + Y*vec[1]
    dot_p[dot_p < thresh] = -1
    return dot_p


def plot_2d_emb():
    path_to_emb = '/home/ankit/Desktop/emb_weights/cnn2d/embedding_info.npy'
    emb_info = np.load(path_to_emb).item()

    embeddings_x, embeddings_y = emb_info['x'], emb_info['y']
    annotation, color_list = emb_info['annotation'], emb_info['color']
    connected_to = emb_info['connected_to']

    fig, ax = plt.subplots()

    for label_ix in embeddings_x:
        ax.scatter(embeddings_x[label_ix], embeddings_y[label_ix], c=color_list[label_ix], alpha=1)
        # ax.annotate(annotation[emb_id], (weights[label_ix][0], weights[label_ix][1]))

    # for from_node in connected_to:
    #     for to_node in connected_to[from_node]:
    #         if to_node in embeddings_x:
    #             plt.plot([embeddings_x[from_node], embeddings_x[to_node]], [embeddings_y[from_node], embeddings_y[to_node]],
    #                      'b-', alpha=0.2)

    # plot contour for
    delta = 0.01
    plot_contours_for_label_id = 16
    x = np.arange(-8.0, 8.0, delta)
    y = np.arange(-8.0, 8.0, delta)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y, np.array([embeddings_x[plot_contours_for_label_id], embeddings_y[plot_contours_for_label_id]]))
    CS = ax.contour(X, Y, Z, levels=100, alpha=0.6)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.scatter(embeddings_x[plot_contours_for_label_id], embeddings_y[plot_contours_for_label_id], c='r', alpha=1)
    plt.plot([0.0, embeddings_x[plot_contours_for_label_id]], [0.0, embeddings_y[plot_contours_for_label_id]], 'r-',
             alpha=1.0)

    # if self.title_text:
    #     fig.suptitle(self.title_text, family='sans-serif')
    fig.set_size_inches(8, 7)
    ax.axis('equal')
    fig.savefig(os.path.join(os.path.dirname(path_to_emb), 'embedding.pdf'), dpi=200)
    fig.savefig(os.path.join(os.path.dirname(path_to_emb), 'embedding.png'), dpi=200)
    print('Successfully saved embedding to disk!')

plot_2d_emb()