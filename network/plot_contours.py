import numpy as np
import os
import math

import matplotlib
# matplotlib.use('pdf')
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi, voronoi_plot_2d


def func(X, Y, vec, thresh=0.0):
    # X = X/np.sqrt(X**2+Y**2)
    # Y = Y/np.sqrt(X**2+Y**2)
    # vec = vec/np.sqrt(vec[0]**2+vec[1]**2)
    dot_p = X*vec[0] + Y*vec[1]
    return dot_p


def get_summary(path_to_summary):
    f = open(path_to_summary, 'r')
    data = {}

    for x in f:
        info = x.split('|')
        data[info[1][3:-3]] = (int(info[5]), float(info[4]))
    return data


def plot_2d_emb_old(save_to_disk=True):
    # path_to_emb = '/home/ankit/Desktop/emb_weights/cnn2d/embedding_info.npy'
    path_to_emb = '/home/ankit/Desktop/emb_weights/cnn2d/embedding_info.npy'
    summary = get_summary(os.path.join(os.path.dirname(path_to_emb), 'summary.md'))
    emb_info = np.load(path_to_emb).item()

    embeddings_x, embeddings_y = emb_info['x'], emb_info['y']
    annotation, color_list = emb_info['annotation'], emb_info['color']
    connected_to = emb_info['connected_to']

    max_val = {}
    for label_ix in embeddings_x:
        if color_list[label_ix] not in max_val:
            max_val[color_list[label_ix]] = summary[annotation[label_ix]][0]
        max_val[color_list[label_ix]] = max(max_val[color_list[label_ix]], summary[annotation[label_ix]][0])

    fig, ax = plt.subplots()

    for label_ix in embeddings_x:
        # ax.scatter(embeddings_x[label_ix], embeddings_y[label_ix], c=color_list[label_ix], alpha=max(0.05, summary[annotation[label_ix]][1]))
        # ax.scatter(embeddings_x[label_ix], embeddings_y[label_ix], c=color_list[label_ix], alpha=max(0.0, summary[annotation[label_ix]][0]/max_val[color_list[label_ix]]))
        ax.scatter(embeddings_x[label_ix], embeddings_y[label_ix], c=color_list[label_ix], alpha=1)

        # if color_list[label_ix] not in ['y', 'k']:
            # ax.annotate(annotation[label_ix], (embeddings_x[label_ix], embeddings_y[label_ix]))
            # ax.annotate(str(summary[annotation[label_ix]][0]), (embeddings_x[label_ix], embeddings_y[label_ix]))
            # ax.annotate('{:.4f}, {}'.format(summary[annotation[label_ix]][1], summary[annotation[label_ix]][0]), (embeddings_x[label_ix], embeddings_y[label_ix]))


    for from_node in connected_to:
        for to_node in connected_to[from_node]:
            if to_node in embeddings_x:
                plt.plot([embeddings_x[from_node], embeddings_x[to_node]], [embeddings_y[from_node], embeddings_y[to_node]],
                         'b-', alpha=0.2)

    # plot contour for
    # delta = 0.01
    # plot_contours_for_label_id = 16
    # x = np.arange(-8.0, 8.0, delta)
    # y = np.arange(-8.0, 8.0, delta)
    # X, Y = np.meshgrid(x, y)
    # Z = func(X, Y, np.array([embeddings_x[plot_contours_for_label_id], embeddings_y[plot_contours_for_label_id]]))
    # CS = ax.contour(X, Y, Z, levels=100, alpha=0.6)
    # ax.clabel(CS, inline=1, fontsize=10)
    # ax.scatter(embeddings_x[plot_contours_for_label_id], embeddings_y[plot_contours_for_label_id], c='r', alpha=1)
    # plt.plot([0.0, embeddings_x[plot_contours_for_label_id]], [0.0, embeddings_y[plot_contours_for_label_id]], 'r-',
    #          alpha=1.0)
    ax.axis('equal')

    points = []
    for label_ix in embeddings_x:
        if color_list[label_ix] == 'm':
            points.append([embeddings_x[label_ix], embeddings_y[label_ix]])
    points = np.array(points)

    # compute Voronoi tesselation
    vor = Voronoi(points)
    fig = voronoi_plot_2d(vor, ax, show_vertices=False, line_colors='orange', line_width=2)


    if save_to_disk:
        fig.set_size_inches(8, 7)
        fig.savefig(os.path.join(os.path.dirname(path_to_emb), 'embedding.pdf'), dpi=200)
        fig.savefig(os.path.join(os.path.dirname(path_to_emb), 'embedding.png'), dpi=200)
        print('Successfully saved embedding to disk!')
    else:
        plt.show()


def plot_2d_emb(save_to_disk=True):
    # path_to_emb = '/home/ankit/Desktop/emb_weights/cnn2d/embedding_info.npy'
    path_to_emb = '/home/ankit/Desktop/emb_weights/cnn2d_v2/0199_embedding_info.npy'
    summary = get_summary(os.path.join(os.path.dirname(path_to_emb), 'summary.md'))
    emb_info = np.load(path_to_emb).item()

    embeddings_x, embeddings_y = emb_info['x'], emb_info['y']
    annotation, color_list = emb_info['annotation'], emb_info['color']
    connected_to = emb_info['connected_to']

    max_val = {}
    for label_ix in embeddings_x:
        if color_list[label_ix] not in max_val:
            max_val[color_list[label_ix]] = summary[annotation[label_ix]][0]
        max_val[color_list[label_ix]] = max(max_val[color_list[label_ix]], summary[annotation[label_ix]][0])

    fig, ax = plt.subplots()

    # dot product voronoi
    extent = 0.0
    points = []
    for label_ix in embeddings_x:
        if extent < embeddings_x[label_ix]:
            extent = embeddings_x[label_ix]
        if extent < embeddings_y[label_ix]:
            extent = embeddings_y[label_ix]
        if color_list[label_ix] == 'k':
            points.append([embeddings_x[label_ix], embeddings_y[label_ix]])
    points = np.array(points)
    print(points.shape)
    print(extent)

    delta = 0.05
    extent = extent + 1
    x = np.arange(-extent, extent, delta)
    y = np.arange(-extent, extent, delta)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((points.shape[0], X.shape[0], X.shape[1]))
    for point_ix in range(points.shape[0]):
        Z[point_ix, :, :] = func(X, Y, np.array([points[point_ix, 0], points[point_ix, 1]]))
    closest_centroid = np.argmax(Z, axis=0)

    cmap = matplotlib.cm.get_cmap('gist_heat')
    from scipy.spatial import ConvexHull
    for point_ix in range(points.shape[0]):
        loc = np.where(closest_centroid == point_ix)
        x_hull, y_hull = X[loc], Y[loc]
        if x_hull.shape != (0, ):
            hull = ConvexHull(np.transpose(np.array([x_hull, y_hull])))
            # ax.fill(x_hull[hull.vertices], y_hull[hull.vertices], c=cmap(point_ix/points.shape[0]), alpha=0.3)
            ax.fill(x_hull[hull.vertices], y_hull[hull.vertices], point_ix, alpha=0.3)
        # ax.scatter(X, Y, c=closest_centroid, alpha=0.6, cmap='gist_heat')

    for label_ix in embeddings_x:
        # ax.scatter(embeddings_x[label_ix], embeddings_y[label_ix], c=color_list[label_ix], alpha=max(0.05, summary[annotation[label_ix]][1]))
        # ax.scatter(embeddings_x[label_ix], embeddings_y[label_ix], c=color_list[label_ix], alpha=max(0.0, summary[annotation[label_ix]][0]/max_val[color_list[label_ix]]))
        ax.scatter(embeddings_x[label_ix], embeddings_y[label_ix], c=color_list[label_ix], alpha=1)

        # if color_list[label_ix] not in ['y', 'k']:
            # ax.annotate(annotation[label_ix], (embeddings_x[label_ix], embeddings_y[label_ix]))
            # ax.annotate(str(summary[annotation[label_ix]][0]), (embeddings_x[label_ix], embeddings_y[label_ix]))
            # ax.annotate('{:.4f}, {}'.format(summary[annotation[label_ix]][1], summary[annotation[label_ix]][0]), (embeddings_x[label_ix], embeddings_y[label_ix]))


    for from_node in connected_to:
        for to_node in connected_to[from_node]:
            if to_node in embeddings_x:
                plt.plot([embeddings_x[from_node], embeddings_x[to_node]], [embeddings_y[from_node], embeddings_y[to_node]],
                         'b-', alpha=0.2)

    # plot contour for
    # delta = 0.01
    # plot_contours_for_label_id = 16
    # x = np.arange(-8.0, 8.0, delta)
    # y = np.arange(-8.0, 8.0, delta)
    # X, Y = np.meshgrid(x, y)
    # Z = func(X, Y, np.array([embeddings_x[plot_contours_for_label_id], embeddings_y[plot_contours_for_label_id]]))
    # CS = ax.contour(X, Y, Z, levels=100, alpha=0.6)
    # ax.clabel(CS, inline=1, fontsize=10)
    # ax.scatter(embeddings_x[plot_contours_for_label_id], embeddings_y[plot_contours_for_label_id], c='r', alpha=1)
    # plt.plot([0.0, embeddings_x[plot_contours_for_label_id]], [0.0, embeddings_y[plot_contours_for_label_id]], 'r-',
    #          alpha=1.0)

    ax.axis('equal')



    if save_to_disk:
        fig.set_size_inches(8, 7)
        fig.savefig(os.path.join(os.path.dirname(path_to_emb), 'embedding.pdf'), dpi=200)
        fig.savefig(os.path.join(os.path.dirname(path_to_emb), 'embedding.png'), dpi=200)
        print('Successfully saved embedding to disk!')
    else:
        plt.show()

plot_2d_emb(save_to_disk=False)
