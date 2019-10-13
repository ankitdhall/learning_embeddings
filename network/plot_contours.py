import numpy as np
import os
import math

import matplotlib
# matplotlib.use('pdf')
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import multiprocessing
import torch
from tqdm import tqdm

from scipy.spatial import Voronoi, voronoi_plot_2d

from data.db import ETHECLabelMapMerged

import networkx as nx
import time

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


class EmbeddingMetrics:
    def __init__(self, e_for_u_v_positive, e_for_u_v_negative, threshold, phase, n_proc=4):
        self.e_for_u_v_positive = e_for_u_v_positive.view(-1)
        self.e_for_u_v_negative = e_for_u_v_negative.view(-1)
        self.threshold = threshold
        self.phase = phase

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_proc = self.n_proc = 256 if torch.cuda.device_count() > 0 else 4

    def calculate_best(self, threshold):
        correct_positives = torch.sum(self.e_for_u_v_positive <= threshold).item()
        correct_negatives = torch.sum(self.e_for_u_v_negative > threshold).item()
        accuracy = (correct_positives + correct_negatives) / (
                    self.e_for_u_v_positive.shape[0] + self.e_for_u_v_negative.shape[0])
        precision = correct_positives / (correct_positives + (self.e_for_u_v_negative.shape[0] - correct_negatives))
        recall = correct_positives / self.e_for_u_v_positive.shape[0]
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = (2 * precision * recall) / (precision + recall)

        return f1_score, threshold, accuracy, precision, recall

    def calculate_metrics(self):
        if self.phase == 'val':
            possible_thresholds = np.unique(
                np.concatenate((self.e_for_u_v_positive, self.e_for_u_v_negative), axis=None))

            F = np.zeros((possible_thresholds.shape[0], 5))
            pool = multiprocessing.Pool(processes=self.n_proc)

            # if number of processes is not specified, it uses the number of core
            F[:, :] = list(tqdm(pool.imap(self.calculate_best,
                                          [possible_thresholds[t_id] for t_id in range(possible_thresholds.shape[0])]),
                                total=possible_thresholds.shape[0]))
            pool.close()
            pool.join()
            best_index = np.argmax(F[:, 0])
            return F[best_index, :]

        else:
            correct_positives = torch.sum(self.e_for_u_v_positive <= self.threshold).item()
            correct_negatives = torch.sum(self.e_for_u_v_negative > self.threshold).item()
            accuracy = (correct_positives + correct_negatives) / (
                    self.e_for_u_v_positive.shape[0] + self.e_for_u_v_negative.shape[0])

            if correct_positives + (self.e_for_u_v_negative.shape[0] - correct_negatives) == 0:
                print('Encountered NaN for precision!')
                precision = 0.0
            else:
                precision = correct_positives / (
                            correct_positives + (self.e_for_u_v_negative.shape[0] - correct_negatives))
            recall = correct_positives / self.e_for_u_v_positive.shape[0]
            if precision + recall == 0:
                f1_score = 0.0
            else:
                f1_score = (2 * precision * recall) / (precision + recall)
            return f1_score, self.threshold, accuracy, precision, recall


class ReconstructionMetricsCNN2D:
    def __init__(self):
        self.labelmap = ETHECLabelMapMerged()
        self.check_graph_embedding_neg_graph = None
        self.n_proc = 4

    @staticmethod
    def dot_operator(x, y):
        out = np.sum(np.multiply(x, y), axis=1)
        return torch.tensor(out)

    def load_graphs(self, label_embeddings):
        if self.check_graph_embedding_neg_graph is None:
            start_time = time.time()
            path_to_folder = '../database/ETHEC/ETHEC_embeddings/graphs'

            self.G = nx.read_gpickle(os.path.join(path_to_folder, 'G'))

            self.G = self.G
            self.G_tc = nx.transitive_closure(self.G)

            # make negative graph
            n_nodes = len(list(self.G.nodes()))

            A = np.ones((n_nodes, n_nodes), dtype=np.bool)

            for u, v in list(self.G.edges()):
                # remove edges that are in G_train_tc
                A[u, v] = 0
            np.fill_diagonal(A, 0)
            self.check_graph_embedding_neg_graph = A

            self.edges_in_G = self.G.edges()
            self.n_nodes_in_G = len(self.G.nodes())
            self.nodes_in_G = [i for i in range(self.n_nodes_in_G)]

            self.pos_u_list, self.pos_v_list = [], []
            for edge in self.edges_in_G:
                self.pos_u_list.append(edge[0])
                self.pos_v_list.append(edge[1])

            self.neg_u_list, self.neg_v_list = [], []
            for i_ix in range(n_nodes):
                for j_ix in range(n_nodes):
                    if self.check_graph_embedding_neg_graph[i_ix, j_ix] == 1:
                        self.neg_u_list.append(i_ix)
                        self.neg_v_list.append(j_ix)
            print('created negative graph in {}'.format(time.time() - start_time))

        positive_e = self.dot_operator(label_embeddings[self.pos_u_list, :], label_embeddings[self.pos_v_list, :])
        negative_e = self.dot_operator(label_embeddings[self.neg_u_list, :], label_embeddings[self.neg_v_list, :])

        metrics = EmbeddingMetrics(positive_e, negative_e, 0.0,
                                   'val', n_proc=self.n_proc)
        best_score, best_threshold, best_accuracy, best_precision, best_recall = metrics.calculate_metrics()

        print('Checking graph reconstruction: +ve edges {}, -ve edges {}'.format(len(self.edges_in_G),
                                                                                 self.check_graph_embedding_neg_graph.size))
        return best_score, best_threshold, best_accuracy, best_precision, best_recall

    def plot_2d_emb(self, save_to_disk=True):
        # path_to_emb = '/home/ankit/Desktop/emb_weights/cnn2d/embedding_info.npy'
        path_to_emb = '/home/ankit/Desktop/emb_weights/cnn2d_v2/0199_embedding_info.npy'
        summary = get_summary(os.path.join(os.path.dirname(path_to_emb), 'summary.md'))
        emb_info = np.load(path_to_emb).item()

        embeddings_x, embeddings_y = emb_info['x'], emb_info['y']
        annotation, color_list = emb_info['annotation'], emb_info['color']
        connected_to = emb_info['connected_to']

        label_embeddings = np.zeros((len(embeddings_x.keys()), 2))
        for label_ix in embeddings_x:
            label_embeddings[label_ix, 0], label_embeddings[label_ix, 1] = embeddings_x[label_ix], embeddings_y[label_ix]

        # # calculate scores
        # best_score, best_threshold, best_accuracy, best_precision, best_recall = self.load_graphs(label_embeddings)
        # print(best_score, best_threshold, best_accuracy, best_precision, best_recall)

        max_val = {}
        for label_ix in embeddings_x:
            if color_list[label_ix] not in max_val:
                max_val[color_list[label_ix]] = summary[annotation[label_ix]][0]
            max_val[color_list[label_ix]] = max(max_val[color_list[label_ix]], summary[annotation[label_ix]][0])

        fig, ax = plt.subplots()

        # # dot product voronoi
        # extent = 0.0
        # points = []
        # for label_ix in embeddings_x:
        #     if extent < embeddings_x[label_ix]:
        #         extent = embeddings_x[label_ix]
        #     if extent < embeddings_y[label_ix]:
        #         extent = embeddings_y[label_ix]
        #     if color_list[label_ix] == 'k':
        #         points.append([embeddings_x[label_ix], embeddings_y[label_ix]])
        # points = np.array(points)
        # print(points.shape)
        # print(extent)
        #
        # delta = 0.05
        # extent = extent + 1
        # x = np.arange(-extent, extent, delta)
        # y = np.arange(-extent, extent, delta)
        # X, Y = np.meshgrid(x, y)
        # Z = np.zeros((points.shape[0], X.shape[0], X.shape[1]))
        # for point_ix in range(points.shape[0]):
        #     Z[point_ix, :, :] = func(X, Y, np.array([points[point_ix, 0], points[point_ix, 1]]))
        # closest_centroid = np.argmax(Z, axis=0)
        #
        # cmap = matplotlib.cm.get_cmap('gist_heat')
        # from scipy.spatial import ConvexHull
        # for point_ix in range(points.shape[0]):
        #     loc = np.where(closest_centroid == point_ix)
        #     x_hull, y_hull = X[loc], Y[loc]
        #     if x_hull.shape != (0, ):
        #         hull = ConvexHull(np.transpose(np.array([x_hull, y_hull])))
        #         # ax.fill(x_hull[hull.vertices], y_hull[hull.vertices], c=cmap(point_ix/points.shape[0]), alpha=0.3)
        #         ax.fill(x_hull[hull.vertices], y_hull[hull.vertices], point_ix, alpha=0.3)
        #     # ax.scatter(X, Y, c=closest_centroid, alpha=0.6, cmap='gist_heat')

        # invert embeddings
        label_norms = np.linalg.norm(label_embeddings, axis=1, ord=2)
        print(label_norms.shape)
        print(np.max(label_norms))
        max_norm = np.max(label_norms)

        for label_ix in embeddings_x:
            embeddings_x[label_ix] = 3.0 * max_norm * embeddings_x[label_ix] / label_norms[label_ix]**2
            embeddings_y[label_ix] = 3.0 * max_norm * embeddings_y[label_ix] / label_norms[label_ix]**2

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

        ax.axis('equal')



        if save_to_disk:
            fig.set_size_inches(8, 7)
            fig.savefig(os.path.join(os.path.dirname(path_to_emb), 'embedding.pdf'), dpi=200)
            fig.savefig(os.path.join(os.path.dirname(path_to_emb), 'embedding.png'), dpi=200)
            print('Successfully saved embedding to disk!')
        else:
            plt.show()

obj = ReconstructionMetricsCNN2D()
obj.plot_2d_emb(save_to_disk=False)
