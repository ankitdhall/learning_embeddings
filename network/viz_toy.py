from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

from tensorboardX import SummaryWriter

import os
from network.experiment import Experiment, WeightedResampler
from network.evaluation import MultiLabelEvaluation, Evaluation, MultiLabelEvaluationSingleThresh, MultiLevelEvaluation
from network.finetuner import CIFAR10

from data.db import ETHECLabelMap, ETHECDB, ETHECDBMerged, ETHECLabelMapMerged, ETHECLabelMapMergedSmall, ETHECDBMergedSmall
from network.loss import MultiLevelCELoss, MultiLabelSMLoss, LastLevelCELoss, MaskedCELoss, HierarchicalSoftmaxLoss

from PIL import Image
import numpy as np
import time
from tqdm import tqdm

import copy
import argparse
import json
import git

import torch
from torch import nn
import torch.nn.functional as F
from data.db import ETHECLabelMap, ETHECLabelMapMergedSmall

from network.finetuner import CIFAR10
import numpy as np
import random
random.seed(0)

import networkx as nx
import matplotlib
matplotlib.use('pdf')
#matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from network.order_embeddings import Embedder
from network.embed_toy import ToyGraph


class VizualizeGraphRepresentation:
    def __init__(self,
                 L, b,
                 dim=2, loss_fn='oe',
                 weights_to_load='/home/ankit/Desktop/hypernym_viz/toy/t3/ec_ppl_0.01_0.01_best_model.pth',
                 title_text=''):
        torch.manual_seed(0)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.labelmap = ToyGraph(L, b)

        self.G = nx.DiGraph()
        for edge in self.labelmap.edges:
            u, v = edge
            self.G.add_edge(u, v)

        if loss_fn == 'ec':
            self.model = Embedder(embedding_dim=dim, labelmap=self.labelmap, K=3.0)
        else:
            self.model = Embedder(embedding_dim=dim, labelmap=self.labelmap)
        self.model =nn.DataParallel(self.model)

        self.load_model(weights_to_load)

        self.title_text = title_text
        if self.title_text == '':
            self.title_text = 'L={}, b={} | F1 score: {:.4f} Accuracy: {:.4f} \n Precision: {:.4f} Recall: {:.4f} | Threshold: {:.4f}'.format(
                str(L-1), str(b), self.reconstruction_f1, self.reconstruction_accuracy, self.reconstruction_prec,
                self.reconstruction_recall, self.reconstruction_threshold)

        self.weights_to_load = weights_to_load


        # run vizualize
        self.vizualize()

    def load_model(self, weights_to_load):
        checkpoint = torch.load(weights_to_load,
                                map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.epoch = checkpoint['epoch']
        self.optimal_threshold = checkpoint['optimal_threshold']

        self.reconstruction_f1, self.reconstruction_threshold, self.reconstruction_accuracy, self.reconstruction_prec, self.reconstruction_recall = \
            checkpoint['reconstruction_scores']['f1'], checkpoint['reconstruction_scores']['threshold'], \
            checkpoint['reconstruction_scores']['accuracy'], checkpoint['reconstruction_scores']['precision'], \
            checkpoint['reconstruction_scores']['recall']

        print('Using optimal threshold = {}'.format(self.optimal_threshold))
        print('Successfully loaded model and img_feat_net epoch {} from {}'.format(self.epoch, weights_to_load))

    def vizualize(self, save_to_disk=True, filename='embedding'):
        phase = 'test'
        self.model.eval()

        norm = matplotlib.colors.Normalize(vmin=0, vmax=self.labelmap.n_levels)
        cmap = matplotlib.cm.get_cmap('gnuplot')

        embeddings_x, embeddings_y, annotation, color_list = {}, {}, {}, {}

        connected_to = {}

        fig, ax = plt.subplots()

        for level_id in range(len(self.labelmap.levels)):
            level_start, level_stop = self.labelmap.level_start[level_id], self.labelmap.level_stop[level_id]
            level_color = [cmap(norm(level_id))]
            print(level_color)
            for label_ix in range(self.labelmap.levels[level_id]):
                emb_id = label_ix + level_start
                emb = self.model(torch.tensor([emb_id]).to(self.device)).cpu().detach()
                emb = emb[0].numpy()

                embeddings_x[emb_id] = emb[0]
                embeddings_y[emb_id] = emb[1]
                # annotation[emb_id] = '{}'.format(getattr(self.labelmap,
                #                                       '{}_ix_to_str'.format(self.labelmap.level_names[level_id]))[label_ix]
                #                                  )
                color_list[emb_id] = level_color

                connected_to[emb_id] = [v for u, v in list(self.G.edges(emb_id))]

                ax.scatter(emb[0], emb[1], c=level_color, alpha=1)
                # ax.annotate(annotation[emb_id], (emb[0], emb[1]))

        # fig, ax = plt.subplots()
        # ax.scatter(embeddings_x, embeddings_y, c=color_list)

        for from_node in connected_to:
            for to_node in connected_to[from_node]:
                if to_node in embeddings_x:
                    plt.plot([embeddings_x[from_node], embeddings_x[to_node]], [embeddings_y[from_node], embeddings_y[to_node]],
                             'b-', alpha=0.2)

        # for i, txt in enumerate(annotation):
        #     ax.annotate(txt, (embeddings_x[i], embeddings_y[i]))
        fig.suptitle(self.title_text, family='sans-serif')
        ax.axis('equal')
        if save_to_disk:
            fig.set_size_inches(8, 7)
            fig.savefig(os.path.join(os.path.dirname(self.weights_to_load), '..', '{}.pdf'.format(filename)), dpi=200)
            fig.savefig(os.path.join(os.path.dirname(self.weights_to_load), '..', '{}.png'.format(filename)), dpi=200)

        return ax

def create_images():
    path_to_weights = '/cluster/scratch/adhall/exp/toy_trees/oe/l_5_b_3/weights'
    loss_fn = 'oe'
    files = os.listdir(path_to_weights)
    files.sort()
    for filename in files:
        if 'best_model' in filename:
            continue
        viz = VizualizeGraphRepresentation(weights_to_load=os.path.join(path_to_weights, filename), title_text='', L=5,
                                           b=3, loss_fn=loss_fn)
        viz.vizualize(save_to_disk=True, filename='{0:03d}'.format(int(filename[:-4])))

if __name__ == '__main__':
    # obj = VizualizeGraphRepresentation()
    create_images()

