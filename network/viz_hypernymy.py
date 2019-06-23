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
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from network.order_embeddings import OrderEmbedding, OrderEmbeddingLoss, SimpleEuclideanEmbLoss, Embedder, EmbeddingMetrics, ETHECHierarchy
from network.order_embeddings_images import create_imageless_dataloaders
from network.inference import Inference


class VizualizeGraphRepresentation:
    def __init__(self, debug=False,
                 dim=2,
                 weights_to_load='/home/ankit/learning_embeddings/exp/ethec_debug/oe_debug/oe_rm_50/weights/best_model.pth'):
        torch.manual_seed(0)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        labelmap = ETHECLabelMapMerged()
        if debug:
            labelmap = ETHECLabelMapMergedSmall()

        dataloaders = create_imageless_dataloaders(debug=debug)

        G = nx.DiGraph()
        for index, data_item in enumerate(dataloaders['train']):
            inputs, labels, level_labels = data_item['image'], data_item['labels'], data_item['level_labels']
            for level_id in range(len(labelmap.levels)-1):
                for sample_id in range(level_labels.shape[0]):
                    G.add_edge(level_labels[sample_id, level_id].item()+labelmap.level_start[level_id],
                               level_labels[sample_id, level_id+1].item()+labelmap.level_start[level_id+1])

        self.G = G
        self.G_tc = nx.transitive_closure(self.G)
        self.labelmap = labelmap

        self.model = Embedder(embedding_dim=dim, labelmap=labelmap, normalize=False)

        self.load_model(weights_to_load)

        # run vizualize
        self.vizualize()

    def load_model(self, weights_to_load):
        checkpoint = torch.load(weights_to_load,
                                map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.epoch = checkpoint['epoch']
        self.optimal_threshold = checkpoint['optimal_threshold']
        print('Using optimal threshold = {}'.format(self.optimal_threshold))
        print('Successfully loaded model and img_feat_net epoch {} from {}'.format(self.epoch, weights_to_load))

    def vizualize(self):
        phase = 'test'
        self.model.eval()

        colors = ['c', 'm', 'y', 'k']
        embeddings_x, embeddings_y, annotation, color_list = [], [], [], []

        connected_to = {}

        for level_id in range(len(self.labelmap.levels)):
            level_start, level_stop = self.labelmap.level_start[level_id], self.labelmap.level_stop[level_id]
            level_color = colors[level_id]
            for label_ix in range(self.labelmap.levels[level_id]):
                emb_id = label_ix + level_start
                emb = self.model(torch.tensor([emb_id]).to(self.device)).cpu().detach()
                emb = emb[0].numpy()

                embeddings_x.append(emb[0])
                embeddings_y.append(emb[1])
                annotation.append('{}'.format(getattr(self.labelmap,
                                                      '{}_ix_to_str'.format(self.labelmap.level_names[level_id]))[label_ix]
                                                 ))
                color_list.append(level_color)

                connected_to[emb_id] = [v for u, v in list(self.G.edges(emb_id))]

        fig, ax = plt.subplots()
        ax.scatter(embeddings_x, embeddings_y, c=color_list)

        for from_node in connected_to:
            for to_node in connected_to[from_node]:
                plt.plot([embeddings_x[from_node], embeddings_x[to_node]], [embeddings_y[from_node], embeddings_y[to_node]],
                         'b-')

        # for i, txt in enumerate(annotation):
        #     ax.annotate(txt, (embeddings_x[i], embeddings_y[i]))

        plt.show()

obj = VizualizeGraphRepresentation(debug=False)