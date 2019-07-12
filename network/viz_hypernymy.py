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

from network.oe import Embedder, FeatNet
from network.oe import create_imageless_dataloaders, load_combined_graphs, EuclideanConesWithImagesHypernymLoss, OrderEmbeddingWithImagesHypernymLoss
from network.oe import my_collate, ETHECHierarchyWithImages
from network.inference import Inference


class VizualizeGraphRepresentation:
    def __init__(self, debug=False,
                 dim=2,
                 # weights_to_load='/home/ankit/learning_embeddings/exp/ethec_debug/ec_debug/d10/oe10d_debug/weights/best_model.pth'):
                 weights_to_load='/home/ankit/learning_embeddings/exp/ethec_debug/oelwi_debug/ec_load_emb/weights/best_model_model.pth'):
        torch.manual_seed(0)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        labelmap = ETHECLabelMapMerged()
        if debug:
            labelmap = ETHECLabelMapMergedSmall()

        # dataloaders = create_imageless_dataloaders(debug=debug)

        # G = nx.DiGraph()
        # for index, data_item in enumerate(dataloaders['train']):
        #     inputs, labels, level_labels = data_item['image'], data_item['labels'], data_item['level_labels']
        #     for level_id in range(len(labelmap.levels)-1):
        #         for sample_id in range(level_labels.shape[0]):
        #             G.add_edge(level_labels[sample_id, level_id].item()+labelmap.level_start[level_id],
        #                        level_labels[sample_id, level_id+1].item()+labelmap.level_start[level_id+1])

        if debug:
            path_to_folder = '../database/ETHEC/ETHECSmall_embeddings/graphs'
        else:
            path_to_folder = '../database/ETHEC/ETHEC_embeddings/graphs'

        G = nx.read_gpickle(os.path.join(path_to_folder, 'G'))

        self.G = G
        self.G_tc = nx.transitive_closure(self.G)
        self.labelmap = labelmap

        if 'ec' in weights_to_load:
            self.model = Embedder(embedding_dim=dim, labelmap=labelmap, normalize=False, K=3.0)
        else:
            self.model = Embedder(embedding_dim=dim, labelmap=labelmap, normalize=False)#, K=3.0)
        self.model =nn.DataParallel(self.model)

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
        embeddings_x, embeddings_y, annotation, color_list = {}, {}, {}, {}

        connected_to = {}

        fig, ax = plt.subplots()

        for level_id in range(4): #len(self.labelmap.levels)):
            level_start, level_stop = self.labelmap.level_start[level_id], self.labelmap.level_stop[level_id]
            level_color = colors[level_id]
            for label_ix in range(self.labelmap.levels[level_id]):
                emb_id = label_ix + level_start
                emb = self.model(torch.tensor([emb_id]).to(self.device)).cpu().detach()
                emb = emb[0].numpy()

                embeddings_x[emb_id] = emb[0]
                embeddings_y[emb_id] = emb[1]
                annotation[emb_id] = '{}'.format(getattr(self.labelmap,
                                                      '{}_ix_to_str'.format(self.labelmap.level_names[level_id]))[label_ix]
                                                 )
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

        plt.show()


class VizualizeGraphRepresentationWithImages:
    def __init__(self, debug=False,
                 dim=2,
                 loss_fn='oe',
                 weights_to_load='/home/ankit/learning_embeddings/exp/ethec_debug/oelwi_debug/ec_load_emb_lr_0.01/weights/best_model_model.pth',
                 img_weights_to_load='/home/ankit/learning_embeddings/exp/ethec_debug/oelwi_debug/ec_load_emb_lr_0.01/weights/best_model_img_feat_net.pth'):
        torch.manual_seed(0)
        self.load_split = 'train'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        labelmap = ETHECLabelMapMerged()
        if debug:
            labelmap = ETHECLabelMapMergedSmall()

        combined_graphs = load_combined_graphs(debug)

        if debug:
            image_fc7 = np.load('../database/ETHEC/ETHECSmall_embeddings/train.npy')[()]
            image_fc7.update(np.load('../database/ETHEC/ETHECSmall_embeddings/val.npy')[()])
            image_fc7.update(np.load('../database/ETHEC/ETHECSmall_embeddings/test.npy')[()])
        else:
            image_fc7 = np.load('../database/ETHEC/ETHEC_embeddings/train.npy')[()]
            image_fc7.update(np.load('../database/ETHEC/ETHEC_embeddings/val.npy')[()])
            image_fc7.update(np.load('../database/ETHEC/ETHEC_embeddings/test.npy')[()])

        self.G = combined_graphs['graph']
        self.G_tc = nx.transitive_closure(self.G)
        self.labelmap = labelmap

        if loss_fn == 'ec':
            self.model = Embedder(embedding_dim=dim, labelmap=labelmap, normalize=False, K=3.0)
            self.img_feat_net = FeatNet(output_dim=2, normalize=None,
                                        K=3).to(self.device)
            self.criterion = EuclideanConesWithImagesHypernymLoss(labelmap=labelmap,
                                                                 neg_to_pos_ratio=5,
                                                                 feature_dict=image_fc7,
                                                                 alpha=0.01,
                                                                 pick_per_level=True,
                                                                 use_CNN=False)

        else:
            self.model = Embedder(embedding_dim=dim, labelmap=labelmap, normalize=False)
            self.img_feat_net = FeatNet(output_dim=2, normalize=None,
                                        K=None).to(self.device)
            self.criterion = OrderEmbeddingWithImagesHypernymLoss(labelmap=labelmap,
                                                                 neg_to_pos_ratio=5,
                                                                 feature_dict=image_fc7,
                                                                 alpha=1.0,
                                                                 pick_per_level=True,
                                                                 use_CNN=False)

        self.model = nn.DataParallel(self.model)
        self.img_feat_net = nn.DataParallel(self.img_feat_net)
        self.graph_dict = combined_graphs

        self.load_model(weights_to_load, img_weights_to_load)

        # prepare model
        self.testloader = self.create_loader()
        self.img_to_emb = dict()
        self.pass_samples()

        # run vizualize
        self.vizualize()

    def load_model(self, weights_to_load, img_weights_to_load):
        checkpoint = torch.load(weights_to_load, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.epoch = checkpoint['epoch']
        self.optimal_threshold = checkpoint['optimal_threshold']
        checkpoint = torch.load(img_weights_to_load, map_location=self.device)
        self.img_feat_net.load_state_dict(checkpoint['model_state_dict'])
        print('Using optimal threshold = {}'.format(self.optimal_threshold))
        print('Successfully loaded model and img_feat_net epoch {} from {}'.format(self.epoch, weights_to_load))

    def create_loader(self):
        val_test_data_transforms = transforms.Compose([transforms.ToPILImage(),
                                                       transforms.Resize((224, 224)),
                                                       transforms.ToTensor(),
                                                       ])
        test_set = ETHECHierarchyWithImages(self.graph_dict['G_{}'.format(self.load_split)],
                                            imageless_dataloaders=None,
                                            transform=val_test_data_transforms)

        testloader = torch.utils.data.DataLoader(test_set, collate_fn=my_collate,
                                                 batch_size=10,
                                                 num_workers=4,
                                                 shuffle=False)
        return testloader

    def pass_samples(self):
        self.img_feat_net = self.img_feat_net.to(self.device)
        for index, data_item in enumerate(tqdm(self.testloader)):
            inputs_from, inputs_to, status = data_item['from'], data_item['to'], data_item['status']

            self.get_from_and_to_emb(inputs_to)

    def get_from_and_to_emb(self, to_elem):
        # get embeddings for concepts and images
        # do the same for to embeddings
        image_elem = [elem for ix, elem in enumerate(to_elem) if type(elem) == str]
        image_ix = [ix for ix, elem in enumerate(to_elem) if type(elem) == str]

        if len(image_elem) != 0:
            image_emb = self.img_feat_net(self.criterion.get_img_features(image_elem).to(self.device))
            to_embeddings = torch.zeros((len(to_elem), image_emb.shape[-1])).to(self.device)

        if len(image_elem) != 0:
            to_embeddings[image_ix, :] = image_emb

        to_embeddings = to_embeddings.cpu().detach().numpy()
        for i in range(len(to_elem)):
            self.img_to_emb[to_elem[i]] = to_embeddings[i, :]


    def vizualize(self):
        phase = 'test'
        self.model.eval()

        colors = ['c', 'm', 'y', 'k']
        embeddings_x, embeddings_y, annotation, color_list = {}, {}, {}, {}

        connected_to = {}

        fig, ax = plt.subplots()

        for level_id in range(4): #len(self.labelmap.levels)):
            level_start, level_stop = self.labelmap.level_start[level_id], self.labelmap.level_stop[level_id]
            level_color = colors[level_id]
            for label_ix in range(self.labelmap.levels[level_id]):
                emb_id = label_ix + level_start
                emb = self.model(torch.tensor([emb_id]).to(self.device)).cpu().detach()
                emb = emb[0].numpy()

                embeddings_x[emb_id] = emb[0]
                embeddings_y[emb_id] = emb[1]
                annotation[emb_id] = '{}'.format(getattr(self.labelmap,
                                                      '{}_ix_to_str'.format(self.labelmap.level_names[level_id]))[label_ix]
                                                 )
                color_list[emb_id] = level_color

                connected_to[emb_id] = [v for u, v in list(self.G.edges(emb_id))]

                ax.scatter(emb[0], emb[1], c=level_color, alpha=1)

        for from_node in connected_to:
            for to_node in connected_to[from_node]:
                if to_node in embeddings_x:
                    plt.plot([embeddings_x[from_node], embeddings_x[to_node]], [embeddings_y[from_node], embeddings_y[to_node]],
                             'b-', alpha=0.2)

        for key in self.img_to_emb:
            emb = self.img_to_emb[key]
            ax.scatter(emb[0], emb[1], c=level_color, alpha=0.3)

            from_ix = max([u for u, v in list(self.graph_dict['G_{}'.format(self.load_split)].in_edges(key))])

            if from_ix in embeddings_x:
                plt.plot([embeddings_x[from_ix], emb[0]],
                         [embeddings_y[from_ix], emb[1]],
                         'b-', alpha=0.2)


        plt.show()

# obj = VizualizeGraphRepresentation(debug=True)
obj = VizualizeGraphRepresentationWithImages(debug=True)