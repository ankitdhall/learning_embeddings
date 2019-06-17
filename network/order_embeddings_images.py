from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

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

from network.order_embeddings import OrderEmbedding, OrderEmbeddingLoss
from network.inference import Inference


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ImageEmb:
    def __init__(self, path_to_exp='../exp/ethec_resnet50_lr_1e-5_1_1_1_1/',
                 image_dir='/media/ankit/DataPartition/IMAGO_build_test_resized',
                 path_to_db='../database/ETHEC/'):
        self.path_to_exp = path_to_exp
        self.image_dir = image_dir
        self.path_to_db = path_to_db
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def load_model(self):
        """
        Load the CNN model to generate embeddings for images.
        :return: NA
        """
        inf_obj = Inference(path_to_exp=self.path_to_exp, image_dir=self.image_dir, mode=None, perform_inference=False)
        self.model = inf_obj.get_model()

    def calculate_save_embeddings(self):
        """
        Generates embeddings using self.model and saves them to disk to be used for downstream tasks.
        :return: NA
        """
        input_size = 224
        val_test_data_transforms = transforms.Compose([transforms.ToPILImage(),
                                                       transforms.Resize((input_size, input_size)),
                                                       transforms.ToTensor(),
                                                       ])
        labelmap = ETHECLabelMapMergedSmall()
        train_set = ETHECDBMergedSmall(path_to_json=os.path.join(self.path_to_db, 'train.json'),
                                       path_to_images=self.image_dir,
                                       labelmap=labelmap, transform=val_test_data_transforms)
        val_set = ETHECDBMergedSmall(path_to_json=os.path.join(self.path_to_db, 'val.json'),
                                     path_to_images=self.image_dir,
                                     labelmap=labelmap, transform=val_test_data_transforms)
        test_set = ETHECDBMergedSmall(path_to_json=os.path.join(self.path_to_db, 'test.json'),
                                      path_to_images=self.image_dir,
                                      labelmap=labelmap, transform=val_test_data_transforms)
        trainloader = torch.utils.data.DataLoader(train_set,
                                                 batch_size=1,
                                                 shuffle=False, num_workers=0)
        valloader = torch.utils.data.DataLoader(val_set,
                                                 batch_size=1,
                                                 shuffle=False, num_workers=0)
        testloader = torch.utils.data.DataLoader(test_set,
                                                 batch_size=1,
                                                 shuffle=False, num_workers=0)
        self.model.module.fc = Identity()

        path_to_save_emb = os.path.join(self.path_to_db, 'embeddings')
        if not os.path.exists(path_to_save_emb):
            os.makedirs(path_to_save_emb)

        for loader, loader_name in zip([trainloader, valloader, testloader], ['train', 'val', 'test']):
            embeddings = {}
            print('{} items in {} loader'.format(len(loader), loader_name))
            for index, data_item in enumerate(tqdm(loader)):
                outputs = self.model(data_item['image']).detach()
                embeddings[data_item['image_filename'][0]] = outputs[0].numpy().tolist()
            np.save(os.path.join(path_to_save_emb, '{}.npy'.format(loader_name)), embeddings)

    def load_generate_and_save(self):
        """Load the model, generate embeddings and save them to disk."""
        self.load_model()
        self.calculate_save_embeddings()


class FeatNet(nn.Module):
    """
    Fully connected NN to learn features on top of image features in the joint embedding space.
    """
    def __init__(self, input_dim=2048, output_dim=10):
        """
        Constructor to prepare layers for the embedding.
        """
        super(FeatNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        """
        Forward pass through the model.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.abs(self.fc3(x))
        return x



class ETHECHierarchyWithImages(torch.utils.data.Dataset):
    """
    Creates a PyTorch dataset for order-embeddings with images.
    """

    def __init__(self, graph, graph_tc, has_negative, neg_to_pos_ratio=1):
        """
        Constructor.
        :param graph: <networkx.DiGraph> Graph to be used.
        """
        self.G = graph
        self.G_tc = graph_tc
        self.num_edges = self.G.size()
        self.has_negative = has_negative
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.edge_list = [e for e in self.G.edges()]
        self.status = [1]*len(self.edge_list)
        self.negative_from, self.negative_to = None, None
        if self.has_negative:
            self.create_negative_pairs()

    def __getitem__(self, item):
        """
        Fetch an entry based on index.
        :param item: <int> Index to fetch.
        :return: <dict> Consumable object (see schema.md)
                {'from': <int>, 'to': <int>}
        """
        if self.has_negative:
            from_list, to_list, status, = [torch.tensor(self.edge_list[item][0])], [self.edge_list[item][1]], [1]
            for pass_ix in range(self.neg_to_pos_ratio):
                from_list.append(self.negative_from[2 * self.neg_to_pos_ratio * item + pass_ix])
                to_list.append(self.negative_to[2 * self.neg_to_pos_ratio * item + pass_ix])
                status.append(0)
                from_list.append(self.negative_from[2 * self.neg_to_pos_ratio * item + pass_ix + self.neg_to_pos_ratio])
                to_list.append(self.negative_to[2 * self.neg_to_pos_ratio * item + pass_ix + self.neg_to_pos_ratio])
                status.append(0)
            return {'from': from_list, 'to': to_list, 'status': status}
        else:
            return {'from': [self.edge_list[item][0]], 'to': [self.edge_list[item][1]], 'status': [1]}

    def __len__(self):
        """
        Return number of entries in the database.
        :return: <int> length of database
        """
        return len(self.status)

    def create_negative_pairs(self):
        random.seed(0)
        reverse_G = nx.reverse(self.G_tc)
        nodes_in_graph = set(list(self.G_tc))

        # find nodes that to make corrupt (label, image) pairs in both directions
        image_nodes_in_graph = set([node for node in list(nodes_in_graph) if type(node) == str])
        non_image_nodes_in_graph = set([node for node in list(nodes_in_graph) if type(node) != str])

        negative_from = torch.zeros((2*self.neg_to_pos_ratio*self.num_edges), dtype=torch.long)
        # negative_to = torch.zeros((2*self.neg_to_pos_ratio*self.num_edges), dtype=torch.long)
        negative_to = [None]*(2*self.neg_to_pos_ratio*self.num_edges)

        for sample_id in range(self.num_edges):
            for pass_ix in range(self.neg_to_pos_ratio):
                inputs_from, inputs_to, status = self.edge_list[sample_id][0], self.edge_list[sample_id][1], self.status[sample_id]
                if status != 1:
                    print('Status is NOT 1!')

                list_of_edges_from_ui = [v for u, v in list(self.G_tc.edges(inputs_from))]
                # corrupting the image while keeping the label
                # for such a case corrupted nodes can only correspond to images
                corrupted_ix = random.choice(list(image_nodes_in_graph - set(list_of_edges_from_ui)))
                negative_from[2*self.neg_to_pos_ratio*sample_id+pass_ix] = inputs_from
                negative_to[2*self.neg_to_pos_ratio*sample_id+pass_ix] = corrupted_ix

                # self.edge_list.append((inputs_from, corrupted_ix))
                # self.status.append(0)

                list_of_edges_to_vi = [v for u, v in list(reverse_G.edges(inputs_to))]
                corrupted_ix = random.choice(list(non_image_nodes_in_graph - set(list_of_edges_to_vi)))
                negative_from[2 * self.neg_to_pos_ratio * sample_id + pass_ix + self.neg_to_pos_ratio] = corrupted_ix
                negative_to[2 * self.neg_to_pos_ratio * sample_id + pass_ix + self.neg_to_pos_ratio] = inputs_to

                # self.edge_list.append((corrupted_ix, inputs_to))
                # self.status.append(0)

        self.negative_from, self.negative_to = negative_from, negative_to


def create_combined_graphs(dataloaders, labelmap):
    G = nx.DiGraph()
    for index, data_item in enumerate(dataloaders['train']):
        inputs, labels, level_labels = data_item['image'], data_item['labels'], data_item['level_labels']
        for level_id in range(len(labelmap.levels) - 1):
            for sample_id in range(level_labels.shape[0]):
                G.add_edge(level_labels[sample_id, level_id].item() + labelmap.level_start[level_id],
                           level_labels[sample_id, level_id + 1].item() + labelmap.level_start[level_id + 1])

    print('Graph with labels connected has {} edges'.format(G.size()))

    G_train_tc, G_val_tc, G_test_tc = copy.deepcopy(G), copy.deepcopy(G), copy.deepcopy(G)
    G_train, G_val, G_test = nx.DiGraph(), nx.DiGraph(), nx.DiGraph()

    for split_name, split_graph, split_graph_tc in zip(['train', 'val', 'test'], [G_train, G_val, G_test], [G_train_tc, G_val_tc, G_test_tc]):
        dataloader = dataloaders[split_name]
        for index, data_item in enumerate(dataloader):
            level_labels = data_item['level_labels']
            for level_id in range(len(labelmap.levels)):
                for sample_id in range(level_labels.shape[0]):
                    split_graph.add_edge(level_labels[sample_id, level_id].item() + labelmap.level_start[level_id],
                                         data_item['image_filename'][sample_id])
                    split_graph_tc.add_edge(level_labels[sample_id, level_id].item() + labelmap.level_start[level_id],
                                            data_item['image_filename'][sample_id])

    print('Graphs with labels (disconnected between themselves) & images: train {}, val {}, test {}'.format(
        G_train.size(), G_val.size(), G_test.size()))

    G_train_tc = nx.transitive_closure(G_train_tc)
    G_val_tc = nx.transitive_closure(G_val_tc)
    G_test_tc = nx.transitive_closure(G_test_tc)

    print('Transitive closure of graphs with labels & images: train {}, val {}, test {}'.format(
        G_train_tc.size(), G_val_tc.size(), G_test_tc.size()))

    return {'graph': G, 'G_train': G_train, 'G_val': G_val, 'G_test': G_test,
            'G_train_tc': G_train_tc, 'G_val_tc': G_val_tc, 'G_test_tc': G_test_tc}


def create_imageless_dataloaders(debug):
    image_dir = None
    input_size = 224
    labelmap = ETHECLabelMapMerged()
    if debug:
        labelmap = ETHECLabelMapMergedSmall()

    train_data_transforms = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize((input_size, input_size)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                ])
    val_test_data_transforms = transforms.Compose([transforms.ToPILImage(),
                                                   transforms.Resize((input_size, input_size)),
                                                   transforms.ToTensor(),
                                                   ])
    if not debug:
        print("== Running in DEBUG mode!")
        train_set = ETHECDBMerged(path_to_json='../database/ETHEC/train.json',
                                  path_to_images=image_dir,
                                  labelmap=labelmap, transform=train_data_transforms, with_images=False)
        val_set = ETHECDBMerged(path_to_json='../database/ETHEC/val.json',
                                path_to_images=image_dir,
                                labelmap=labelmap, transform=val_test_data_transforms, with_images=False)
        test_set = ETHECDBMerged(path_to_json='../database/ETHEC/test.json',
                                 path_to_images=image_dir,
                                 labelmap=labelmap, transform=val_test_data_transforms, with_images=False)
    else:
        train_set = ETHECDBMergedSmall(path_to_json='../database/ETHEC/train.json',
                                       path_to_images=image_dir,
                                       labelmap=labelmap, transform=train_data_transforms, with_images=False)
        val_set = ETHECDBMergedSmall(path_to_json='../database/ETHEC/val.json',
                                     path_to_images=image_dir,
                                     labelmap=labelmap, transform=val_test_data_transforms, with_images=False)
        test_set = ETHECDBMergedSmall(path_to_json='../database/ETHEC/test.json',
                                      path_to_images=image_dir,
                                      labelmap=labelmap, transform=val_test_data_transforms, with_images=False)

    print('Dataset images: train {}, val {}, test {}'.format(len(train_set), len(val_set), len(test_set)))
    trainloader = torch.utils.data.DataLoader(train_set,
                                              batch_size=4,
                                              num_workers=1,
                                              shuffle=False,
                                              sampler=None)
    valloader = torch.utils.data.DataLoader(val_set,
                                            batch_size=1,
                                            shuffle=False, num_workers=1)
    testloader = torch.utils.data.DataLoader(test_set,
                                             batch_size=1,
                                             shuffle=False, num_workers=1)

    data_loaders = {'train': trainloader, 'val': valloader, 'test': testloader}
    return data_loaders


debug = True
labelmap = ETHECLabelMapMerged()
if debug:
    labelmap = ETHECLabelMapMergedSmall()
dataloaders = create_imageless_dataloaders(debug=debug)
graph_dict = create_combined_graphs(dataloaders, labelmap)

db_object = ETHECHierarchyWithImages(graph_dict['G_test'], graph_dict['G_test_tc'], has_negative=True, neg_to_pos_ratio=5)
for ix in range(len(db_object)):
    print(db_object[ix])

