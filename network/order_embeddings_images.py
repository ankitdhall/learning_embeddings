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

from network.order_embeddings import OrderEmbedding, OrderEmbeddingLoss, SimpleEuclideanEmbLoss, Embedder, EmbeddingMetrics
from network.inference import Inference


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ImageEmb:
    def __init__(self, path_to_exp='../exp/ethec_resnet50_lr_1e-5_1_1_1_1/',
                 image_dir='/media/ankit/DataPartition/IMAGO_build_test_resized',
                 path_to_db='../database/ETHEC/', debug=False):
        self.path_to_exp = path_to_exp
        self.image_dir = image_dir
        self.path_to_db = path_to_db
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.debug = debug

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
        if self.debug:
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
        else:
            labelmap = ETHECLabelMapMerged()
            train_set = ETHECDBMerged(path_to_json=os.path.join(self.path_to_db, 'train.json'),
                                      path_to_images=self.image_dir,
                                      labelmap=labelmap, transform=val_test_data_transforms)
            val_set = ETHECDBMerged(path_to_json=os.path.join(self.path_to_db, 'val.json'),
                                    path_to_images=self.image_dir,
                                    labelmap=labelmap, transform=val_test_data_transforms)
            test_set = ETHECDBMerged(path_to_json=os.path.join(self.path_to_db, 'test.json'),
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
    def __init__(self, feature_dict, input_dim=2048, output_dim=10):
        """
        Constructor to prepare layers for the embedding.
        """
        super(FeatNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, output_dim)

        self.feature_dict = feature_dict
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        """
        Forward pass through the model.
        """
        # print(x)
        img_feat = None
        for filename in x:
            if img_feat is None:
                if type(filename) == tuple:
                    img_feat = torch.tensor(self.feature_dict[filename[0]]).unsqueeze(0)
                else:
                    img_feat = torch.tensor(self.feature_dict[filename]).unsqueeze(0)
            else:
                if type(filename) == tuple:
                    img_feat = torch.cat((img_feat, torch.tensor(self.feature_dict[filename[0]]).unsqueeze(0)), dim=0)
                else:
                    img_feat = torch.cat((img_feat, torch.tensor(self.feature_dict[filename]).unsqueeze(0)), dim=0)

        x = img_feat.to(self.device)
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


class OrderEmbeddingWithImagesLoss(OrderEmbeddingLoss):
    def __init__(self, labelmap, neg_to_pos_ratio, alpha=1.0):
        OrderEmbeddingLoss.__init__(self, labelmap, neg_to_pos_ratio, alpha)

        self.image_nodes_in_graph = set()
        self.non_image_nodes_in_graph = set()

    def set_graph_tc(self, graph_tc):
        """
        Set G_train_tc transitive closure for the training set as this is used to generate negative edges dynamically.
        :param graph_tc: <networkx.DiGraph> Transitive closure for the training graph to generate corrupted samples
        :return: NA
        """
        self.G_tc = graph_tc
        self.reverse_G = nx.reverse(self.G_tc)
        nodes_in_graph = set(list(self.G_tc))
        self.num_edges = self.G_tc.size()

        self.image_nodes_in_graph = set([node for node in list(nodes_in_graph) if type(node) == str])
        self.non_image_nodes_in_graph = set([node for node in list(nodes_in_graph) if type(node) != str])

    def forward(self, model, img_feat_net, inputs_from, inputs_to, status, phase):
        loss = 0.0
        e_for_u_v_positive_all, e_for_u_v_negative_all = torch.tensor([]), torch.tensor([])
        predicted_from_embeddings_all = torch.tensor([]) # model(inputs_from)
        predicted_to_embeddings_all = torch.tensor([]) # model(inputs_to)

        for batch_id in range(len(inputs_from)):
            predicted_from_embeddings = model(inputs_from[batch_id])
            predicted_to_embeddings = img_feat_net(inputs_to[batch_id])
            predicted_from_embeddings_all = torch.cat((predicted_from_embeddings_all, predicted_from_embeddings))
            predicted_to_embeddings_all = torch.cat((predicted_to_embeddings_all, predicted_to_embeddings))

            if phase != 'train':
                # loss for positive pairs
                positive_indices = (status[batch_id] == 1).nonzero().squeeze(dim=1)
                e_for_u_v_positive = self.positive_pair(predicted_from_embeddings[positive_indices],
                                                        predicted_to_embeddings[positive_indices])
                loss += torch.sum(e_for_u_v_positive)
                e_for_u_v_positive_all = torch.cat((e_for_u_v_positive_all, e_for_u_v_positive))

                # loss for negative pairs
                negative_indices = (status[batch_id] == 0).nonzero().squeeze(dim=1)
                neg_term, e_for_u_v_negative = self.negative_pair(predicted_from_embeddings[negative_indices],
                                                                 predicted_to_embeddings[negative_indices])
                loss += torch.sum(neg_term)
                e_for_u_v_negative_all = torch.cat((e_for_u_v_negative_all, e_for_u_v_negative))

            else:
                # loss for positive pairs
                positive_indices = (status[batch_id] == 1).nonzero().squeeze(dim=1)
                e_for_u_v_positive = self.positive_pair(predicted_from_embeddings[positive_indices],
                                                        predicted_to_embeddings[positive_indices])
                loss += torch.sum(e_for_u_v_positive)
                e_for_u_v_positive_all = torch.cat((e_for_u_v_positive_all, e_for_u_v_positive))

                # loss for negative pairs

                negative_from = torch.zeros((2 * self.neg_to_pos_ratio * inputs_from[batch_id].shape[0]), dtype=torch.long)
                negative_to = [None]*(2 * self.neg_to_pos_ratio * inputs_from[batch_id].shape[0])

                for sample_id in range(inputs_from[batch_id].shape[0]):
                    sample_inputs_from, sample_inputs_to = inputs_from[batch_id][sample_id], inputs_to[batch_id][sample_id]
                    for pass_ix in range(self.neg_to_pos_ratio):

                        list_of_edges_from_ui = [v for u, v in list(self.G_tc.edges(sample_inputs_from.item()))]
                        corrupted_ix = random.choice(list(self.image_nodes_in_graph - set(list_of_edges_from_ui)))
                        negative_from[2 * self.neg_to_pos_ratio * sample_id + pass_ix] = sample_inputs_from
                        negative_to[2 * self.neg_to_pos_ratio * sample_id + pass_ix] = corrupted_ix

                        list_of_edges_to_vi = [v for u, v in list(self.reverse_G.edges(sample_inputs_to))]
                        corrupted_ix = random.choice(list(self.non_image_nodes_in_graph - set(list_of_edges_to_vi)))
                        negative_from[
                            2 * self.neg_to_pos_ratio * sample_id + pass_ix + self.neg_to_pos_ratio] = corrupted_ix
                        negative_to[2 * self.neg_to_pos_ratio * sample_id + pass_ix + self.neg_to_pos_ratio] = sample_inputs_to

                negative_from_embeddings, negative_to_embeddings = model(negative_from), img_feat_net(negative_to)
                neg_term, e_for_u_v_negative = self.negative_pair(negative_from_embeddings, negative_to_embeddings)
                loss += torch.sum(neg_term)
                e_for_u_v_negative_all = torch.cat((e_for_u_v_negative_all, e_for_u_v_negative))

        return predicted_from_embeddings_all, predicted_to_embeddings_all, loss, e_for_u_v_positive_all, e_for_u_v_negative_all


class OrderEmbeddingWithImagesLossvCaption(OrderEmbeddingLoss):
    def __init__(self, labelmap, neg_to_pos_ratio, alpha=1.0):
        OrderEmbeddingLoss.__init__(self, labelmap, neg_to_pos_ratio, alpha)

        self.image_nodes_in_graph = set()
        self.non_image_nodes_in_graph = set()

    def set_graph_tc(self, graph_tc):
        """
        Set G_train_tc transitive closure for the training set as this is used to generate negative edges dynamically.
        :param graph_tc: <networkx.DiGraph> Transitive closure for the training graph to generate corrupted samples
        :return: NA
        """
        self.G_tc = graph_tc
        self.reverse_G = nx.reverse(self.G_tc)
        nodes_in_graph = set(list(self.G_tc))
        self.num_edges = self.G_tc.size()

        self.image_nodes_in_graph = set([node for node in list(nodes_in_graph) if type(node) == str])
        self.non_image_nodes_in_graph = set([node for node in list(nodes_in_graph) if type(node) != str])

    def positive_pair(self, x, y):
        return self.E_operator(x, y)

    def negative_pair(self, x, y):
        return torch.clamp(self.alpha-self.E_operator(x, y), min=0.0), self.E_operator(x, y)

    def get_image_label_loss(self, e_for_u_v_positive, e_for_u_v_negative):
        # print(e_for_u_v_positive.shape, e_for_u_v_negative.shape)
        loss_term = 0.0
        s_for_u_v_positive = -e_for_u_v_positive
        s_for_u_v_negative = -e_for_u_v_negative
        for sample_id in range(s_for_u_v_positive.shape[0]):
            for corrupted_ix in range(self.neg_to_pos_ratio*2):
                # print(self.alpha, s_for_u_v_positive[sample_id], s_for_u_v_negative[sample_id*2*self.neg_to_pos_ratio+corrupted_ix])
                S = self.alpha - s_for_u_v_positive[sample_id] + s_for_u_v_negative[sample_id*2*self.neg_to_pos_ratio+corrupted_ix]
                # print(S)
                S = torch.sum(torch.clamp(S, min=0.0))
                loss_term += S
        return loss_term

    def forward(self, model, img_feat_net, inputs_from, inputs_to, status, phase):
        loss = 0.0
        e_for_u_v_positive_all, e_for_u_v_negative_all = torch.tensor([]), torch.tensor([])
        predicted_from_embeddings_all = torch.tensor([]) # model(inputs_from)
        predicted_to_embeddings_all = torch.tensor([]) # model(inputs_to)

        if phase != 'train':
            predicted_from_embeddings = model(torch.tensor(inputs_from))
            predicted_to_embeddings = img_feat_net(inputs_to)
            predicted_from_embeddings_all = torch.cat((predicted_from_embeddings_all, predicted_from_embeddings))
            predicted_to_embeddings_all = torch.cat((predicted_to_embeddings_all, predicted_to_embeddings))

            # loss for positive pairs
            positive_indices = (torch.tensor(status) == 1).nonzero().squeeze(dim=1)
            e_for_u_v_positive = self.positive_pair(predicted_from_embeddings[positive_indices],
                                                    predicted_to_embeddings[positive_indices])
            # loss += torch.sum(e_for_u_v_positive)
            e_for_u_v_positive_all = torch.cat((e_for_u_v_positive_all, e_for_u_v_positive))

            # loss for negative pairs
            negative_indices = (torch.tensor(status) == 0).nonzero().squeeze(dim=1)
            neg_term, e_for_u_v_negative = self.negative_pair(predicted_from_embeddings[negative_indices],
                                                              predicted_to_embeddings[negative_indices])
            # loss += torch.sum(neg_term)
            e_for_u_v_negative_all = torch.cat((e_for_u_v_negative_all, e_for_u_v_negative))

            # print(e_for_u_v_positive.shape, e_for_u_v_negative.shape)
            # print(e_for_u_v_positive, e_for_u_v_negative)

            loss += self.get_image_label_loss(e_for_u_v_positive, e_for_u_v_negative)

        else:
            for batch_id in range(len(inputs_from)):
                predicted_from_embeddings = model(inputs_from[batch_id])
                predicted_to_embeddings = img_feat_net(inputs_to[batch_id])
                predicted_from_embeddings_all = torch.cat((predicted_from_embeddings_all, predicted_from_embeddings))
                predicted_to_embeddings_all = torch.cat((predicted_to_embeddings_all, predicted_to_embeddings))

                # loss for positive pairs
                positive_indices = (status[batch_id] == 1).nonzero().squeeze(dim=1)
                e_for_u_v_positive = self.positive_pair(predicted_from_embeddings[positive_indices],
                                                        predicted_to_embeddings[positive_indices])
                # loss += torch.sum(e_for_u_v_positive)
                e_for_u_v_positive_all = torch.cat((e_for_u_v_positive_all, e_for_u_v_positive))

                # loss for negative pairs

                negative_from = torch.zeros((2 * self.neg_to_pos_ratio * inputs_from[batch_id].shape[0]), dtype=torch.long)
                negative_to = [None]*(2 * self.neg_to_pos_ratio * inputs_from[batch_id].shape[0])

                for sample_id in range(inputs_from[batch_id].shape[0]):
                    sample_inputs_from, sample_inputs_to = inputs_from[batch_id][sample_id], inputs_to[batch_id][sample_id]
                    for pass_ix in range(self.neg_to_pos_ratio):

                        list_of_edges_from_ui = [v for u, v in list(self.G_tc.edges(sample_inputs_from.item()))]
                        corrupted_ix = random.choice(list(self.image_nodes_in_graph - set(list_of_edges_from_ui)))
                        negative_from[2 * self.neg_to_pos_ratio * sample_id + pass_ix] = sample_inputs_from
                        negative_to[2 * self.neg_to_pos_ratio * sample_id + pass_ix] = corrupted_ix

                        list_of_edges_to_vi = [v for u, v in list(self.reverse_G.edges(sample_inputs_to))]
                        corrupted_ix = random.choice(list(self.non_image_nodes_in_graph - set(list_of_edges_to_vi)))
                        negative_from[
                            2 * self.neg_to_pos_ratio * sample_id + pass_ix + self.neg_to_pos_ratio] = corrupted_ix
                        negative_to[2 * self.neg_to_pos_ratio * sample_id + pass_ix + self.neg_to_pos_ratio] = sample_inputs_to

                negative_from_embeddings, negative_to_embeddings = model(negative_from), img_feat_net(negative_to)
                neg_term, e_for_u_v_negative = self.negative_pair(negative_from_embeddings, negative_to_embeddings)
                # loss += torch.sum(neg_term)
                e_for_u_v_negative_all = torch.cat((e_for_u_v_negative_all, e_for_u_v_negative))

                loss += self.get_image_label_loss(e_for_u_v_positive, e_for_u_v_negative)

        return predicted_from_embeddings_all, predicted_to_embeddings_all, loss, e_for_u_v_positive_all, e_for_u_v_negative_all


class EmbeddingLabelsWithImages:
    def __init__(self, graph_dict, labelmap, criterion, lr,
                 batch_size,
                 experiment_name,
                 embedding_dim,
                 neg_to_pos_ratio,
                 lr_step=[],
                 experiment_dir='../exp/',
                 n_epochs=10,
                 eval_interval=2,
                 feature_extracting=True,
                 use_pretrained=True,
                 load_wt=False,
                 model_name=None,
                 optimizer_method='adam',
                 use_grayscale=False):
        torch.manual_seed(0)

        self.classes = labelmap.classes
        self.n_classes = labelmap.n_classes
        self.levels = labelmap.levels
        self.n_levels = len(self.levels)
        self.level_names = labelmap.level_names
        self.lr = lr
        self.lr_step = lr_step
        self.batch_size = batch_size
        self.feature_extracting = feature_extracting
        self.optimal_thresholds = np.zeros(self.n_classes)
        self.optimizer_method = optimizer_method
        self.labelmap = labelmap
        self.model_name = model_name

        self.best_model_wts = None
        self.best_score = 0.0

        self.epoch = 0
        self.exp_dir = experiment_dir
        self.load_wt = load_wt

        self.criterion = criterion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using device: {}'.format(self.device))
        if torch.cuda.device_count() > 1:
            print("== Using", torch.cuda.device_count(), "GPUs!")
        # self.model = model.to(self.device)
        self.n_epochs = n_epochs
        self.eval_interval = eval_interval

        self.log_dir = os.path.join(self.exp_dir, '{}').format(experiment_name)
        self.path_to_save_model = os.path.join(self.log_dir, 'weights')
        self.make_dir_if_non_existent(self.path_to_save_model)

        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'tensorboard'))

        # embedding specific stuff
        self.graph_dict = graph_dict

        self.optimal_threshold = 1.0
        self.embedding_dim = embedding_dim # 10
        self.neg_to_pos_ratio = neg_to_pos_ratio # 5

        # prepare models (embedding module and image feature extractor)
        self.model = Embedder(embedding_dim=self.embedding_dim, labelmap=labelmap)

        # load precomputed features as look-up table
        image_fc7 = np.load('../database/ETHEC/ETHECSmall_embeddings/train.npy')[()]
        image_fc7.update(np.load('../database/ETHEC/ETHECSmall_embeddings/val.npy')[()])
        image_fc7.update(np.load('../database/ETHEC/ETHECSmall_embeddings/test.npy')[()])
        self.img_feat_net = FeatNet(feature_dict=image_fc7, output_dim=self.embedding_dim)

        # TODO
        # self.G_tc = nx.transitive_closure(self.G)
        self.criterion.set_graph_tc(self.graph_dict['G_train_tc'])
        # self.create_splits()
        #
        # nx.draw_networkx(self.G, arrows=True)
        # plt.show()

        self.create_splits()
        print('Training set has {} samples. Validation set has {} samples. Test set has {} samples'.format(
            len(self.dataloaders['train'].dataset),
            len(self.dataloaders['val'].dataset),
            len(self.dataloaders['test'].dataset)))

        self.prepare_model()
        # self.set_optimizer()

    @staticmethod
    def make_dir_if_non_existent(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def prepare_model(self):
        self.params_to_update = list(self.model.parameters()) + list(self.img_feat_net.parameters())

        # if self.feature_extracting:
        #     self.params_to_update = []
        #     for name, param in self.model.named_parameters():
        #         if param.requires_grad:
        #             self.params_to_update.append(param)
        #             print("Will update: {}".format(name))
        # else:
        #     print("Fine-tuning")

    def create_splits(self):
        random.seed(0)
        train_set = ETHECHierarchyWithImages(self.graph_dict['G_train'], self.graph_dict['G_train_tc'],
                                             has_negative=False)
        val_set = ETHECHierarchyWithImages(self.graph_dict['G_val'], self.graph_dict['G_val_tc'], has_negative=True,
                                           neg_to_pos_ratio=self.neg_to_pos_ratio)
        test_set = ETHECHierarchyWithImages(self.graph_dict['G_test'], self.graph_dict['G_test_tc'], has_negative=True,
                                            neg_to_pos_ratio=self.neg_to_pos_ratio)

        # create dataloaders
        trainloader = torch.utils.data.DataLoader(train_set,
                                                  batch_size=self.batch_size,
                                                  num_workers=16,
                                                  shuffle=True)
        valloader = torch.utils.data.DataLoader(val_set,
                                                batch_size=1,
                                                num_workers=0,
                                                shuffle=True)
        testloader = torch.utils.data.DataLoader(test_set,
                                                 batch_size=1,
                                                 num_workers=0,
                                                 shuffle=True)

        self.dataloaders = {'train': trainloader, 'val': valloader, 'test': testloader}
        self.dataset_length = {'train': len(train_set), 'val': len(val_set), 'test': len(test_set)}

    def run_model(self, optimizer):
        self.optimizer = optimizer
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.lr_step, gamma=0.1)

        if self.load_wt:
            self.find_existing_weights()

        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.best_score = 0.0

        since = time.time()

        for self.epoch in range(self.epoch, self.n_epochs):
            print('=' * 10)
            print('Epoch {}/{}'.format(self.epoch, self.n_epochs - 1))
            print('=' * 10)

            self.pass_samples(phase='train')
            if self.epoch % self.eval_interval == 0:
                self.pass_samples(phase='val')
                self.pass_samples(phase='test')

            scheduler.step()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val score: {:4f}'.format(self.best_score))

        # load best model weights
        self.model.load_state_dict(self.best_model_wts)

        self.writer.close()
        return self.model

    def pass_samples(self, phase, save_to_tensorboard=True):
        if phase == 'train':
            self.model.train()
            self.img_feat_net.train()
        else:
            self.model.eval()
            self.img_feat_net.eval()

        running_loss = 0.0

        predicted_from_embeddings, predicted_to_embeddings = torch.tensor([]), torch.tensor([])
        e_positive, e_negative = torch.tensor([]), torch.tensor([])

        # Iterate over data.
        for index, data_item in enumerate(self.dataloaders[phase]):
            inputs_from, inputs_to, status = data_item['from'], data_item['to'], data_item['status']

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                self.model = self.model.to(self.device)
                self.img_feat_net = self.img_feat_net.to(self.device)

                outputs_from, outputs_to, loss, e_for_u_v_positive, e_for_u_v_negative =\
                    self.criterion(self.model, self.img_feat_net, inputs_from, inputs_to, status, phase)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()

            # statistics
            running_loss += loss.item()

            outputs_from, outputs_to = outputs_from.cpu().detach(), outputs_to.cpu().detach()

            predicted_from_embeddings = torch.cat((predicted_from_embeddings, outputs_from.data))
            predicted_to_embeddings = torch.cat((predicted_to_embeddings, outputs_to.data))
            e_positive = torch.cat((e_positive, e_for_u_v_positive.data))
            e_negative = torch.cat((e_negative, e_for_u_v_negative.data))

        metrics = EmbeddingMetrics(e_positive, e_negative, self.optimal_threshold, phase)

        f1_score, threshold, accuracy = metrics.calculate_metrics()
        if phase == 'val':
            self.optimal_threshold = threshold

        epoch_loss = running_loss / self.dataset_length[phase]

        if save_to_tensorboard:
            self.writer.add_scalar('{}_loss'.format(phase), epoch_loss, self.epoch)
            self.writer.add_scalar('{}_f1_score'.format(phase), f1_score, self.epoch)
            self.writer.add_scalar('{}_accuracy'.format(phase), accuracy, self.epoch)
            self.writer.add_scalar('{}_thresh'.format(phase), self.optimal_threshold, self.epoch)

        print('{} Loss: {:.4f}, F1-score: {:.4f}, Accuracy: {:.4f}'.format(phase, epoch_loss, f1_score, accuracy))

        # deep copy the model
        if phase == 'val':
            if self.epoch % 10 == 0:
                self.save_model(epoch_loss)
            if f1_score >= self.best_score:
                self.best_score = f1_score
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                self.save_model(epoch_loss, filename='best_model')

    def save_model(self, loss, filename=None):
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'optimal_threshold': self.optimal_threshold
        }, os.path.join(self.path_to_save_model, '{}_model.pth'.format(filename if filename else self.epoch)))
        print('Successfully saved model epoch {} to {} as {}_model.pth'.format(self.epoch, self.path_to_save_model,
                                                                               filename if filename else self.epoch))

        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.img_feat_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'optimal_threshold': self.optimal_threshold
        }, os.path.join(self.path_to_save_model, '{}_img_feat_net.pth'.format(filename if filename else self.epoch)))
        print('Successfully saved img feat net epoch {} to {} as {}_img_feat_net.pth'.format(self.epoch,
                                                                                             self.path_to_save_model,
                                                                                             filename if filename else self.epoch))

    def load_model(self, epoch_to_load):
        checkpoint = torch.load(os.path.join(self.path_to_save_model, '{}_model.pth'.format(epoch_to_load)),
                                map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.optimal_threshold = checkpoint['optimal_threshold']
        checkpoint = torch.load(os.path.join(self.path_to_save_model, '{}_img_feat_net.pth'.format(epoch_to_load)),
                                map_location=self.device)
        self.img_feat_net.load_state_dict(checkpoint['model_state_dict'])
        print('Successfully loaded model and img_feat_net epoch {} from {}'.format(self.epoch, self.path_to_save_model))

    def train(self):
        if self.optimizer_method == 'sgd':
            self.run_model(optim.SGD(self.params_to_update, lr=self.lr, momentum=0.9))
        elif self.optimizer_method == 'adam':
            self.run_model(optim.Adam(self.params_to_update, lr=self.lr))
        self.load_best_model()

    def load_best_model(self):
        self.load_model(epoch_to_load='best_model')
        self.pass_samples(phase='test', save_to_tensorboard=False)


def order_embedding_labels_with_images_train_model(arguments):
    if not os.path.exists(os.path.join(arguments.experiment_dir, arguments.experiment_name)):
        os.makedirs(os.path.join(arguments.experiment_dir, arguments.experiment_name))
    args_dict = vars(arguments)
    repo = git.Repo(search_parent_directories=True)
    args_dict['commit_hash'] = repo.head.object.hexsha
    args_dict['branch'] = repo.active_branch.name
    with open(os.path.join(arguments.experiment_dir, arguments.experiment_name, 'config_params.txt'), 'w') as file:
        file.write(json.dumps(args_dict, indent=4))

    print('Config parameters for this run are:\n{}'.format(json.dumps(vars(arguments), indent=4)))

    labelmap = None
    if arguments.merged:
        labelmap = ETHECLabelMapMerged()
    if arguments.debug:
        labelmap = ETHECLabelMapMergedSmall()

    dataloaders = create_imageless_dataloaders(debug=arguments.debug)
    graph_dict = create_combined_graphs(dataloaders, labelmap)

    batch_size = arguments.batch_size
    n_workers = arguments.n_workers

    if arguments.debug:
        print("== Running in DEBUG mode!")

    use_criterion = None
    if arguments.loss == 'order_emb_loss':
        # use_criterion = OrderEmbeddingWithImagesLoss(labelmap=labelmap, neg_to_pos_ratio=arguments.neg_to_pos_ratio)
        use_criterion = OrderEmbeddingWithImagesLossvCaption(labelmap=labelmap, neg_to_pos_ratio=arguments.neg_to_pos_ratio)
    elif arguments.loss == 'euc_emb_loss':
        print('Not implemented!')
        # use_criterion = SimpleEuclideanEmbLoss(labelmap=labelmap)
    else:
        print("== Invalid --loss argument")

    oelwi = EmbeddingLabelsWithImages(graph_dict=graph_dict, labelmap=labelmap,
                                      criterion=use_criterion,
                                      lr=arguments.lr,
                                      batch_size=batch_size,
                                      experiment_name=arguments.experiment_name,  # 'cifar_test_ft_multi',
                                      experiment_dir=arguments.experiment_dir,
                                      embedding_dim=arguments.embedding_dim,
                                      neg_to_pos_ratio=arguments.neg_to_pos_ratio,
                                      eval_interval=arguments.eval_interval,
                                      n_epochs=arguments.n_epochs,
                                      feature_extracting=arguments.freeze_weights,
                                      use_pretrained=True,
                                      load_wt=arguments.resume,
                                      model_name=arguments.model,
                                      optimizer_method=arguments.optimizer_method,
                                      use_grayscale=arguments.use_grayscale,
                                      lr_step=arguments.lr_step)

    if arguments.set_mode == 'train':
        oelwi.train()
    elif arguments.set_mode == 'test':
        print('Not Implemented!')


if __name__ == '__main__':
    generate_emb = True
    if generate_emb:
        ImageEmb().load_generate_and_save()
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--debug", help='Use DEBUG mode.', action='store_true')
        parser.add_argument("--lr", help='Input learning rate.', type=float, default=0.001)
        parser.add_argument("--batch_size", help='Batch size.', type=int, default=8)
        # parser.add_argument("--evaluator", help='Evaluator type.', type=str, default='ML')
        parser.add_argument("--experiment_name", help='Experiment name.', type=str, required=True)
        parser.add_argument("--experiment_dir", help='Experiment directory.', type=str, required=True)
        parser.add_argument("--image_dir", help='Image parent directory.', type=str, required=True)
        parser.add_argument("--n_epochs", help='Number of epochs to run training for.', type=int, required=True)
        parser.add_argument("--n_workers", help='Number of workers.', type=int, default=4)
        parser.add_argument("--eval_interval", help='Evaluate model every N intervals.', type=int, default=1)
        parser.add_argument("--embedding_dim", help='Dimensions of learnt embeddings.', type=int, default=10)
        parser.add_argument("--neg_to_pos_ratio", help='Number of negatives to sample for one positive.', type=int, default=5)
        # parser.add_argument("--prop_of_nb_edges", help='Proportion of non-basic edges to be added to train set.', type=float, default=0.0)
        parser.add_argument("--resume", help='Continue training from last checkpoint.', action='store_true')
        parser.add_argument("--optimizer_method", help='[adam, sgd]', type=str, default='adam')
        parser.add_argument("--merged", help='Use dataset which has genus and species combined.', action='store_true')
        # parser.add_argument("--weight_strategy", help='Use inverse freq or inverse sqrt freq. ["inv", "inv_sqrt"]',
        #                     type=str, default='inv')
        parser.add_argument("--model", help='NN model to use.', type=str, default='alexnet')
        parser.add_argument("--loss",
                            help='Loss function to use. [order_emb_loss, euc_emb_loss]',
                            type=str, required=True)
        parser.add_argument("--use_grayscale", help='Use grayscale images.', action='store_true')
        # parser.add_argument("--class_weights", help='Re-weigh the loss function based on inverse class freq.',
        #                     action='store_true')
        parser.add_argument("--freeze_weights", help='This flag fine tunes only the last layer.', action='store_true')
        parser.add_argument("--set_mode", help='If use training or testing mode (loads best model).', type=str,
                            required=True)
        # parser.add_argument("--level_weights", help='List of weights for each level', nargs=4, default=None, type=float)
        parser.add_argument("--lr_step", help='List of epochs to make multiple lr by 0.1', nargs='*', default=[], type=int)
        args = parser.parse_args()

        order_embedding_labels_with_images_train_model(args)


# debug = True
# labelmap = ETHECLabelMapMerged()
# if debug:
#     labelmap = ETHECLabelMapMergedSmall()
# dataloaders = create_imageless_dataloaders(debug=debug)
# graph_dict = create_combined_graphs(dataloaders, labelmap)
#
# model = Embedder(embedding_dim=10, labelmap=labelmap)
#
# image_fc7 = np.load('../database/ETHEC/ETHECSmall_embeddings/test.npy')[()]
# feat_net = FeatNet(feature_dict=image_fc7)
#
# db_object = ETHECHierarchyWithImages(graph_dict['G_test'], graph_dict['G_test_tc'], has_negative=True, neg_to_pos_ratio=5)
# for ix in range(len(db_object)):
#     inputs_from, inputs_to = db_object[ix]['from'], db_object[ix]['to']
#     for sample_id in range(len(inputs_from)):
#         # print(sample_id, inputs_from, inputs_to)
#         print(model(inputs_from[sample_id]), feat_net(inputs_to[sample_id]))
