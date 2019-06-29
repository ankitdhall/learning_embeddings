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
from network.summarize import Summarize
from network.order_embeddings_images import EmbeddingMetrics

from data.db import ETHECLabelMap, ETHECDB, ETHECDBMerged, ETHECLabelMapMerged, ETHECLabelMapMergedSmall, ETHECDBMergedSmall
from network.loss import MultiLevelCELoss, MultiLabelSMLoss, LastLevelCELoss, MaskedCELoss, HierarchicalSoftmaxLoss

from PIL import Image
import numpy as np
import time
from tqdm import tqdm
import itertools

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


class Embedder(nn.Module):
    def __init__(self, embedding_dim, labelmap, normalize):
        super(Embedder, self).__init__()
        self.labelmap = labelmap
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        if self.normalize == 'max_norm':
            self.embeddings = nn.Embedding(self.labelmap.n_classes, self.embedding_dim, max_norm=1.0)
        else:
            self.embeddings = nn.Embedding(self.labelmap.n_classes, self.embedding_dim)
        print('Embeds {} objects'.format(self.labelmap.n_classes))

    def forward(self, inputs):
        embeds = self.embeddings(inputs)#.view((1, -1))
        if self.normalize == 'unit_norm':
            return F.normalize(embeds, p=2, dim=1)
        else:
            return embeds


class FeatNet(nn.Module):
    """
    Fully connected NN to learn features on top of image features in the joint embedding space.
    """
    def __init__(self, normalize, input_dim=2048, output_dim=10):
        """
        Constructor to prepare layers for the embedding.
        """
        super(FeatNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)

        self.normalize = normalize

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        """
        Forward pass through the model.
        """
        x = F.relu(self.fc1(x))
        if self.normalize == 'unit_norm':
            x = self.fc2(x)
            original_shape = x.shape
            x = x.view(-1, original_shape[-1])
            x = F.normalize(x, p=2, dim=1)
            x = x.view(original_shape)
        elif self.normalize == 'max_norm':
            x = self.fc2(x)
            original_shape = x.shape
            x = x.view(-1, original_shape[-1])
            norm_x = torch.norm(x, p=2, dim=1)
            x[norm_x > 1.0] = F.normalize(x[norm_x > 1.0], p=2, dim=1)
            x = x.view(original_shape)
        else:
            x = self.fc2(x)
        return x


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


def my_collate(data):
    from_data, to_data, status_data = [], [], []
    for data_item in data:
        from_data.append(data_item['from'])
        to_data.append(data_item['to'])
        status_data.append(data_item['status'])
    return {'from': from_data, 'to': to_data, 'status': torch.tensor(status_data)}


class EmbeddingMetrics:
    def __init__(self, e_for_u_v_positive, e_for_u_v_negative, threshold, phase):
        self.e_for_u_v_positive = e_for_u_v_positive.view(-1)
        self.e_for_u_v_negative = e_for_u_v_negative.view(-1)
        self.threshold = threshold
        self.phase = phase

    def calculate_metrics(self):
        if self.phase == 'val':
            possible_thresholds = np.unique(np.concatenate((self.e_for_u_v_positive, self.e_for_u_v_negative), axis=None))
            best_score, best_threshold, best_accuracy = 0.0, 0.0, 0.0
            for t_id in range(possible_thresholds.shape[0]):
                correct_positives = torch.sum(self.e_for_u_v_positive <= possible_thresholds[t_id]).item()
                correct_negatives = torch.sum(self.e_for_u_v_negative > possible_thresholds[t_id]).item()
                accuracy = (correct_positives+correct_negatives)/(self.e_for_u_v_positive.shape[0]+self.e_for_u_v_negative.shape[0])
                precision = correct_positives/(correct_positives+(self.e_for_u_v_negative.shape[0]-correct_negatives))
                recall = correct_positives/self.e_for_u_v_positive.shape[0]
                if precision+recall == 0:
                    print('Encountered NaN for f1!')
                    f1_score = 0.0
                else:
                    f1_score = (2*precision*recall)/(precision+recall)
                if f1_score > best_score:
                    best_accuracy = accuracy
                    best_score = f1_score
                    best_threshold = possible_thresholds[t_id]

            return best_score, best_threshold, best_accuracy

        else:
            correct_positives = torch.sum(self.e_for_u_v_positive <= self.threshold).item()
            correct_negatives = torch.sum(self.e_for_u_v_negative > self.threshold).item()
            accuracy = (correct_positives + correct_negatives) / (
                        self.e_for_u_v_positive.shape[0] + self.e_for_u_v_negative.shape[0])

            if correct_positives + (self.e_for_u_v_negative.shape[0] - correct_negatives) == 0:
                print('Encountered NaN for precision!')
                precision = 0.0
            else:
                precision = correct_positives / (correct_positives + (self.e_for_u_v_negative.shape[0] - correct_negatives))
            recall = correct_positives / self.e_for_u_v_positive.shape[0]
            if precision + recall == 0:
                f1_score = 0.0
                print('Encountered NaN for f1!')
            else:
                f1_score = (2 * precision * recall) / (precision + recall)
            return f1_score, self.threshold, accuracy

def create_combined_graphs(dataloaders, labelmap):
    G = nx.DiGraph()
    for index, data_item in enumerate(dataloaders['train']):
        inputs, labels, level_labels = data_item['image'], data_item['labels'], data_item['level_labels']
        for level_id in range(len(labelmap.levels) - 1):
            for sample_id in range(level_labels.shape[0]):
                G.add_edge(level_labels[sample_id, level_id].item() + labelmap.level_start[level_id],
                           level_labels[sample_id, level_id + 1].item() + labelmap.level_start[level_id + 1])

    print('Graph with labels connected has {} edges, {} nodes'.format(G.size(), len(G.nodes())))

    G_train_tc = copy.deepcopy(G)
    G_train, G_val, G_test = nx.DiGraph(), nx.DiGraph(), nx.DiGraph()

    for split_name, split_graph, split_graph_tc in zip(['train', 'val', 'test'], [G_train, G_val, G_test], [G_train_tc, None, None]):
        dataloader = dataloaders[split_name]
        for index, data_item in enumerate(dataloader):
            level_labels = data_item['level_labels']
            for level_id in range(len(labelmap.levels)):
                for sample_id in range(level_labels.shape[0]):
                    split_graph.add_edge(level_labels[sample_id, level_id].item() + labelmap.level_start[level_id],
                                         data_item['image_filename'][sample_id])
                    if split_name != 'train':
                        continue
                    split_graph_tc.add_edge(level_labels[sample_id, level_id].item() + labelmap.level_start[level_id],
                                            data_item['image_filename'][sample_id])

    print('Graphs with labels (disconnected between themselves) & images: train {}, val {}, test {}'.format(
        G_train.size(), G_val.size(), G_test.size()))

    G_train_skeleton_full = copy.deepcopy(G_train_tc)
    print('Graphs with labels connected + labels & images connected for train has: {} edges'.format(G_train_skeleton_full.size()))

    G_train_tc = nx.transitive_closure(G_train_tc)

    print('Transitive closure of graphs with labels & images: train {}'.format(G_train_tc.size()))

    nx.write_gpickle(G, 'G')
    nx.write_gpickle(G_train, 'G_train')
    nx.write_gpickle(G_val, 'G_val')
    nx.write_gpickle(G_test, 'G_test')
    nx.write_gpickle(G_train_skeleton_full, 'G_train_skeleton_full')
    nx.write_gpickle(G_train_tc, 'G_train_tc')

    return {'graph': G,  # graph with labels only; edges between labels only
            'G_train': G_train, 'G_val': G_val, 'G_test': G_test,  # graph with labels and images; edges between labels and images only
            'G_train_skeleton_full': G_train_skeleton_full, # graph with edge between labels + between labels and images
            'G_train_tc': G_train_tc}  # graph with labels and images; tc(graph with edge between labels; labels and images)


class ETHECHierarchyWithImages(torch.utils.data.Dataset):
    """
    Creates a PyTorch dataset for order-embeddings with images.
    """

    def __init__(self, graph, has_negative=False, neg_to_pos_ratio=1):
        """
        Constructor.
        :param graph: <networkx.DiGraph> Graph to be used.
        """
        self.G = graph
        self.num_edges = self.G.size()
        self.has_negative = has_negative
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.edge_list = [e for e in self.G.edges()]
        self.status = [1]*len(self.edge_list)
        self.negative_from, self.negative_to = None, None

    def __getitem__(self, item):
        """
        Fetch an entry based on index.
        :param item: <int> Index to fetch.
        :return: <dict> Consumable object (see schema.md)
                {'from': <int>, 'to': <int>}
        """
        # if self.has_negative:
        #     from_list, to_list, status, = [torch.tensor(self.edge_list[item][0])], [self.edge_list[item][1]], [1]
        #     for pass_ix in range(self.neg_to_pos_ratio):
        #         from_list.append(self.negative_from[2 * self.neg_to_pos_ratio * item + pass_ix])
        #         to_list.append(self.negative_to[2 * self.neg_to_pos_ratio * item + pass_ix])
        #         status.append(0)
        #         from_list.append(self.negative_from[2 * self.neg_to_pos_ratio * item + pass_ix + self.neg_to_pos_ratio])
        #         to_list.append(self.negative_to[2 * self.neg_to_pos_ratio * item + pass_ix + self.neg_to_pos_ratio])
        #         status.append(0)
        #     return {'from': from_list, 'to': to_list, 'status': status}
        # else:
        return {'from': self.edge_list[item][0], 'to': self.edge_list[item][1], 'status': 1}

    def __len__(self):
        """
        Return number of entries in the database.
        :return: <int> length of database
        """
        return len(self.status)


class OrderEmbeddingWithImagesHypernymLoss(torch.nn.Module):
    def __init__(self, labelmap, neg_to_pos_ratio, feature_dict, alpha):
        print('Using order-embedding loss!')
        torch.nn.Module.__init__(self)
        self.labelmap = labelmap
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.alpha = alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.G_tc = None
        # self.reverse_G = None
        # self.nodes_in_graph = None
        # self.num_edges = None
        self.negative_G = None

        # self.image_nodes_in_graph = set()
        # self.non_image_nodes_in_graph = set()

        self.list_of_edges_from_ui, self.list_of_edges_to_vi = {}, {}

        self.feature_dict = feature_dict

    def get_img_features(self, x):
        retval = None
        unsqueeze_once_more = False
        for sublist_id, sublist in enumerate(x):
            if type(sublist) == str:
                unsqueeze_once_more = True
                if retval is None:
                    img_emb_feat = torch.tensor(self.feature_dict[sublist]).unsqueeze(0)
                    retval = torch.zeros((len(x), img_emb_feat.shape[-1]))
                    retval[sublist_id, :] = img_emb_feat
                else:
                    retval[sublist_id, :] = torch.tensor(self.feature_dict[sublist]).unsqueeze(0)
            else:
                img_feat = None
                for file_id, filename in enumerate(sublist):
                    if img_feat is None:
                        img_emb_feat = torch.tensor(self.feature_dict[filename]).unsqueeze(0)
                        img_feat = torch.zeros((len(sublist), img_emb_feat.shape[-1]))
                        img_feat[file_id, :] = img_emb_feat
                    else:
                        img_feat[file_id, :] = torch.tensor(self.feature_dict[filename]).unsqueeze(0)

                if retval is None:
                    retval = torch.tensor([])
                retval = torch.cat((retval, img_feat.unsqueeze(0)), dim=0)
        if unsqueeze_once_more:
            retval = retval.unsqueeze(0)
        return retval

    def set_negative_graph(self, n_G):
        """
        Get graph to pick negative edges from.
        :param n_G: <networkx.DiGraph> Training graph to generate corrupted samples
        :return: NA
        """
        self.negative_G = n_G

    @staticmethod
    def E_operator(x, y):
        original_shape = x.shape
        x = x.contiguous().view(-1, original_shape[-1])
        y = y.contiguous().view(-1, original_shape[-1])

        return torch.sum(torch.clamp(x - y, min=0.0) ** 2, dim=1).view(original_shape[:-1])

    def positive_pair(self, x, y):
        return self.E_operator(x, y)

    def negative_pair(self, x, y):
        return torch.clamp(self.alpha-self.E_operator(x, y), min=0.0), self.E_operator(x, y)

    def get_image_label_loss(self, e_for_u_v_positive, e_for_u_v_negative):
        S = torch.sum(e_for_u_v_positive) + torch.sum(torch.clamp(self.alpha - e_for_u_v_negative, min=0.0))
        return S

    def forward(self, model, img_feat_net, inputs_from, inputs_to, status, phase):
        loss = 0.0
        e_for_u_v_positive_all, e_for_u_v_negative_all = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)

        if phase != 'train':
            predicted_from_embeddings, predicted_to_embeddings = self.calculate_from_and_to_emb(model,
                                                                                                img_feat_net,
                                                                                                inputs_from,
                                                                                                inputs_to)
            # loss for positive pairs
            e_for_u_v_positive = self.positive_pair(predicted_from_embeddings[:, 0, :],
                                                    predicted_to_embeddings[:, 0, :])

            e_for_u_v_positive_all = torch.cat((e_for_u_v_positive_all, e_for_u_v_positive))

            # loss for negative pairs
            neg_term, e_for_u_v_negative = self.negative_pair(predicted_from_embeddings[:, 1:, :],
                                                              predicted_to_embeddings[:, 1:, :])

            e_for_u_v_negative_all = torch.cat((e_for_u_v_negative_all, e_for_u_v_negative))

            loss += torch.sum(self.get_image_label_loss(e_for_u_v_positive, e_for_u_v_negative))

        else:
            # get embeddings for concepts and images
            predicted_from_embeddings_pos, predicted_to_embeddings_pos = self.calculate_from_and_to_emb(model,
                                                                                                        img_feat_net,
                                                                                                        inputs_from,
                                                                                                        inputs_to)
            e_for_u_v_positive_all = self.positive_pair(predicted_from_embeddings_pos,
                                                        predicted_to_embeddings_pos)

            negative_from = [None] * (2 * self.neg_to_pos_ratio * len(inputs_from))
            negative_to = [None] * (2 * self.neg_to_pos_ratio * len(inputs_to))

            for batch_id in range(len(inputs_from)):
                # loss for negative pairs
                sample_inputs_from, sample_inputs_to = inputs_from[batch_id], inputs_to[batch_id]
                for pass_ix in range(self.neg_to_pos_ratio):

                    if sample_inputs_from not in self.list_of_edges_from_ui:
                        self.list_of_edges_from_ui[sample_inputs_from] = [v for u, v in list(self.negative_G.out_edges(sample_inputs_from))]

                    corrupted_ix = random.choice(self.list_of_edges_from_ui[sample_inputs_from])
                    negative_from[2 * self.neg_to_pos_ratio * batch_id + pass_ix] = sample_inputs_from
                    negative_to[2 * self.neg_to_pos_ratio * batch_id + pass_ix] = corrupted_ix

                    if sample_inputs_to not in self.list_of_edges_to_vi:
                        self.list_of_edges_to_vi[sample_inputs_to] = [u for u, v in list(self.negative_G.in_edges(sample_inputs_to))]

                    corrupted_ix = random.choice(self.list_of_edges_to_vi[sample_inputs_to])
                    negative_from[
                        2 * self.neg_to_pos_ratio * batch_id + pass_ix + self.neg_to_pos_ratio] = corrupted_ix
                    negative_to[2 * self.neg_to_pos_ratio * batch_id + pass_ix + self.neg_to_pos_ratio] = sample_inputs_to

            # get embeddings for concepts and images
            negative_from_embeddings, negative_to_embeddings = self.calculate_from_and_to_emb(model, img_feat_net, negative_from, negative_to)

            neg_term, e_for_u_v_negative_all = self.negative_pair(negative_from_embeddings, negative_to_embeddings)
            e_for_u_v_negative_all = e_for_u_v_negative_all.view(len(inputs_from), 2 * self.neg_to_pos_ratio, -1)

            loss += torch.sum(self.get_image_label_loss(torch.squeeze(e_for_u_v_positive_all), torch.squeeze(e_for_u_v_negative_all)))

        return loss, e_for_u_v_positive_all, e_for_u_v_negative_all

    def calculate_from_and_to_emb(self, model, img_feat_net, from_elem, to_elem):
        # get embeddings for concepts and images
        image_elem = [elem for ix, elem in enumerate(from_elem) if type(elem) == str]
        image_ix = [ix for ix, elem in enumerate(from_elem) if type(elem) == str]
        non_image_elem = [elem for ix, elem in enumerate(from_elem) if type(elem) == int]
        non_image_ix = [ix for ix, elem in enumerate(from_elem) if type(elem) == int]

        if len(image_elem) != 0:
            image_emb = img_feat_net(self.get_img_features(image_elem).to(self.device))
            from_embeddings = torch.zeros((len(from_elem), image_emb.shape[-1])).to(self.device)
        if len(non_image_elem) != 0:
            non_image_emb = model(torch.tensor(non_image_elem, dtype=torch.long).to(self.device))
            from_embeddings = torch.zeros((len(from_elem), non_image_emb.shape[-1])).to(self.device)

        if len(image_elem) != 0:
            from_embeddings[image_ix, :] = image_emb
        if len(non_image_elem) != 0:
            from_embeddings[non_image_ix, :] = non_image_emb

        # do the same for to embeddings
        image_elem = [elem for ix, elem in enumerate(to_elem) if type(elem) == str]
        image_ix = [ix for ix, elem in enumerate(to_elem) if type(elem) == str]
        non_image_elem = [elem for ix, elem in enumerate(to_elem) if type(elem) == int]
        non_image_ix = [ix for ix, elem in enumerate(to_elem) if type(elem) == int]

        if len(image_elem) != 0:
            image_emb = img_feat_net(self.get_img_features(image_elem).to(self.device))
            to_embeddings = torch.zeros((len(to_elem), image_emb.shape[-1])).to(self.device)
        if len(non_image_elem) != 0:
            non_image_emb = model(torch.tensor(non_image_elem, dtype=torch.long).to(self.device))
            to_embeddings = torch.zeros((len(to_elem), non_image_emb.shape[-1])).to(self.device)

        if len(image_elem) != 0:
            to_embeddings[image_ix, :] = image_emb
        if len(non_image_elem) != 0:
            to_embeddings[non_image_ix, :] = non_image_emb

        return from_embeddings, to_embeddings


class JointEmbeddings:
    def __init__(self, graph_dict, labelmap, criterion, lr,
                 batch_size,
                 experiment_name,
                 embedding_dim,
                 neg_to_pos_ratio,
                 image_fc7,
                 normalize,
                 alpha,
                 has_fixed_alpha,
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

        self.n_epochs = n_epochs
        self.eval_interval = eval_interval

        self.log_dir = os.path.join(self.exp_dir, '{}').format(experiment_name)
        self.path_to_save_model = os.path.join(self.log_dir, 'weights')
        self.make_dir_if_non_existent(self.path_to_save_model)

        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'tensorboard'))

        # embedding specific stuff
        self.graph_dict = graph_dict

        self.optimal_threshold = 0
        self.alpha = alpha
        self.embedding_dim = embedding_dim # 10
        self.neg_to_pos_ratio = neg_to_pos_ratio # 5
        self.normalize = None

        # prepare models (embedding module and image feature extractor)
        self.model = Embedder(embedding_dim=self.embedding_dim, labelmap=labelmap, normalize=self.normalize).to \
            (self.device)
        # load precomputed features as look-up table
        self.img_feat_net = FeatNet(output_dim=self.embedding_dim, normalize=self.normalize).to(self.device)

        full_G = nx.complete_graph(len(list(self.graph_dict['G_train_tc'].nodes())), create_using=nx.DiGraph())
        full_G = nx.relabel_nodes(full_G, dict(enumerate(list(self.graph_dict['G_train_tc'].nodes()))), copy=True)

        neg_G = copy.deepcopy(full_G)
        print('Neg_G has {} edges before removing tc edges'.format(neg_G.size()))
        neg_G.remove_edges_from(self.graph_dict['G_train_tc'].edges())

        print('Train graph has {} nodes'.format(len(list(self.graph_dict['G_train_tc'].nodes()))))
        print('Full graph has {} edges, transitive closure has {} (={}), Negatives graph has {}'.format(full_G.size(),
                                                                                                  self.graph_dict['G_train_tc'].size(), len(self.graph_dict['G_train_tc'].edges()),
                                                                                                  neg_G.size()
                                                                                                  ))

        # set this graph to use for generating corrupt pairs on the fly
        # so this graph should correspond to the graph: fully connected graph - tc graph
        # for the train set
        self.criterion.set_negative_graph(neg_G)
        #
        # nx.draw_networkx(self.G, arrows=True)
        # plt.show()

        self.create_splits()
        print('Training set has {} positive edges. Validation set has {} positive edges. Test set has {} positive edges'.format(
            len(self.dataloaders['train'].dataset),
            len(self.dataloaders['val'].dataset),
            len(self.dataloaders['test'].dataset)))

        self.prepare_model()

        self.model = nn.DataParallel(self.model)
        self.img_feat_net = nn.DataParallel(self.img_feat_net)

    @staticmethod
    def make_dir_if_non_existent(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def prepare_model(self):
        self.params_to_update = list(self.model.parameters()) + list(self.img_feat_net.parameters())

    def create_splits(self):
        random.seed(0)
        train_set = ETHECHierarchyWithImages(self.graph_dict['G_train_skeleton_full'])
        val_set = ETHECHierarchyWithImages(self.graph_dict['G_val'])
        test_set = ETHECHierarchyWithImages(self.graph_dict['G_test'])

        # create dataloaders
        trainloader = torch.utils.data.DataLoader(train_set,
                                                  batch_size=self.batch_size,
                                                  num_workers=16, collate_fn=my_collate,
                                                  shuffle=True)
        valloader = torch.utils.data.DataLoader(val_set,
                                                batch_size=self.batch_size, collate_fn=my_collate,
                                                num_workers=16,
                                                shuffle=False)
        testloader = torch.utils.data.DataLoader(test_set, collate_fn=my_collate,
                                                 batch_size=self.batch_size,
                                                 num_workers=16,
                                                 shuffle=False)

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

            epoch_start_time = time.time()
            self.pass_samples(phase='train')
            if self.epoch % self.eval_interval == 0:
                self.pass_samples(phase='val')
                self.pass_samples(phase='test')

            scheduler.step()

            epoch_time = time.time() - epoch_start_time
            print('Epoch time {:.0f}m {:.0f}s'.format(epoch_time // 60, epoch_time % 60))

            self.writer.add_scalar('epoch_time', epoch_time, self.epoch)

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

            running_loss = 0.0

            e_positive, e_negative = torch.tensor([]), torch.tensor([])

            index = 0

            # Iterate over data.
            for index, data_item in enumerate(tqdm(self.dataloaders[phase])):
                inputs_from, inputs_to, status = data_item['from'], data_item['to'], data_item['status']

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    self.model = self.model.to(self.device)
                    self.img_feat_net = self.img_feat_net.to(self.device)

                    loss, e_for_u_v_positive, e_for_u_v_negative = \
                        self.criterion(self.model, self.img_feat_net, inputs_from, inputs_to, status, phase)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                # statistics
                running_loss += loss.item()

                e_positive = torch.cat((e_positive, e_for_u_v_positive.detach().cpu().data))
                e_negative = torch.cat((e_negative, e_for_u_v_negative.detach().cpu().data))

            # metrics = EmbeddingMetrics(e_positive, e_negative, self.optimal_threshold, phase)
            # f1_score, threshold, accuracy = metrics.calculate_metrics()
            classification_metrics = self.calculate_classification_metrics(phase)

            epoch_loss = running_loss / ((index + 1) * self.batch_size * self.neg_to_pos_ratio * 2)
            if save_to_tensorboard:
                self.writer.add_scalar('{}_loss'.format(phase), epoch_loss, self.epoch)
                self.writer.add_scalar('{}_thresh'.format(phase), self.optimal_threshold, self.epoch)
            print('train loss: {}'.format(epoch_loss))

        else:
            self.model.eval()
            self.img_feat_net.eval()

            classification_metrics = self.calculate_classification_metrics(phase)

            if phase == 'val':
                # self.optimal_threshold = threshold
                # deep copy the model
                if self.epoch % 10 == 0:
                    self.save_model(-9999.0)
                if classification_metrics['m-f1'] >= self.best_score:
                    self.best_score = classification_metrics['m-f1']
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())
                    self.save_model(-9999.0, filename='best_model')

            if phase == 'test':
                reconstruction_f1, reconstruction_threshold, reconstruction_accuracy = self.check_graph_embedding()
                print('Reconstruction task: F1: {:.4f},  Accuracy: {:.4f}, Threshold: {:.4f}'.format(reconstruction_f1,
                                                                                          reconstruction_accuracy,
                                                                                          reconstruction_threshold))

            if save_to_tensorboard:
                self.writer.add_scalar('{}_thresh'.format(phase), self.optimal_threshold, self.epoch)
                if phase == 'test':
                    self.writer.add_scalar('reconstruction_thresh', reconstruction_threshold, self.epoch)
                    self.writer.add_scalar('reconstruction_f1', reconstruction_f1, self.epoch)
                    self.writer.add_scalar('reconstruction_accuracy', reconstruction_accuracy, self.epoch)

        path_to_save_summary = os.path.join(self.log_dir, 'stats',
                                            ('best_' if not save_to_tensorboard else '') + phase + str(self.epoch))
        self.make_dir_if_non_existent(path_to_save_summary)
        self.summarizer = Summarize(path_to_save_summary)

        # add classification metrics
        self.summarizer.make_heading('Classification Summary - Epoch {} {}'.format(self.epoch, phase), 1)
        level_wise_y_labels, level_wise_x_labels, level_wise_data = [], [], []
        for metric_name in classification_metrics:
            if type(classification_metrics[metric_name]) != dict:
                if save_to_tensorboard:
                    self.writer.add_scalar('{}_classification_{}'.format(phase, metric_name),
                                           classification_metrics[metric_name], self.epoch)
                self.summarizer.make_text(
                    text='{}_classification_{}: {}'.format(phase, metric_name, classification_metrics[metric_name]),
                    bullet=False)
            else:
                for level_id in range(len(self.labelmap.levels)):
                    level_wise_y_labels.append(self.labelmap.level_names[level_id])
                    level_row = []
                    level_wise_x_labels = []
                    for key in classification_metrics['level_metrics'][level_id]:
                        if save_to_tensorboard:
                            self.writer.add_scalar('{}_classification_level_{}_{}'.format(phase, level_id, key),
                                                   classification_metrics['level_metrics'][level_id][key], self.epoch)

                        level_row.append(classification_metrics['level_metrics'][level_id][key])
                        level_wise_x_labels.append(key)
                    level_wise_data.append(level_row)

        self.summarizer.make_heading('Level-wise Classification Summary - Epoch {} {}'.format(self.epoch, phase), 1)
        self.summarizer.make_table(data=level_wise_data, x_labels=level_wise_x_labels, y_labels=level_wise_y_labels)

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

    def calculate_classification_metrics(self, phase, k=[1, 3, 5]):
        calculated_metrics = {}

        G = self.graph_dict['G_{}'.format(phase)]
        G_rev = nx.reverse(G)
        nodes_in_graph = list(G)
        images_in_graph = [node for node in nodes_in_graph if type(node) == str]
        labels_in_graph = [node for node in nodes_in_graph if type(node) == int]
        labels_in_graph.sort()

        predictions = {'tp': {}, 'fp': {}, 'tn': {}, 'fn': {}, 'precision': {}, 'recall': {}, 'f1': {}, 'accuracy': {}}
        hit_at_k = {k_val: {} for k_val in k}
        for label_ix in labels_in_graph:
            predictions['tp'][label_ix] = 0
            predictions['fp'][label_ix] = 0
            predictions['tn'][label_ix] = 0
            predictions['fn'][label_ix] = 0
            for k_val in k:
                hit_at_k[k_val][label_ix] = 0

        images_to_ix = {image_name: ix for ix, image_name in enumerate(images_in_graph)}
        img_rep = torch.zeros((len(images_in_graph), self.embedding_dim)).to(self.device)
        for ix in range(0, len(images_in_graph), 10):
            img_rep[ix:min(ix+10, len(images_in_graph)-1), :] = self.img_feat_net(self.criterion.get_img_features(images_in_graph[ix:min(ix+10, len(images_in_graph)-1)]).to(self.device))
        img_rep = img_rep.cpu().detach().unsqueeze(0)

        label_rep = torch.zeros((len(labels_in_graph), self.embedding_dim)).to(self.device)
        for ix in range(0, len(labels_in_graph), 10):
            label_rep[ix:min(ix + 10, len(labels_in_graph) - 1), :] = self.model(
                torch.tensor(labels_in_graph[ix:min(ix + 10, len(labels_in_graph) - 1)], dtype=torch.long).to(
                    self.device))
        label_rep = label_rep.cpu().detach().unsqueeze(0)

        for image_name in images_in_graph:
            img_ix = images_to_ix[image_name]
            img_emb = img_rep[:, img_ix, :]

            image_is_a_member_of = [v for u, v in list(G_rev.edges(image_name))]
            image_is_a_member_of.sort()

            img_emb = img_emb.repeat(1, label_rep.shape[1]).view(-1, label_rep.shape[1], img_emb.shape[1])

            e = self.criterion.E_operator(label_rep, img_emb)

            for level_id in range(len(self.labelmap.levels)):
                # print(self.labelmap.level_start[level_id], self.labelmap.level_stop[level_id])
                values, indices = torch.topk(
                    e[0, self.labelmap.level_start[level_id]:self.labelmap.level_stop[level_id]],
                    k=max(k), largest=False)

                # add offset to get the correct indices
                indices = indices + self.labelmap.level_start[level_id]

                for k_val in k:
                    if image_is_a_member_of[level_id] in indices[:k_val]:
                        hit_at_k[k_val][image_is_a_member_of[level_id]] += 1

                if image_is_a_member_of[level_id] == indices[0].item():
                    predictions['tp'][image_is_a_member_of[level_id]] += 1
                    for not_a_mem_of_ix in range(self.labelmap.level_start[level_id],
                                                 self.labelmap.level_stop[level_id]):
                        if not_a_mem_of_ix != image_is_a_member_of[level_id]:
                            predictions['tn'][not_a_mem_of_ix] += 1
                else:
                    predictions['fp'][indices[0].item()] += 1
                    predictions['fn'][image_is_a_member_of[level_id]] += 1

        # aggregate stats for each label
        total_cmat = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
        overall_hit_at_k = {k_val: 0 for k_val in k}

        for label_ix in labels_in_graph:
            for k_val in k:
                overall_hit_at_k[k_val] += hit_at_k[k_val][label_ix]

            for perf_met in ['tp', 'fp', 'tn', 'fn']:
                total_cmat[perf_met] += predictions[perf_met][label_ix]

            predictions['accuracy'][label_ix] = (predictions['tp'][label_ix] + predictions['tn'][label_ix]) / (
                    predictions['tp'][label_ix] +
                    predictions['tn'][label_ix] +
                    predictions['fp'][label_ix] +
                    predictions['fn'][label_ix])
            if predictions['tp'][label_ix] + predictions['fp'][label_ix] == 0.0:
                predictions['precision'][label_ix] = 0.0
            else:
                predictions['precision'][label_ix] = predictions['tp'][label_ix] / (
                        predictions['tp'][label_ix] + predictions['fp'][label_ix])

            if predictions['tp'][label_ix] + predictions['fn'][label_ix] == 0.0:
                predictions['recall'][label_ix] = 0.0
            else:
                predictions['recall'][label_ix] = predictions['tp'][label_ix] / (
                        predictions['tp'][label_ix] + predictions['fn'][label_ix])

            if predictions['precision'][label_ix] + predictions['recall'][label_ix] == 0:
                predictions['f1'][label_ix] = 0.0
            else:
                predictions['f1'][label_ix] = (2 * predictions['precision'][label_ix] * predictions['recall'][
                    label_ix]) / (predictions['precision'][label_ix] + predictions['recall'][label_ix])

        # calculate micro metrics
        accuracy = (total_cmat['tp'] + total_cmat['tn']) / (
                    total_cmat['tp'] + total_cmat['tn'] + total_cmat['fp'] + total_cmat['fn'])
        precision = total_cmat['tp'] / (total_cmat['tp'] + total_cmat['fp'])
        recall = total_cmat['tp'] / (total_cmat['tp'] + total_cmat['fn'])
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = (2 * precision * recall) / (precision + recall)

        calculated_metrics['accuracy'] = accuracy
        calculated_metrics['m-precision'] = precision
        calculated_metrics['m-recall'] = recall
        calculated_metrics['m-f1'] = f1_score

        # calculate hit@k
        for k_val in k:
            overall_hit_at_k[k_val] = overall_hit_at_k[k_val] / (len(self.labelmap.levels) * len(images_in_graph))
            calculated_metrics['hit@{}'.format(k_val)] = overall_hit_at_k[k_val]

        # calculate macro metrics
        macro_precision, macro_recall, macro_f1 = 0.0, 0.0, 0.0
        for label_ix in labels_in_graph:
            macro_precision += predictions['precision'][label_ix]
            macro_recall += predictions['recall'][label_ix]
            macro_f1 += predictions['f1'][label_ix]

        macro_precision /= len(labels_in_graph)
        macro_recall /= len(labels_in_graph)
        macro_f1 /= len(labels_in_graph)

        calculated_metrics['M-precision'] = macro_precision
        calculated_metrics['M-recall'] = macro_recall
        calculated_metrics['M-f1'] = macro_f1

        calculated_metrics['level_metrics'] = {}

        # print('='*30, 'Level metrics', '='*30)
        for level_id in range(len(self.labelmap.levels)):
            calculated_metrics['level_metrics'][level_id] = {}

            start, stop = self.labelmap.level_start[level_id], self.labelmap.level_stop[level_id]
            tp, tn, fp, fn, level_macro_f1 = 0, 0, 0, 0, 0.0
            level_hit_at_k = {k_val: 0 for k_val in k}
            for label_ix in range(start, stop):
                tp += predictions['tp'][label_ix]
                tn += predictions['tn'][label_ix]
                fp += predictions['fp'][label_ix]
                fn += predictions['fn'][label_ix]
                level_macro_f1 += predictions['f1'][label_ix]
                for k_val in k:
                    level_hit_at_k[k_val] += hit_at_k[k_val][label_ix]

            # calculate hit@k for each level
            hit_at_k_string = ''
            for k_val in k:
                level_hit_at_k[k_val] /= len(images_in_graph)
                hit_at_k_string += 'Overall Hit@{}: {:.4f}, '.format(k_val, level_hit_at_k[k_val])
                calculated_metrics['level_metrics'][level_id]['hit@{}'.format(k_val)] = level_hit_at_k[k_val]
            # print(hit_at_k_string)

            # calculate micro metrics
            level_macro_f1 /= (stop - start + 1)
            level_accuracy = (tp + tn) / (tp + tn + fp + fn)
            level_precision = tp / (tp + fp)
            level_recall = tp / (tp + fn)
            if level_precision + level_recall == 0:
                level_f1_score = 0.0
            else:
                level_f1_score = (2 * level_precision * level_recall) / (level_precision + level_recall)

            calculated_metrics['level_metrics'][level_id]['m-precision'] = level_precision
            calculated_metrics['level_metrics'][level_id]['m-recall'] = level_recall
            calculated_metrics['level_metrics'][level_id]['m-f1'] = level_f1_score
            calculated_metrics['level_metrics'][level_id]['M-f1'] = level_macro_f1
            calculated_metrics['level_metrics'][level_id]['accuracy'] = level_accuracy
            # print('Level {} Accuracy: {:.4f}, m-Precision: {:.4f}, m-Recall: {:.4f}, m-F1: {:.4f}, M-F1: {:.4f}'.format(level_id, level_accuracy, level_precision, level_recall, level_f1_score, level_macro_f1))

        print('=' * 30, '{} - Classification metrics'.format(phase), '=' * 30)
        print('m-F1: {:.4f} Accuracy: {:.4f}'.format(f1_score, accuracy))
        # print('m-Precision: {:.4f}, m-Recall: {:.4f}, m-F1: {:.4f}'.format(precision, recall, f1_score))
        # print('M-Precision: {:.4f}, M-Recall: {:.4f}, M-F1: {:.4f}'.format(macro_precision, macro_recall, macro_f1))
        hit_at_k_string = ''
        for k_val in k:
            hit_at_k_string += 'Overall Hit@{}: {:.4f}, '.format(k_val, overall_hit_at_k[k_val])
        # print(hit_at_k_string)

        # print('=' * 70)

        return calculated_metrics

    def check_graph_embedding(self):
        edges_in_G = self.graph_dict['graph'].edges()
        n_nodes_in_G = len(self.graph_dict['graph'].nodes())
        nodes_in_G = [i for i in range(n_nodes_in_G)]

        label_embeddings = torch.zeros((len(nodes_in_G), self.embedding_dim)).to(self.device)
        for ix in range(0, len(nodes_in_G), 10):
            label_embeddings[ix:min(ix + 10, len(nodes_in_G) - 1), :] = self.model(
                torch.tensor(nodes_in_G[ix:min(ix + 10, len(nodes_in_G) - 1)], dtype=torch.long).to(
                    self.device))
        label_embeddings = label_embeddings.detach().cpu()

        positive_e = torch.zeros(len(edges_in_G))
        for ix, edge in enumerate(edges_in_G):
            u, v = edge
            positive_e[ix] = self.criterion.E_operator(label_embeddings[u, :], label_embeddings[v, :])

        # make negative graph
        full_G = nx.complete_graph(len(list(self.graph_dict['graph'].nodes())), create_using=nx.DiGraph())
        full_G = nx.relabel_nodes(full_G, dict(enumerate(list(self.graph_dict['graph'].nodes()))), copy=True)

        neg_G = copy.deepcopy(full_G)
        neg_G.remove_edges_from(self.graph_dict['graph'].edges())
        negative_e = torch.zeros(neg_G.size())
        for ix, edge in enumerate(neg_G.edges()):
            u, v = edge
            negative_e[ix] = self.criterion.E_operator(label_embeddings[u, :], label_embeddings[v, :])

        possible_thresholds = np.unique(np.concatenate((positive_e.detach().cpu().numpy(),
                                                        negative_e.detach().cpu().numpy()), axis=None))
        best_score, best_threshold, best_accuracy = 0.0, 0.0, 0.0
        for t_id in range(possible_thresholds.shape[0]):
            correct_positives = torch.sum(positive_e <= possible_thresholds[t_id]).item()
            correct_negatives = torch.sum(negative_e > possible_thresholds[t_id]).item()
            accuracy = (correct_positives + correct_negatives) / (
                        positive_e.shape[0] + negative_e.shape[0])
            precision = correct_positives / (
                        correct_positives + (negative_e.shape[0] - correct_negatives))
            recall = correct_positives / positive_e.shape[0]
            if precision + recall == 0:
                f1_score = 0.0
            else:
                f1_score = (2 * precision * recall) / (precision + recall)
            if f1_score > best_score:
                best_accuracy = accuracy
                best_score = f1_score
                best_threshold = possible_thresholds[t_id]

        print('Checking graph reconstruction: +ve edges {}, -ve edges {}'.format(len(edges_in_G), neg_G.size()))
        return best_score, best_threshold, best_accuracy


def load_combined_graphs(debug):
    print('Reading graphs from disk!')
    if debug:
        path_to_folder = '../database/ETHEC/ETHECSmall_embeddings/graphs'
    else:
        path_to_folder = '../database/ETHEC/ETHEC_embeddings/graphs'

    G = nx.read_gpickle(os.path.join(path_to_folder, 'G'))
    G_train = nx.read_gpickle(os.path.join(path_to_folder, 'G_train'))
    G_val = nx.read_gpickle(os.path.join(path_to_folder, 'G_val'))
    G_test = nx.read_gpickle(os.path.join(path_to_folder, 'G_test'))
    G_train_skeleton_full = nx.read_gpickle(os.path.join(path_to_folder, 'G_train_skeleton_full'))
    G_train_tc = nx.read_gpickle(os.path.join(path_to_folder, 'G_train_tc'))

    print('Graph with labels connected has {} edges, {} nodes'.format(G.size(), len(G.nodes())))
    print('Graphs with labels (disconnected between themselves) & images: train {}, val {}, test {}'.format(
        G_train.size(), G_val.size(), G_test.size()))

    print('Graphs with labels connected + labels & images connected for train has: {} edges'.format(
        G_train_skeleton_full.size()))

    print('Transitive closure of graphs with labels & images: train {}'.format(G_train_tc.size()))

    return {'graph': G,  # graph with labels only; edges between labels only
            'G_train': G_train, 'G_val': G_val, 'G_test': G_test,
            # graph with labels and images; edges between labels and images only
            'G_train_skeleton_full': G_train_skeleton_full,
            # graph with edge between labels + between labels and images
            'G_train_tc': G_train_tc}  # graph with labels and images; tc(graph with edge between labels; labels and images)


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

    alpha = arguments.alpha

    labelmap = None
    if arguments.merged:
        labelmap = ETHECLabelMapMerged()
    if arguments.debug:
        labelmap = ETHECLabelMapMergedSmall()

    dataloaders = create_imageless_dataloaders(debug=arguments.debug)
    if arguments.load_G_from_disk:
        graph_dict = load_combined_graphs(debug=arguments.debug)
    else:
        graph_dict = create_combined_graphs(dataloaders, labelmap)

    batch_size = arguments.batch_size
    n_workers = arguments.n_workers

    if arguments.debug:
        print("== Running in DEBUG mode!")

    if arguments.debug:
        image_fc7 = np.load('../database/ETHEC/ETHECSmall_embeddings/train.npy')[()]
        image_fc7.update(np.load('../database/ETHEC/ETHECSmall_embeddings/val.npy')[()])
        image_fc7.update(np.load('../database/ETHEC/ETHECSmall_embeddings/test.npy')[()])
    else:
        image_fc7 = np.load('../database/ETHEC/ETHEC_embeddings/train.npy')[()]
        image_fc7.update(np.load('../database/ETHEC/ETHEC_embeddings/val.npy')[()])
        image_fc7.update(np.load('../database/ETHEC/ETHEC_embeddings/test.npy')[()])

    use_criterion = None
    if arguments.loss == 'order_emb_loss':
        # use_criterion = OrderEmbeddingWithImagesLoss(labelmap=labelmap, neg_to_pos_ratio=arguments.neg_to_pos_ratio, alpha=alpha)
        use_criterion = OrderEmbeddingWithImagesHypernymLoss(labelmap=labelmap,
                                                             neg_to_pos_ratio=arguments.neg_to_pos_ratio,
                                                             feature_dict=image_fc7,
                                                             alpha=alpha)
    elif arguments.loss == 'euc_emb_loss':
        print('Not implemented!')
        # use_criterion = SimpleEuclideanEmbLoss(labelmap=labelmap)
    else:
        print("== Invalid --loss argument")

    oelwi = JointEmbeddings(graph_dict=graph_dict, labelmap=labelmap,
                            criterion=use_criterion,
                            lr=arguments.lr,
                            batch_size=batch_size,
                            experiment_name=arguments.experiment_name,  # 'cifar_test_ft_multi',
                            experiment_dir=arguments.experiment_dir,
                            image_fc7=image_fc7,
                            alpha=alpha,
                            has_fixed_alpha=arguments.has_fixed_alpha,
                            normalize=None,
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
    generate_emb = False
    if generate_emb:
        # ImageEmb().load_generate_and_save()
        ValidateGraphRepresentation()
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--debug", help='Use DEBUG mode.', action='store_true')
        parser.add_argument("--lr", help='Input learning rate.', type=float, default=0.01)
        parser.add_argument("--alpha", help='Margin for the loss.', type=float, default=0.05)
        parser.add_argument("--batch_size", help='Batch size.', type=int, default=8)
        parser.add_argument("--load_G_from_disk", help='If set, then loads precomputed graphs from disk.',
                            action='store_true')
        parser.add_argument("--has_fixed_alpha", help='If alpha should be constant else tuned on val set.',
                            action='store_true')
        # parser.add_argument("--evaluator", help='Evaluator type.', type=str, default='ML')
        parser.add_argument("--experiment_name", help='Experiment name.', type=str, required=True)
        parser.add_argument("--experiment_dir", help='Experiment directory.', type=str, required=True)
        parser.add_argument("--image_dir", help='Image parent directory.', type=str, required=True)
        parser.add_argument("--n_epochs", help='Number of epochs to run training for.', type=int, required=True)
        parser.add_argument("--n_workers", help='Number of workers.', type=int, default=4)
        parser.add_argument("--eval_interval", help='Evaluate model every N intervals.', type=int, default=1)
        parser.add_argument("--embedding_dim", help='Dimensions of learnt embeddings.', type=int, default=10)
        parser.add_argument("--neg_to_pos_ratio", help='Number of negatives to sample for one positive.', type=int,
                            default=5)
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
        parser.add_argument("--lr_step", help='List of epochs to make multiple lr by 0.1', nargs='*', default=[],
                            type=int)
        args = parser.parse_args()

        order_embedding_labels_with_images_train_model(args)
