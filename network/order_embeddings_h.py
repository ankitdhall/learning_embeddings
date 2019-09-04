from __future__ import print_function
from __future__ import division
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
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

import copy
import argparse
import json
import git
from tqdm import tqdm

import torch
from torch import nn
from data.db import ETHECLabelMap, ETHECLabelMapMergedSmall

from network.finetuner import CIFAR10
import numpy as np
import multiprocessing
import random
random.seed(0)

import networkx as nx
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

import torch.nn.functional as F


def my_collate(data):
    from_data, to_data, status_data, original_from, original_to = [], [], [], [], []
    for data_item in data:
        from_data.extend(data_item['from'])
        to_data.extend(data_item['to'])
        status_data.extend(data_item['status'])
        # original_from.append(data_item['original_from'])
        # original_to.append(data_item['original_to'])
    return {'from': from_data, 'to': to_data, 'status': torch.tensor(status_data)}

class ETHECHierarchy(torch.utils.data.Dataset):
    """
    Creates a PyTorch dataset for order-embeddings, without images.
    """

    def __init__(self, graph, graph_tc, labelmap, has_negative, neg_to_pos_ratio=1, pick_per_level=False):
        """
        Constructor.
        :param graph: <networkx.DiGraph> Graph to be used.
        """
        self.G = graph
        self.G_tc = graph_tc
        self.labelmap = labelmap

        self.num_edges = graph.size()
        self.has_negative = has_negative
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.pick_per_level = pick_per_level

        self.edge_list = [e for e in graph.edges()]
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
            from_list, to_list, status, = [self.edge_list[item][0]], [self.edge_list[item][1]], [1]
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

    def sample_negative_edge(self, u=None, v=None, level_id=None):
        if level_id is not None:
            level_id = level_id % (len(self.labelmap.level_names)+1)
        if u is not None and v is None:
            choose_from = np.where(self.negative_G[self.mapping_from_node_to_ix[u], :] == 1)[0]
        elif u is None and v is not None:
            choose_from = np.where(self.negative_G[:, self.mapping_from_node_to_ix[v]] == 1)[0]
        else:
            print('Error! Both (u, v) given or neither (u, v) given!')

        if self.pick_per_level:
            if level_id < len(self.labelmap.levels):
                level_start, level_stop = self.labelmap.level_start[level_id], self.labelmap.level_stop[level_id]
                choose_from = choose_from[np.where(np.logical_and(choose_from >= level_start, choose_from < level_stop))].tolist()
            else:
                choose_from = choose_from.tolist()
        else:
            choose_from = choose_from.tolist()
        corrupted_ix = random.choice(choose_from)
        return corrupted_ix

    def create_negative_pairs(self):
        random.seed(0)
        # create negative graph
        mapping_ix_to_node = {}
        img_label = len(self.G_tc.nodes())
        for node in list(self.G_tc.nodes()):
            if type(node) == int:
                mapping_ix_to_node[node] = node
            else:
                mapping_ix_to_node[img_label] = node
                img_label += 1

        mapping_node_to_ix = {mapping_ix_to_node[k]: k for k in mapping_ix_to_node}

        n_nodes = len(list(self.G_tc.nodes()))

        A = np.ones((n_nodes, n_nodes), dtype=np.bool)

        for u, v in list(self.G_tc.edges()):
            # remove edges that are in G_train_tc
            A[mapping_node_to_ix[u], mapping_node_to_ix[v]] = 0
        np.fill_diagonal(A, 0)
        self.negative_G = A
        self.mapping_from_ix_to_node = mapping_ix_to_node
        self.mapping_from_node_to_ix = mapping_node_to_ix

        negative_from = [None] * (2 * self.neg_to_pos_ratio * self.num_edges)
        negative_to = [None] * (2 * self.neg_to_pos_ratio * self.num_edges)

        for sample_id in range(self.num_edges):
            # loss for negative pairs
            sample_inputs_from, sample_inputs_to, status = self.edge_list[sample_id][0], self.edge_list[sample_id][1], self.status[
                sample_id]
            for pass_ix in range(self.neg_to_pos_ratio):
                corrupted_ix = self.sample_negative_edge(u=sample_inputs_from, v=None, level_id=pass_ix)
                negative_from[2 * self.neg_to_pos_ratio * sample_id + pass_ix] = sample_inputs_from
                negative_to[2 * self.neg_to_pos_ratio * sample_id + pass_ix] = self.mapping_from_ix_to_node[
                    corrupted_ix]

                corrupted_ix = self.sample_negative_edge(u=None, v=sample_inputs_to, level_id=pass_ix)
                negative_from[
                    2 * self.neg_to_pos_ratio * sample_id + pass_ix + self.neg_to_pos_ratio] = \
                    self.mapping_from_ix_to_node[corrupted_ix]
                negative_to[
                    2 * self.neg_to_pos_ratio * sample_id + pass_ix + self.neg_to_pos_ratio] = sample_inputs_to

        self.negative_from, self.negative_to = negative_from, negative_to


class Embedder(nn.Module):
    def __init__(self, embedding_dim, labelmap, K=None):
        super(Embedder, self).__init__()
        self.labelmap = labelmap
        self.embedding_dim = embedding_dim
        self.normalize = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.K = K
        self.inner_radius = 2 * self.K / (1 + np.sqrt(1 + 4 * self.K * self.K))

        if self.normalize == 'max_norm':
            self.embeddings = nn.Embedding(self.labelmap.n_classes, self.embedding_dim, max_norm=1.0)
        else:
            self.embeddings = nn.Embedding(self.labelmap.n_classes, self.embedding_dim)
        print('Embeds {} objects'.format(self.labelmap.n_classes))
        self.epsilon = 1e-5

        with torch.no_grad():
            norm = torch.norm(self.embeddings.weight.data, dim=1, keepdim=True).repeat(1, self.embedding_dim)
            # add to the inner radius a U[0, 0.05] for randomizing norm of embeddings
            new_norm = self.inner_radius + torch.rand((self.embeddings.weight.data.shape[0]))*0.05
            new_norm = torch.unsqueeze(new_norm, 1).repeat(1, self.embedding_dim)
            self.embeddings.weight.data = new_norm*self.embeddings.weight.data/norm

    def forward(self, inputs):
        embeds = self.embeddings(inputs)#.view((1, -1))
        embeds = embeds + 1e-15

        if self.normalize == 'unit_norm':
            return F.normalize(embeds, p=2, dim=1)
        else:
            if self.K:
                return self.soft_clip(embeds)
            else:
                return embeds

    def soft_clip(self, x):
        original_shape = x.shape
        x = x.view(-1, original_shape[-1])
        # direction = F.normalize(x, dim=1)
        # norm = torch.norm(x, dim=1, keepdim=True)
        # x = direction * (norm + self.inner_radius)

        with torch.no_grad():
            norm = torch.norm(x, dim=1, keepdim=True).repeat(1, self.embedding_dim)
            x[norm <= self.inner_radius] = x[norm <= self.inner_radius] / norm[norm <= self.inner_radius] * self.inner_radius
            x[norm >= 1.0] = x[norm >= 1.0]/norm[norm >= 1.0]*(1.0-self.epsilon)
        return x.view(original_shape)


class EmbeddingMetricsOld:
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
            else:
                f1_score = (2 * precision * recall) / (precision + recall)
            return f1_score, self.threshold, accuracy


class EmbeddingMetrics:
    def __init__(self, e_for_u_v_positive, e_for_u_v_negative, threshold, phase, n_proc=4):
        self.e_for_u_v_positive = e_for_u_v_positive.view(-1)
        self.e_for_u_v_negative = e_for_u_v_negative.view(-1)
        self.threshold = threshold
        self.phase = phase
        self.n_proc = n_proc

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

        return f1_score, threshold, accuracy, precision, recall, correct_positives, correct_negatives

    def calculate_metrics(self):
        if self.phase == 'val':
            possible_thresholds = np.unique(
                np.concatenate((self.e_for_u_v_positive, self.e_for_u_v_negative), axis=None))

            F = np.zeros((possible_thresholds.shape[0], 7))
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
            return f1_score, self.threshold, accuracy, precision, recall, correct_positives, correct_negatives

class OrderEmbedding:
    def __init__(self, data_loaders, labelmap, criterion, lr, batch_size, evaluator, experiment_name, embedding_dim,
                 neg_to_pos_ratio, alpha, proportion_of_nb_edges_in_train, lr_step=[], pick_per_level=False,
                 experiment_dir='../exp/', n_epochs=10, eval_interval=2, feature_extracting=True, load_wt=False,
                 optimizer_method='adam', lr_decay=1.0, random_seed=0, load_cosine_emb=None):
        torch.manual_seed(random_seed)

        self.epoch = 0
        self.exp_dir = experiment_dir
        self.load_wt = load_wt
        self.pick_per_level = pick_per_level

        self.eval = evaluator
        self.criterion = criterion
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using device: {}'.format(self.device))
        if torch.cuda.device_count() > 1:
            print("== Using", torch.cuda.device_count(), "GPUs!")
        self.n_epochs = n_epochs
        self.eval_interval = eval_interval

        self.log_dir = os.path.join(self.exp_dir, '{}').format(experiment_name)
        self.path_to_save_model = os.path.join(self.log_dir, 'weights')
        if not os.path.exists(self.path_to_save_model):
            os.makedirs(self.path_to_save_model)

        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'tensorboard'))

        self.classes = labelmap.classes
        self.n_classes = labelmap.n_classes
        self.levels = labelmap.levels
        self.n_levels = len(self.levels)
        self.level_names = labelmap.level_names
        self.lr = lr
        self.batch_size = batch_size
        self.feature_extracting = feature_extracting
        self.optimizer_method = optimizer_method
        self.lr_step = lr_step

        self.optimal_threshold = 0
        self.embedding_dim = embedding_dim
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.proportion_of_nb_edges_in_train = proportion_of_nb_edges_in_train

        if isinstance(criterion, EucConesLoss):
            self.model = Embedder(embedding_dim=self.embedding_dim, labelmap=labelmap, K=self.criterion.K)
            if load_cosine_emb:
                self.load_inverted_cosine_emb(load_cosine_emb)
        else:
            self.model = Embedder(embedding_dim=self.embedding_dim, labelmap=labelmap)
        self.model = self.model.to(self.device)
        self.model = nn.DataParallel(self.model)
        self.labelmap = labelmap

        self.G, self.G_train, self.G_val, self.G_test = nx.DiGraph(), nx.DiGraph(), nx.DiGraph(), nx.DiGraph()
        for index, data_item in enumerate(data_loaders['train']):
            inputs, labels, level_labels = data_item['image'], data_item['labels'], data_item['level_labels']
            for level_id in range(len(self.labelmap.levels)-1):
                for sample_id in range(level_labels.shape[0]):
                    self.G.add_edge(level_labels[sample_id, level_id].item()+self.labelmap.level_start[level_id],
                                    level_labels[sample_id, level_id+1].item()+self.labelmap.level_start[level_id+1])

        self.G_tc = nx.transitive_closure(self.G)
        self.create_splits()

        self.criterion.set_negative_graph(self.G_train_neg, self.mapping_ix_to_node, self.mapping_node_to_ix)
        self.criterion.set_graph_tc(self.G_tc)

        self.lr_decay = lr_decay
        self.check_graph_embedding_neg_graph = None

        self.check_reconstr_every = 10
        self.save_model_every = 10
        self.reconstruction_f1, self.reconstruction_threshold, self.reconstruction_accuracy, self.reconstruction_prec, self.reconstruction_recall = 0.0, 0.0, 0.0, 0.0, 0.0
        self.n_proc = 32 if torch.cuda.device_count() > 0 else 4

    def prepare_model(self):
        self.params_to_update = self.model.parameters()

        if self.feature_extracting:
            self.params_to_update = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.params_to_update.append(param)
                    print("Will update: {}".format(name))
        else:
            print("Fine-tuning")

    def create_splits(self):
        random.seed(0)
        # prepare train graph
        # bare-bones graph without transitive edges
        self.G_train = copy.deepcopy(self.G)

        # create negative graph
        mapping_ix_to_node = {}
        img_label = len(self.G_tc.nodes())
        for node in list(self.G_tc.nodes()):
            if type(node) == int:
                mapping_ix_to_node[node] = node
            else:
                mapping_ix_to_node[img_label] = node
                img_label += 1

        mapping_node_to_ix = {mapping_ix_to_node[k]: k for k in mapping_ix_to_node}

        n_nodes = len(list(self.G_tc.nodes()))

        A = np.ones((n_nodes, n_nodes), dtype=np.bool)

        for u, v in list(self.G_tc.edges()):
            # remove edges that are in G_train_tc
            A[mapping_node_to_ix[u], mapping_node_to_ix[v]] = 0
        np.fill_diagonal(A, 0)
        self.G_train_neg = A
        self.mapping_ix_to_node = mapping_ix_to_node
        self.mapping_node_to_ix = mapping_node_to_ix

        # prepare test and val sub-graphs
        print('Has {} edges in original graph'.format(self.G.size()))
        print('Has {} edges in transitive closure'.format(self.G_tc.size()))

        copy_of_G_tc = copy.deepcopy(self.G_tc)
        edge_in_g = [e for e in self.G.edges]
        for edge_e in edge_in_g:
            copy_of_G_tc.remove_edge(edge_e[0], edge_e[1])

        total_number_of_edges = self.G_tc.size()
        total_number_of_nb_edges = copy_of_G_tc.size()
        n_edges_to_add_to_train = int(total_number_of_nb_edges*self.proportion_of_nb_edges_in_train)
        edges_for_test_val = int(0.05*total_number_of_nb_edges)
        print('Has {} non-basic edges. {} for val and test.'.format(total_number_of_nb_edges, edges_for_test_val))
        non_basic_edges = self.G_tc.size()-self.G.size()

        # create val graph
        total_number_of_nb_edges = copy_of_G_tc.size()
        remove_edges = random.sample(range(total_number_of_nb_edges), k=edges_for_test_val)
        edges_in_tc = [e for e in copy_of_G_tc.edges()]
        for edge_ix in remove_edges:
            self.G_val.add_edge(edges_in_tc[edge_ix][0], edges_in_tc[edge_ix][1])
        for edge_ix in remove_edges:
            copy_of_G_tc.remove_edge(edges_in_tc[edge_ix][0], edges_in_tc[edge_ix][1])

        # create test graph
        total_number_of_nb_edges = copy_of_G_tc.size()
        remove_edges = random.sample(range(total_number_of_nb_edges), k=edges_for_test_val)
        edges_in_tc = [e for e in copy_of_G_tc.edges()]
        for edge_ix in remove_edges:
            self.G_test.add_edge(edges_in_tc[edge_ix][0], edges_in_tc[edge_ix][1])
        for edge_ix in remove_edges:
            copy_of_G_tc.remove_edge(edges_in_tc[edge_ix][0], edges_in_tc[edge_ix][1])

        print('Edges in train: {}, val: {}, test: {}'.format(self.G_train.size(), self.G_val.size(), self.G_test.size()))

        # if need to add non-basic edges, add them to G_train
        total_number_of_nb_edges = copy_of_G_tc.size()
        remove_edges = random.sample(range(total_number_of_nb_edges), k=n_edges_to_add_to_train)
        edges_in_tc = [e for e in copy_of_G_tc.edges()]
        for edge_ix in remove_edges:
            self.G_train.add_edge(edges_in_tc[edge_ix][0], edges_in_tc[edge_ix][1])
        for edge_ix in remove_edges:
            copy_of_G_tc.remove_edge(edges_in_tc[edge_ix][0], edges_in_tc[edge_ix][1])

        print('Added {:.2f}% of non-basic edges = {}'.format(self.proportion_of_nb_edges_in_train, n_edges_to_add_to_train))
        print('Edges in train: {}, val: {}, test: {}'.format(self.G_train.size(), self.G_val.size(), self.G_test.size()))
        print('Edges in transitive closure: {}'.format(self.G_tc.size()))

        # create dataloaders
        train_set = ETHECHierarchy(self.G_train, self.G_tc, labelmap=self.labelmap, has_negative=False,
                                   pick_per_level=self.pick_per_level)
        val_set = ETHECHierarchy(self.G_val, self.G_tc, labelmap=self.labelmap, has_negative=True,
                                 neg_to_pos_ratio=self.neg_to_pos_ratio, pick_per_level=self.pick_per_level)
        test_set = ETHECHierarchy(self.G_test, self.G_tc, labelmap=self.labelmap, has_negative=True,
                                  neg_to_pos_ratio=self.neg_to_pos_ratio, pick_per_level=self.pick_per_level)
        trainloader = torch.utils.data.DataLoader(train_set,
                                                  batch_size=self.batch_size, collate_fn=my_collate,
                                                  num_workers=16,
                                                  shuffle=True)
        valloader = torch.utils.data.DataLoader(val_set,
                                                batch_size=1, collate_fn=my_collate,
                                                num_workers=0,
                                                shuffle=True)
        testloader = torch.utils.data.DataLoader(test_set,
                                                batch_size=1, collate_fn=my_collate,
                                                num_workers=0,
                                                shuffle=True)
        self.dataloaders = {'train': trainloader, 'val': valloader, 'test': testloader}
        self.graphs = {'train': self.G_train, 'val': self.G_val, 'test': self.G_test}
        self.dataset_length = {phase: len(self.dataloaders[phase].dataset) for phase in ['train', 'val', 'test']}

    def calculate_best(self, threshold):
        correct_positives = torch.sum(self.positive_e <= threshold).item()
        correct_negatives = torch.sum(self.negative_e > threshold).item()
        accuracy = (correct_positives + correct_negatives) / (self.positive_e.shape[0] + self.negative_e.shape[0])
        precision = correct_positives / (correct_positives + (self.negative_e.shape[0] - correct_negatives))
        recall = correct_positives / self.positive_e.shape[0]
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = (2 * precision * recall) / (precision + recall)

        return f1_score, threshold, accuracy, precision, recall

    def check_graph_embedding(self):
        if self.check_graph_embedding_neg_graph is None:
            start_time = time.time()
            # make negative graph
            n_nodes = len(list(self.G_tc.nodes()))

            A = np.ones((n_nodes, n_nodes), dtype=np.bool)

            for u, v in list(self.G_tc.edges()):
                # remove edges that are in G_train_tc
                A[u, v] = 0
            np.fill_diagonal(A, 0)
            self.check_graph_embedding_neg_graph = A

            self.edges_in_G = self.G_tc.edges()
            self.n_nodes_in_G = len(self.G_tc.nodes())
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

        label_embeddings = torch.zeros((len(self.nodes_in_G), self.embedding_dim)).to(self.device)
        for ix in range(0, len(self.nodes_in_G), 100):
            label_embeddings[ix:min(ix + 100, len(self.nodes_in_G) - 1), :] = self.model(
                torch.tensor(self.nodes_in_G[ix:min(ix + 100, len(self.nodes_in_G) - 1)], dtype=torch.long).to(
                    self.device))
        label_embeddings = label_embeddings.detach().cpu()

        positive_e = self.criterion.E_operator(label_embeddings[self.pos_u_list, :], label_embeddings[self.pos_v_list, :])
        negative_e = self.criterion.E_operator(label_embeddings[self.neg_u_list, :], label_embeddings[self.neg_v_list, :])

        metrics = EmbeddingMetrics(positive_e.detach().cpu(), negative_e.detach().cpu(), 0.0,
                                   'val', n_proc=self.n_proc)
        best_score, best_threshold, best_accuracy, best_precision, best_recall, c_pos, c_neg = metrics.calculate_metrics()

        print('Checking graph reconstruction: +ve edges {}, -ve edges {}'.format(len(self.edges_in_G),
                                                                                 np.sum(self.check_graph_embedding_neg_graph)))
        return best_score, best_threshold, best_accuracy, best_precision, best_recall, c_pos, c_neg

    def train(self):
        if self.optimizer_method == 'sgd':
            self.run_model(optim.SGD(self.params_to_update, lr=self.lr, momentum=0.0))
        else:
            print('Invalid option!')
        return self.load_best_model()

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

            self.lr *= self.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= self.lr_decay

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val score: {:4f}'.format(self.best_score))

        # load best model weights
        self.model.load_state_dict(self.best_model_wts)

        self.writer.close()
        return self.model

    def soft_clip(self, x):
        original_shape = x.shape
        x = x.view(-1, original_shape[-1])
        # direction = F.normalize(x, dim=1)
        # norm = torch.norm(x, dim=1, keepdim=True)
        # x = direction * (norm + self.inner_radius)

        with torch.no_grad():
            norm = torch.norm(x, dim=1, keepdim=True).repeat(1, self.embedding_dim)
            # print('norm', norm)
            # input()
            x[norm <= self.criterion.inner_radius] = x[norm <= self.criterion.inner_radius] / norm[norm <= self.criterion.inner_radius] * self.criterion.inner_radius
            x[norm >= 1.0] = x[norm >= 1.0]/norm[norm >= 1.0]*(1.0-1e-5)
        return x.view(original_shape)

    def mob_add(self, u, v):
        v = v + 1e-6
        tf_dot_u_v = 2. * torch.sum(u*v, dim=1, keepdim=True)
        tf_norm_u_sq = torch.sum(u*u, dim=1, keepdim=True)
        tf_norm_v_sq = torch.sum(v*v, dim=1, keepdim=True)
        denominator = 1. + tf_dot_u_v + tf_norm_v_sq * tf_norm_u_sq
        tf_dot_u_v = tf_dot_u_v.repeat(1, self.embedding_dim)
        tf_norm_u_sq = tf_norm_u_sq.repeat(1, self.embedding_dim)
        tf_norm_v_sq = tf_norm_v_sq.repeat(1, self.embedding_dim)
        denominator = denominator.repeat(1, self.embedding_dim)
        result = (1. + tf_dot_u_v + tf_norm_v_sq) / denominator * u + (1. - tf_norm_u_sq) / denominator * v
        return self.soft_clip(result)

    def lambda_x(self, x):
        # print('lx norm', torch.norm(x, p=2, dim=1, keepdim=True))
        # print((2. / (1 - torch.norm(x, p=2, dim=1, keepdim=True).repeat(1, self.embedding_dim)))**2)
        # input()
        return 2. / (1 - torch.norm(x, p=2, dim=1, keepdim=True).repeat(1, self.embedding_dim))

    def exp_map_x(self, x, v):
        v = v + 1e-15
        norm_v = torch.norm(v, p=2, dim=1, keepdim=True).repeat(1, self.embedding_dim)
        second_term = torch.tanh(torch.clamp(self.lambda_x(x) * norm_v / 2, min=-15.0, max=15.0)) * v/norm_v
        # print('second_term', second_term)
        # input()
        return self.mob_add(x, second_term)

    def plot_label_embeddings(self):
        self.vizualize()

    def vizualize(self, save_to_disk=True):
        filename = '{:04d}'.format(self.epoch)

        labels = self.model.module.embeddings.weight.data.cpu()

        colors = ['c', 'm', 'y', 'k']
        embeddings_x, embeddings_y, annotation, color_list = {}, {}, {}, {}

        connected_to = {}

        fig, ax = plt.subplots()

        for level_id in range(len(self.labelmap.levels)):
            level_start, level_stop = self.labelmap.level_start[level_id], self.labelmap.level_stop[level_id]
            level_color = colors[level_id]
            for label_ix in range(self.labelmap.levels[level_id]):
                emb_id = label_ix + level_start
                emb = labels[emb_id, :].numpy()

                embeddings_x[emb_id] = emb[0]
                embeddings_y[emb_id] = emb[1]
                annotation[emb_id] = '{}'.format(getattr(self.labelmap,
                                                      '{}_ix_to_str'.format(self.labelmap.level_names[level_id]))[label_ix]
                                                 )
                color_list[emb_id] = level_color

                connected_to[emb_id] = [v for u, v in list(self.G.edges(emb_id))]

                if level_id == 3:
                    ax.scatter(emb[0], emb[1], c=level_color, alpha=0.5, linewidth=0)
                else:
                    ax.scatter(emb[0], emb[1], c=level_color, alpha=1)
                # ax.annotate(annotation[emb_id], (emb[0], emb[1]))

                # if level_id in [0, 1]:
                #     p = self.get_wedge(emb, radius=50)
                #     ax.add_collection(p)


        # fig, ax = plt.subplots()
        # ax.scatter(embeddings_x, embeddings_y, c=color_list)

        for from_node in connected_to:
            for to_node in connected_to[from_node]:
                if to_node in embeddings_x:
                    plt.plot([embeddings_x[from_node], embeddings_x[to_node]], [embeddings_y[from_node], embeddings_y[to_node]],
                             'b-', alpha=0.2)

        ax.axis('equal')
        # if self.title_text:
        #     fig.suptitle(self.title_text, family='sans-serif')
        if save_to_disk:
            fig.set_size_inches(8, 7)
            fig.savefig(os.path.join(self.log_dir, '{}.pdf'.format(filename)), dpi=200)
            fig.savefig(os.path.join(self.log_dir, '{}.png'.format(filename)), dpi=200)
        plt.close(fig)
        return ax

    def pass_samples(self, phase, save_to_tensorboard=True):
        reconstruction_f1, reconstruction_threshold, reconstruction_accuracy = -1, -1, -1
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0

        predicted_from_embeddings, predicted_to_embeddings = torch.tensor([]), torch.tensor([])
        e_positive, e_negative = torch.tensor([]), torch.tensor([])

        # Iterate over data.
        for index, data_item in enumerate(tqdm(self.dataloaders[phase])):
            inputs_from, inputs_to, status = data_item['from'], data_item['to'], data_item['status']

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                self.model = self.model.to(self.device)
                outputs_from, outputs_to, loss, e_for_u_v_positive, e_for_u_v_negative =\
                    self.criterion(self.model, inputs_from, inputs_to, status, phase, self.neg_to_pos_ratio)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # convert euclidean gradients to riemannian gradients for the label embeddings
                    # print('euc grad', self.model.module.embeddings.weight.grad.data)
                    # print('lambda', (1.0/self.lambda_x(self.model.module.embeddings.weight.data))**2)
                    self.model.module.embeddings.weight.grad.data *= (1.0 / self.lambda_x(
                        self.model.module.embeddings.weight.data)) ** 2  # ((1 - (torch.norm(self.model.module.embeddings.weight.grad.data, p=2, dim=1, keepdim=True) ** 2)) ** 2) / 4).repeat(1, self.embedding_dim)
                    # print('weights', self.model.module.embeddings.weight.data)
                    # print('grad', self.model.module.embeddings.weight.grad.data)
                    # input()
                    self.model.module.embeddings.weight.data = self.exp_map_x(self.model.module.embeddings.weight.data,
                                                                              -self.lr * self.model.module.embeddings.weight.grad.data)
                    # print('weights', self.model.module.embeddings.weight.data)
                    # input()
                    # self.optimizer.step()

            # statistics
            running_loss += loss.item()

            outputs_from, outputs_to = outputs_from.cpu().detach(), outputs_to.cpu().detach()

            predicted_from_embeddings = torch.cat((predicted_from_embeddings, outputs_from.data))
            predicted_to_embeddings = torch.cat((predicted_to_embeddings, outputs_to.data))
            e_positive = torch.cat((e_positive, e_for_u_v_positive.cpu().detach().data))
            e_negative = torch.cat((e_negative, e_for_u_v_negative.cpu().detach().data))

        metrics = EmbeddingMetrics(e_positive, e_negative, self.optimal_threshold, phase, n_proc=self.n_proc)

        f1_score, threshold, accuracy, precision, recall, _, _ = metrics.calculate_metrics()
        if phase == 'train':
            self.plot_label_embeddings()

        if phase == 'val':
            self.optimal_threshold = threshold

        if phase == 'test' and (self.epoch % self.check_reconstr_every == 0 or not save_to_tensorboard):
            reconstruction_f1, reconstruction_threshold, reconstruction_accuracy, reconstruction_prec, reconstruction_recall, c_pos, c_neg = self.check_graph_embedding()
            print('Reconstruction task: F1: {:.4f},  Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, Threshold: {:.4f}'.format(reconstruction_f1,
                                                                                                 reconstruction_accuracy,
                                                                                                 reconstruction_prec,
                                                                                                 reconstruction_recall,
                                                                                                 reconstruction_threshold))
            self.reconstruction_f1, self.reconstruction_threshold, self.reconstruction_accuracy, self.reconstruction_prec, self.reconstruction_recall = reconstruction_f1, reconstruction_threshold, reconstruction_accuracy, reconstruction_prec, reconstruction_recall
            self.correct_pos, self.correct_neg = c_pos, c_neg

        epoch_loss = running_loss / self.dataset_length[phase]

        if save_to_tensorboard:
            self.writer.add_scalar('{}_loss'.format(phase), epoch_loss, self.epoch)
            self.writer.add_scalar('{}_f1_score'.format(phase), f1_score, self.epoch)
            self.writer.add_scalar('{}_accuracy'.format(phase), accuracy, self.epoch)
            self.writer.add_scalar('{}_thresh'.format(phase), self.optimal_threshold, self.epoch)

            if phase == 'test' and self.epoch % self.check_reconstr_every == 0:
                self.writer.add_scalar('reconstruction_thresh', self.reconstruction_threshold, self.epoch)
                self.writer.add_scalar('reconstruction_f1', self.reconstruction_f1, self.epoch)
                self.writer.add_scalar('reconstruction_precision', self.reconstruction_prec, self.epoch)
                self.writer.add_scalar('reconstruction_recall', self.reconstruction_recall, self.epoch)
                self.writer.add_scalar('reconstruction_accuracy', self.reconstruction_accuracy, self.epoch)
                self.writer.add_scalar('correct_positives', self.correct_pos, self.epoch)
                self.writer.add_scalar('correct_negatives', self.correct_neg, self.epoch)

        print('{} Loss: {:.4f} lr: {:.5f}, F1-score: {:.4f}, Accuracy: {:.4f}'.format(phase, epoch_loss, self.lr,
                                                                                      f1_score, accuracy))

        # deep copy the model
        if phase == 'test':
            if self.epoch % self.save_model_every == 0:
                self.save_model(epoch_loss)
        if phase == 'val':
            if f1_score >= self.best_score:
                self.best_score = f1_score
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                self.save_model(epoch_loss, filename='best_model')

        return reconstruction_f1, accuracy

    def save_model(self, loss, filename=None):
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'optimal_threshold': self.optimal_threshold,
            'reconstruction_scores': {'f1': self.reconstruction_f1, 'precision': self.reconstruction_prec,
                                      'recall': self.reconstruction_recall, 'accuracy': self.reconstruction_accuracy,
                                      'threshold': self.reconstruction_threshold}
        }, os.path.join(self.path_to_save_model, '{}.pth'.format(filename if filename else self.epoch)))
        print('Successfully saved model epoch {} to {} as {}.pth'.format(self.epoch, self.path_to_save_model,
                                                                         filename if filename else self.epoch))

    def load_model(self, epoch_to_load):
        checkpoint = torch.load(os.path.join(self.path_to_save_model, '{}.pth'.format(epoch_to_load)), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.optimal_threshold = checkpoint['optimal_threshold']
        self.reconstruction_f1, self.reconstruction_threshold, self.reconstruction_accuracy, self.reconstruction_prec, self.reconstruction_recall = \
        checkpoint['reconstruction_scores']['f1'], checkpoint['reconstruction_scores']['threshold'], \
        checkpoint['reconstruction_scores']['accuracy'], checkpoint['reconstruction_scores']['precision'], \
        checkpoint['reconstruction_scores']['recall']
        print('Successfully loaded model epoch {} from {}'.format(self.epoch, self.path_to_save_model))

    def load_inverted_cosine_emb(self, path_to_weights):
        path_to_emb = path_to_weights
        emb_info = np.load(path_to_emb).item()

        embeddings_x, embeddings_y = emb_info['x'], emb_info['y']
        label_embeddings = np.zeros((len(embeddings_x.keys()), self.embedding_dim))
        for label_ix in embeddings_x:
            label_embeddings[label_ix, 0], label_embeddings[label_ix, 1] = embeddings_x[label_ix], embeddings_y[label_ix]

        # invert embeddings
        label_norms = np.linalg.norm(label_embeddings, axis=1, ord=2)
        max_norm = np.max(label_norms)

        for label_ix in embeddings_x:
            label_embeddings[label_ix, :] *= (3.0 * max_norm / label_norms[label_ix] ** 2)
        print(label_embeddings)
        self.model.embeddings.weight.data.copy_(torch.from_numpy(label_embeddings))
        print(self.model.embeddings.weight.is_leaf)
        print(list(self.model.parameters()))
        print('Succesfully loaded inverted cosine embeddings from {}'.format(path_to_weights))

    def find_existing_weights(self):
        weights = sorted([filename for filename in os.listdir(self.path_to_save_model)])
        weights = weights[:-1]
        weights.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        if len(weights) < 1:
            print('Could not find weights to load from, will train from scratch.')
        else:
            self.load_model(epoch_to_load=weights[-1].split('.')[0])

    def load_best_model(self, only_load=False):
        if only_load:
            self.load_model(epoch_to_load='best_model')
        else:
            self.load_model(epoch_to_load='best_model')
            return self.pass_samples(phase='test', save_to_tensorboard=False)


class OrderEmbeddingLoss(torch.nn.Module):
    def __init__(self, labelmap, neg_to_pos_ratio, alpha=1.0, pick_per_level=True, weigh_neg_term=False, level_weights=None, weigh_pos_term=False):
        print('Using order-embedding loss!')
        torch.nn.Module.__init__(self)
        self.labelmap = labelmap
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.alpha = alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pick_per_level = pick_per_level
        self.weigh_neg_term = weigh_neg_term
        self.weigh_pos_term = weigh_pos_term

        self.level_weights = level_weights
        if self.level_weights is None:
            self.level_weights = torch.ones((len(self.labelmap.levels)))

        self.G_tc = None
        self.nodes_in_G_tc = None
        self.n_nodes_G_tc = None

    def set_graph_tc(self, graph_tc):
        self.G_tc = graph_tc
        self.nodes_in_G_tc = set(list(self.G_tc))
        self.n_nodes_G_tc = len(set(list(self.G_tc)))

    def set_negative_graph(self, n_G, mapping_from_node_to_ix, mapping_from_ix_to_node):
        """
        Get graph to pick negative edges from.
        :param n_G: <np.array> Bool adjacency matrix; containing 1s for edges (u, v) which represent a negative edge
        :param mapping_from_node_to_ix: <dict> mapping from integer indices to node names
        :param mapping_from_ix_to_node: <dict> mapping from node names to integer indices
        :return: NA
        """
        self.negative_G = n_G
        self.mapping_from_node_to_ix = mapping_from_node_to_ix
        self.mapping_from_ix_to_node = mapping_from_ix_to_node

    def sample_negative_edge(self, u=None, v=None, level_id=None):
        if level_id is not None:
            level_id = level_id % len(self.labelmap.level_names)
        if u is not None and v is None:
            choose_from = np.where(self.negative_G[self.mapping_from_node_to_ix[u], :] == 1)[0]
        elif u is None and v is not None:
            choose_from = np.where(self.negative_G[:, self.mapping_from_node_to_ix[v]] == 1)[0]
        else:
            print('Error! Both (u, v) given or neither (u, v) given!')

        if self.pick_per_level:
            if level_id < len(self.labelmap.levels):
                level_start, level_stop = self.labelmap.level_start[level_id], self.labelmap.level_stop[level_id]
                choose_from = choose_from[np.where(np.logical_and(choose_from >= level_start, choose_from < level_stop))].tolist()
            else:
                choose_from = choose_from.tolist()
        else:
            choose_from = choose_from.tolist()
        corrupted_ix = random.choice(choose_from)
        return corrupted_ix

    @staticmethod
    def E_operator(x, y):
        original_shape = x.shape
        x = x.contiguous().view(-1, original_shape[-1])
        y = y.contiguous().view(-1, original_shape[-1])

        return torch.sum(torch.clamp(x-y, min=0.0)**2, dim=1).view(original_shape[:-1])

    def positive_pair(self, x, y):
        return self.E_operator(x, y)

    def negative_pair(self, x, y):
        return torch.clamp(self.alpha-self.E_operator(x, y), min=0.0), self.E_operator(x, y)

    def get_level_weight_for_edge(self, to):
        retval = torch.ones((len(to)))
        for level_ix, (level_start, level_stop) in enumerate(zip(self.labelmap.level_start, self.labelmap.level_stop)):
            for to_ix in range(len(to)):
                if level_start <= to[to_ix] < level_stop:
                    retval[to_ix] = self.level_weights[level_ix]
        return retval

    def forward(self, model, inputs_from, inputs_to, status, phase, neg_to_pos_ratio):
        loss = 0.0
        e_for_u_v_positive_all, e_for_u_v_negative_all = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
        predicted_from_embeddings_all = torch.tensor([]).to(self.device)
        predicted_to_embeddings_all = torch.tensor([]).to(self.device)

        predicted_from_embeddings = model(torch.tensor(inputs_from).to(self.device))
        predicted_to_embeddings = model(torch.tensor(inputs_to).to(self.device))
        predicted_from_embeddings_all = torch.cat((predicted_from_embeddings_all, predicted_from_embeddings))
        predicted_to_embeddings_all = torch.cat((predicted_to_embeddings_all, predicted_to_embeddings))

        if phase != 'train':
            # loss for positive pairs
            positive_indices = (status == 1).nonzero().squeeze(dim=1)
            e_for_u_v_positive = self.positive_pair(predicted_from_embeddings[positive_indices],
                                                    predicted_to_embeddings[positive_indices])
            loss += torch.sum(e_for_u_v_positive)
            e_for_u_v_positive_all = torch.cat((e_for_u_v_positive_all, e_for_u_v_positive))

            # loss for negative pairs
            negative_indices = (status == 0).nonzero().squeeze(dim=1)
            neg_term, e_for_u_v_negative = self.negative_pair(predicted_from_embeddings[negative_indices],
                                                             predicted_to_embeddings[negative_indices])
            loss += torch.sum(neg_term)
            e_for_u_v_negative_all = torch.cat((e_for_u_v_negative_all, e_for_u_v_negative))

        else:
            # get level weights for each edge
            level_weights_per_edge = self.get_level_weight_for_edge(to=inputs_to)
            level_weights_per_edge = level_weights_per_edge.to(self.device)

            # loss for positive pairs
            positive_indices = (status == 1).nonzero().squeeze(dim=1)
            e_for_u_v_positive = self.positive_pair(predicted_from_embeddings[positive_indices],
                                                    predicted_to_embeddings[positive_indices])
            loss += torch.sum(level_weights_per_edge * e_for_u_v_positive)
            e_for_u_v_positive_all = torch.cat((e_for_u_v_positive_all, e_for_u_v_positive))

            # loss for negative pairs
            negative_from = [None] * (2 * self.neg_to_pos_ratio * len(inputs_from))
            negative_to = [None] * (2 * self.neg_to_pos_ratio * len(inputs_to))
            if self.weigh_neg_term:
                negative_weights = torch.ones((2 * self.neg_to_pos_ratio * len(inputs_to)))*self.n_nodes_G_tc/self.neg_to_pos_ratio
            else:
                negative_weights = torch.ones((2 * self.neg_to_pos_ratio * len(inputs_to)))

            for sample_id in range(len(inputs_from)):
                # loss for negative pairs
                sample_inputs_from, sample_inputs_to = inputs_from[sample_id], inputs_to[
                    sample_id]
                for pass_ix in range(self.neg_to_pos_ratio):
                    corrupted_ix = self.sample_negative_edge(u=sample_inputs_from, v=None, level_id=pass_ix)
                    negative_from[2 * self.neg_to_pos_ratio * sample_id + pass_ix] = sample_inputs_from
                    negative_to[2 * self.neg_to_pos_ratio * sample_id + pass_ix] = self.mapping_from_ix_to_node[
                        corrupted_ix]

                    if self.weigh_neg_term:
                        deg_tc_u = len(self.G_tc.in_edges(negative_to[2 * self.neg_to_pos_ratio * sample_id + pass_ix]))
                        if deg_tc_u != 0:
                            negative_weights[2 * self.neg_to_pos_ratio * sample_id + pass_ix] *= (1.0/deg_tc_u)
                    if not self.weigh_pos_term:
                        negative_weights[2 * self.neg_to_pos_ratio * sample_id + pass_ix] *= level_weights_per_edge[sample_id]

                    corrupted_ix = self.sample_negative_edge(u=None, v=sample_inputs_to, level_id=pass_ix)
                    negative_from[
                        2 * self.neg_to_pos_ratio * sample_id + pass_ix + self.neg_to_pos_ratio] = \
                        self.mapping_from_ix_to_node[corrupted_ix]
                    negative_to[
                        2 * self.neg_to_pos_ratio * sample_id + pass_ix + self.neg_to_pos_ratio] = sample_inputs_to

                    if self.weigh_neg_term:
                        deg_tc_v = len(self.G_tc.out_edges(negative_from[2 * self.neg_to_pos_ratio * sample_id + pass_ix + self.neg_to_pos_ratio]))
                        if deg_tc_v != 0:
                            negative_weights[2 * self.neg_to_pos_ratio * sample_id + pass_ix + self.neg_to_pos_ratio] *= (1.0/deg_tc_v)
                    if not self.weigh_pos_term:
                        negative_weights[2 * self.neg_to_pos_ratio * sample_id + pass_ix + self.neg_to_pos_ratio] *= level_weights_per_edge[sample_id]

            negative_from_embeddings, negative_to_embeddings = model(torch.tensor(negative_from).to(self.device)), model(torch.tensor(negative_to).to(self.device))
            neg_term, e_for_u_v_negative = self.negative_pair(negative_from_embeddings, negative_to_embeddings)
            loss += torch.sum(negative_weights.to(self.device) * neg_term)
            e_for_u_v_negative_all = torch.cat((e_for_u_v_negative_all, e_for_u_v_negative))


        return predicted_from_embeddings_all, predicted_to_embeddings_all, loss, e_for_u_v_positive_all, e_for_u_v_negative_all


class EucConesLoss(torch.nn.Module):
    def __init__(self, labelmap, neg_to_pos_ratio, alpha=1.0, pick_per_level=False):
        print('Using hyp cones loss!')
        torch.nn.Module.__init__(self)
        self.labelmap = labelmap
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.alpha = alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pick_per_level = pick_per_level

        self.level_weights = torch.ones((len(self.labelmap.levels)))

        self.G_tc = None
        self.nodes_in_G_tc = None
        self.n_nodes_G_tc = None

        self.K = 0.1
        self.inner_radius = 2 * self.K / (1 + np.sqrt(1 + 4 * self.K * self.K))
        self.epsilon = 1e-5

    def set_graph_tc(self, graph_tc):
        self.G_tc = graph_tc
        self.nodes_in_G_tc = set(list(self.G_tc))
        self.n_nodes_G_tc = len(set(list(self.G_tc)))

    def E_operator(self, x, y):

        original_shape = x.shape
        x = x.view(-1, original_shape[-1])
        y = y.view(-1, original_shape[-1])

        x_norm = torch.norm(x, p=2, dim=1)
        y_norm = torch.norm(y, p=2, dim=1)
        x_y_dist = torch.norm(x - y, p=2, dim=1)

        x_dot_y = torch.sum(x * y, dim=1)

        acos_arg = (x_dot_y * (1 + x_norm ** 2) - (x_norm ** 2) * (1 + y_norm ** 2)) / (
                    x_norm * x_y_dist * torch.sqrt(1 + (x_norm * y_norm) ** 2 - 2 * x_dot_y))

        # in angle space (radians)
        theta_between_x_y = torch.acos(torch.clamp(acos_arg, min=-1 + 1e-5, max=1 - 1e-5))
        psi_x = torch.asin(torch.clamp(self.K * (1 - x_norm ** 2) / x_norm, min=-1 + 1e-5, max=1 - 1e-5))

        # in cos space
        # theta_between_x_y = acos_arg
        # psi_x = -torch.sqrt(1 - (self.K*(1-x_norm**2)/x_norm)**2)

        return torch.clamp(theta_between_x_y - psi_x, min=0.0).view(original_shape[:-1])

    def positive_pair(self, x, y):
        return self.E_operator(x, y)

    def negative_pair(self, x, y):
        return torch.clamp(self.alpha-self.E_operator(x, y), min=0.0), self.E_operator(x, y)

    def set_negative_graph(self, n_G, mapping_from_node_to_ix, mapping_from_ix_to_node):
        """
        Get graph to pick negative edges from.
        :param n_G: <np.array> Bool adjacency matrix; containing 1s for edges (u, v) which represent a negative edge
        :param mapping_from_node_to_ix: <dict> mapping from integer indices to node names
        :param mapping_from_ix_to_node: <dict> mapping from node names to integer indices
        :return: NA
        """
        self.negative_G = n_G
        self.mapping_from_node_to_ix = mapping_from_node_to_ix
        self.mapping_from_ix_to_node = mapping_from_ix_to_node

    def sample_negative_edge(self, u=None, v=None, level_id=None):
        if level_id is not None:
            level_id = level_id % len(self.labelmap.level_names)
        if u is not None and v is None:
            choose_from = np.where(self.negative_G[self.mapping_from_node_to_ix[u], :] == 1)[0]
        elif u is None and v is not None:
            choose_from = np.where(self.negative_G[:, self.mapping_from_node_to_ix[v]] == 1)[0]
        else:
            print('Error! Both (u, v) given or neither (u, v) given!')

        if self.pick_per_level:
            if level_id < len(self.labelmap.levels):
                level_start, level_stop = self.labelmap.level_start[level_id], self.labelmap.level_stop[level_id]
                choose_from = choose_from[np.where(np.logical_and(choose_from >= level_start, choose_from < level_stop))].tolist()
            else:
                choose_from = choose_from.tolist()
        else:
            choose_from = choose_from.tolist()
        corrupted_ix = random.choice(choose_from)
        return corrupted_ix

    def get_level_weight_for_edge(self, to):
        retval = torch.ones((len(to)))
        for level_ix, (level_start, level_stop) in enumerate(zip(self.labelmap.level_start, self.labelmap.level_stop)):
            for to_ix in range(len(to)):
                if level_start <= to[to_ix] < level_stop:
                    retval[to_ix] = self.level_weights[level_ix]
        return retval

    def forward(self, model, inputs_from, inputs_to, status, phase, neg_to_pos_ratio):
        loss = 0.0
        e_for_u_v_positive_all, e_for_u_v_negative_all = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
        predicted_from_embeddings_all = torch.tensor([]).to(self.device)
        predicted_to_embeddings_all = torch.tensor([]).to(self.device)

        predicted_from_embeddings = model(torch.tensor(inputs_from).to(self.device))
        predicted_to_embeddings = model(torch.tensor(inputs_to).to(self.device))
        predicted_from_embeddings_all = torch.cat((predicted_from_embeddings_all, predicted_from_embeddings))
        predicted_to_embeddings_all = torch.cat((predicted_to_embeddings_all, predicted_to_embeddings))

        if phase != 'train':
            # loss for positive pairs
            positive_indices = (status == 1).nonzero().squeeze(dim=1)
            e_for_u_v_positive = self.positive_pair(predicted_from_embeddings[positive_indices],
                                                    predicted_to_embeddings[positive_indices])
            loss += torch.sum(e_for_u_v_positive)
            e_for_u_v_positive_all = torch.cat((e_for_u_v_positive_all, e_for_u_v_positive))

            # loss for negative pairs
            negative_indices = (status == 0).nonzero().squeeze(dim=1)
            neg_term, e_for_u_v_negative = self.negative_pair(predicted_from_embeddings[negative_indices],
                                                             predicted_to_embeddings[negative_indices])
            loss += torch.sum(neg_term)
            e_for_u_v_negative_all = torch.cat((e_for_u_v_negative_all, e_for_u_v_negative))

        else:
            # get level weights for each edge
            # level_weights_per_edge = self.get_level_weight_for_edge(to=inputs_to)
            # level_weights_per_edge = level_weights_per_edge.to(self.device)
            pos_weights = torch.ones((len(inputs_to))).to(self.device)
            negative_weights = torch.ones((2 * self.neg_to_pos_ratio * len(inputs_to))).to(self.device)

            pos_weights *= 1.0
            negative_weights *= 1.0

            # loss for positive pairs
            positive_indices = (status == 1).nonzero().squeeze(dim=1)
            e_for_u_v_positive = self.positive_pair(predicted_from_embeddings[positive_indices],
                                                    predicted_to_embeddings[positive_indices])
            # loss += torch.sum(level_weights_per_edge*e_for_u_v_positive)
            loss += torch.sum(pos_weights*e_for_u_v_positive)
            e_for_u_v_positive_all = torch.cat((e_for_u_v_positive_all, e_for_u_v_positive))

            # loss for negative pairs
            negative_from = [None] * (2 * self.neg_to_pos_ratio * len(inputs_from))
            negative_to = [None] * (2 * self.neg_to_pos_ratio * len(inputs_to))
            # if self.weigh_neg_term:
            #     negative_weights = torch.ones((2 * self.neg_to_pos_ratio * len(inputs_to)))*self.n_nodes_G_tc/self.neg_to_pos_ratio
            # else:
            #     negative_weights = torch.ones((2 * self.neg_to_pos_ratio * len(inputs_to)))

            for sample_id in range(len(inputs_from)):
                # loss for negative pairs
                sample_inputs_from, sample_inputs_to = inputs_from[sample_id], inputs_to[sample_id]
                for pass_ix in range(self.neg_to_pos_ratio):
                    corrupted_ix = self.sample_negative_edge(u=sample_inputs_from, v=None, level_id=pass_ix)
                    negative_from[2 * self.neg_to_pos_ratio * sample_id + pass_ix] = sample_inputs_from
                    negative_to[2 * self.neg_to_pos_ratio * sample_id + pass_ix] = self.mapping_from_ix_to_node[
                        corrupted_ix]


                    corrupted_ix = self.sample_negative_edge(u=None, v=sample_inputs_to, level_id=pass_ix)
                    negative_from[
                        2 * self.neg_to_pos_ratio * sample_id + pass_ix + self.neg_to_pos_ratio] = \
                        self.mapping_from_ix_to_node[corrupted_ix]
                    negative_to[
                        2 * self.neg_to_pos_ratio * sample_id + pass_ix + self.neg_to_pos_ratio] = sample_inputs_to

            negative_from_embeddings, negative_to_embeddings = model(torch.tensor(negative_from).to(self.device)), model(torch.tensor(negative_to).to(self.device))
            neg_term, e_for_u_v_negative = self.negative_pair(negative_from_embeddings, negative_to_embeddings)
            loss += torch.sum(negative_weights.to(self.device)*neg_term)
            e_for_u_v_negative_all = torch.cat((e_for_u_v_negative_all, e_for_u_v_negative))

        return predicted_from_embeddings_all, predicted_to_embeddings_all, loss, e_for_u_v_positive_all, e_for_u_v_negative_all


class SimpleEuclideanEmbLoss(torch.nn.Module):
    def __init__(self, labelmap, neg_to_pos_ratio):
        print('Using order-embedding loss!')
        torch.nn.Module.__init__(self)
        self.labelmap = labelmap
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.neg_to_pos_ratio = neg_to_pos_ratio

        self.G_tc = None
        self.reverse_G = None
        self.nodes_in_graph = None
        self.num_edges = None

    def set_graph_tc(self, graph_tc):
        self.G_tc = graph_tc
        self.reverse_G = nx.reverse(self.G_tc)
        self.nodes_in_graph = set(list(self.G_tc))
        self.num_edges = self.G_tc.size()

    @staticmethod
    def d_fn(x, y):
        return torch.sum((y-x)**2, dim=1)

    def forward(self, model, inputs_from, inputs_to, status, phase, neg_to_pos_ratio):
        loss = 0.0
        d_for_u_v_positive_all, d_for_u_v_negative_all = torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
        predicted_from_embeddings_all = torch.tensor([]).to(self.device) # model(inputs_from)
        predicted_to_embeddings_all = torch.tensor([]).to(self.device) # model(inputs_to)

        if phase != 'train':
            inputs_from, inputs_to, status = torch.tensor(inputs_from), torch.tensor(inputs_to), torch.tensor(status)
            predicted_from_embeddings = model(inputs_from.to(self.device))
            predicted_to_embeddings = model(inputs_to.to(self.device))

            # loss for positive pairs
            positive_indices = (status == 1).nonzero().squeeze(dim=1)
            d_for_u_v_positive = self.d_fn(predicted_from_embeddings[positive_indices],
                                           predicted_to_embeddings[positive_indices])
            d_u_u = self.d_fn(predicted_from_embeddings[positive_indices], predicted_from_embeddings[positive_indices])

            d_for_u_v_positive_all = torch.cat((d_for_u_v_positive_all, d_for_u_v_positive))

            # loss for negative pairs
            negative_indices = (status == 0).nonzero().squeeze(dim=1)
            d_for_u_v_negative = self.d_fn(predicted_from_embeddings[negative_indices],
                                           predicted_to_embeddings[negative_indices])

            d_for_u_v_negative_all = torch.cat((d_for_u_v_negative_all, d_for_u_v_negative))
            loss += d_for_u_v_positive + torch.log(torch.sum(torch.exp(-d_for_u_v_negative), dim=0) + torch.exp(-d_u_u))

        else:
            for batch_id in range(len(inputs_from)):

                predicted_from_embeddings = model(inputs_from[batch_id].to(self.device))
                predicted_to_embeddings = model(inputs_to[batch_id].to(self.device))
                predicted_from_embeddings_all = torch.cat((predicted_from_embeddings_all, predicted_from_embeddings))
                predicted_to_embeddings_all = torch.cat((predicted_to_embeddings_all, predicted_to_embeddings))

                # loss for positive pairs
                positive_indices = (status[batch_id] == 1).nonzero().squeeze(dim=1)
                d_for_u_v_positive = self.d_fn(predicted_from_embeddings[positive_indices],
                                               predicted_to_embeddings[positive_indices])
                d_u_u = self.d_fn(predicted_from_embeddings[positive_indices],
                                  predicted_from_embeddings[positive_indices])

                d_for_u_v_positive_all = torch.cat((d_for_u_v_positive_all, d_for_u_v_positive))

                # loss for negative pair
                for sample_id in range(inputs_from[batch_id].shape[0]):
                    negative_from = torch.zeros((2 * self.neg_to_pos_ratio), dtype=torch.long)
                    negative_to = torch.zeros((2 * self.neg_to_pos_ratio), dtype=torch.long)

                    sample_inputs_from, sample_inputs_to = inputs_from[batch_id][sample_id], inputs_to[batch_id][sample_id]
                    for pass_ix in range(self.neg_to_pos_ratio):

                        list_of_edges_from_ui = [v for u, v in list(self.G_tc.edges(sample_inputs_from.item()))]
                        corrupted_ix = random.choice(list(self.nodes_in_graph - set(list_of_edges_from_ui)))
                        negative_from[pass_ix] = sample_inputs_from
                        negative_to[pass_ix] = corrupted_ix

                        list_of_edges_to_vi = [v for u, v in list(self.reverse_G.edges(sample_inputs_to.item()))]
                        corrupted_ix = random.choice(list(self.nodes_in_graph - set(list_of_edges_to_vi)))
                        negative_from[pass_ix + self.neg_to_pos_ratio] = corrupted_ix
                        negative_to[pass_ix + self.neg_to_pos_ratio] = sample_inputs_to

                    negative_from_embeddings, negative_to_embeddings = model(negative_from.to(self.device)), model(negative_to.to(self.device))
                    d_for_u_v_negative = self.d_fn(negative_from_embeddings,
                                                   negative_to_embeddings)

                    d_for_u_v_negative_all = torch.cat((d_for_u_v_negative_all, d_for_u_v_negative))

                    # add the positive and negative terms
                    loss += d_for_u_v_positive[sample_id] + torch.log(torch.sum(torch.exp(-torch.cat((d_for_u_v_negative, d_u_u[sample_id].unsqueeze(dim=0)))), dim=0))

        return predicted_from_embeddings_all, predicted_to_embeddings_all, loss, d_for_u_v_positive_all, d_for_u_v_negative_all


def order_embedding_train_model(arguments):
    if not os.path.exists(os.path.join(arguments.experiment_dir, arguments.experiment_name)):
        os.makedirs(os.path.join(arguments.experiment_dir, arguments.experiment_name))
    args_dict = vars(arguments)
    repo = git.Repo(search_parent_directories=True)
    args_dict['commit_hash'] = repo.head.object.hexsha
    args_dict['branch'] = repo.active_branch.name
    with open(os.path.join(arguments.experiment_dir, arguments.experiment_name, 'config_params.txt'), 'w') as file:
        file.write(json.dumps(args_dict, indent=4))

    print('Config parameters for this run are:\n{}'.format(json.dumps(vars(arguments), indent=4)))

    # initial_crop = 324
    input_size = 224
    labelmap = ETHECLabelMap()
    if arguments.merged:
        labelmap = ETHECLabelMapMerged()
    if arguments.debug:
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
    if arguments.use_grayscale:
        train_data_transforms = transforms.Compose([transforms.ToPILImage(),
                                                    transforms.Grayscale(),
                                                    transforms.Resize((input_size, input_size)),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    ])
        val_test_data_transforms = transforms.Compose([transforms.ToPILImage(),
                                                       transforms.Grayscale(),
                                                       transforms.Resize((input_size, input_size)),
                                                       transforms.ToTensor(),
                                                       ])

    if not arguments.merged:
        train_set = ETHECDB(path_to_json='../database/ETHEC/train.json',
                            path_to_images=arguments.image_dir,
                            labelmap=labelmap, transform=train_data_transforms, with_images=False)
        val_set = ETHECDB(path_to_json='../database/ETHEC/val.json',
                          path_to_images=arguments.image_dir,
                          labelmap=labelmap, transform=val_test_data_transforms, with_images=False)
        test_set = ETHECDB(path_to_json='../database/ETHEC/test.json',
                           path_to_images=arguments.image_dir,
                           labelmap=labelmap, transform=val_test_data_transforms, with_images=False)
    elif not arguments.debug:
        train_set = ETHECDBMerged(path_to_json='../database/ETHEC/train.json',
                                  path_to_images=arguments.image_dir,
                                  labelmap=labelmap, transform=train_data_transforms, with_images=False)
        val_set = ETHECDBMerged(path_to_json='../database/ETHEC/val.json',
                                path_to_images=arguments.image_dir,
                                labelmap=labelmap, transform=val_test_data_transforms, with_images=False)
        test_set = ETHECDBMerged(path_to_json='../database/ETHEC/test.json',
                                 path_to_images=arguments.image_dir,
                                 labelmap=labelmap, transform=val_test_data_transforms, with_images=False)
    else:
        labelmap = ETHECLabelMapMergedSmall(single_level=False)
        train_set = ETHECDBMergedSmall(path_to_json='../database/ETHEC/train.json',
                                       path_to_images=arguments.image_dir,
                                       labelmap=labelmap, transform=train_data_transforms, with_images=False)
        val_set = ETHECDBMergedSmall(path_to_json='../database/ETHEC/val.json',
                                     path_to_images=arguments.image_dir,
                                     labelmap=labelmap, transform=val_test_data_transforms, with_images=False)
        test_set = ETHECDBMergedSmall(path_to_json='../database/ETHEC/test.json',
                                      path_to_images=arguments.image_dir,
                                      labelmap=labelmap, transform=val_test_data_transforms, with_images=False)

    print('Dataset has following splits: train: {}, val: {}, test: {}'.format(len(train_set), len(val_set),
                                                                              len(test_set)))

    batch_size = arguments.batch_size
    n_workers = arguments.n_workers

    if arguments.debug:
        print("== Running in DEBUG mode!")
    trainloader = torch.utils.data.DataLoader(train_set,
                                              batch_size=batch_size,
                                              num_workers=n_workers,
                                              shuffle=True if arguments.class_weights else False,
                                              sampler=None if arguments.class_weights else WeightedResampler(
                                                  train_set, weight_strategy=arguments.weight_strategy))
    valloader = torch.utils.data.DataLoader(val_set,
                                            batch_size=batch_size,
                                            shuffle=False, num_workers=n_workers)
    testloader = torch.utils.data.DataLoader(test_set,
                                             batch_size=batch_size,
                                             shuffle=False, num_workers=n_workers)

    data_loaders = {'train': trainloader, 'val': valloader, 'test': testloader}

    weight = None
    if arguments.class_weights:
        n_train = torch.zeros(labelmap.n_classes)
        for data_item in data_loaders['train']:
            n_train += torch.sum(data_item['labels'], 0)
        weight = 1.0/n_train

    eval_type = MultiLabelEvaluation(os.path.join(arguments.experiment_dir, arguments.experiment_name), labelmap)
    if arguments.evaluator == 'MLST':
        eval_type = MultiLabelEvaluationSingleThresh(os.path.join(arguments.experiment_dir, arguments.experiment_name),
                                                     labelmap)

    use_criterion = None
    if arguments.loss == 'order_emb_loss':
        use_criterion = OrderEmbeddingLoss(labelmap=labelmap, neg_to_pos_ratio=arguments.neg_to_pos_ratio,
                                           alpha=arguments.alpha, pick_per_level=arguments.pick_per_level,
                                           weigh_neg_term=arguments.weigh_neg_term,
                                           level_weights=arguments.level_weights,
                                           weigh_pos_term=arguments.weigh_pos_term)
    elif arguments.loss == 'euc_cones_loss':
        use_criterion = EucConesLoss(labelmap=labelmap, neg_to_pos_ratio=arguments.neg_to_pos_ratio,
                                     alpha=arguments.alpha, pick_per_level=arguments.pick_per_level)
    else:
        print("== Invalid --loss argument")

    oe = OrderEmbedding(data_loaders=data_loaders, labelmap=labelmap, criterion=use_criterion, lr=arguments.lr,
                        batch_size=batch_size, evaluator=eval_type, experiment_name=arguments.experiment_name,
                        embedding_dim=arguments.embedding_dim, neg_to_pos_ratio=arguments.neg_to_pos_ratio, alpha=arguments.alpha,
                        proportion_of_nb_edges_in_train=arguments.prop_of_nb_edges, lr_step=arguments.lr_step,
                        experiment_dir=arguments.experiment_dir, n_epochs=arguments.n_epochs, pick_per_level=arguments.pick_per_level,
                        eval_interval=arguments.eval_interval, feature_extracting=arguments.freeze_weights,
                        load_wt=arguments.resume, optimizer_method=arguments.optimizer_method, lr_decay=arguments.lr_decay,
                        random_seed=arguments.random_seed, load_cosine_emb=arguments.load_cosine_emb)
    oe.prepare_model()
    if arguments.set_mode == 'train':
        reconstr_f1, acc = oe.train()
        title = 'Reconstruction F1 score: {:.4f} Accuracy: {:.4f}'.format(reconstr_f1, acc)

        from network.viz_hypernymy import VizualizeGraphRepresentation
        path_to_best = os.path.join(arguments.experiment_dir, arguments.experiment_name, 'weights', 'best_model.pth')
        viz = VizualizeGraphRepresentation(weights_to_load=path_to_best, title_text=title, debug=arguments.debug,
                                           loss_fn='ec' if arguments.loss == 'order_emb_loss' else 'oe')
    elif arguments.set_mode == 'test':
        oe.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", help='Use DEBUG mode.', action='store_true')
    parser.add_argument("--lr", help='Input learning rate.', type=float, default=0.001)
    parser.add_argument("--batch_size", help='Batch size.', type=int, default=8)
    parser.add_argument("--evaluator", help='Evaluator type.', type=str, default='ML')
    parser.add_argument("--experiment_name", help='Experiment name.', type=str, required=True)
    parser.add_argument("--experiment_dir", help='Experiment directory.', type=str, required=True)
    parser.add_argument("--image_dir", help='Image parent directory.', type=str, required=True)
    parser.add_argument("--n_epochs", help='Number of epochs to run training for.', type=int, required=True)
    parser.add_argument("--n_workers", help='Number of workers.', type=int, default=4)
    parser.add_argument("--eval_interval", help='Evaluate model every N intervals.', type=int, default=1)
    parser.add_argument("--embedding_dim", help='Dimensions of learnt embeddings.', type=int, default=10)
    parser.add_argument("--neg_to_pos_ratio", help='Number of negatives to sample for one positive.', type=int, default=5)
    parser.add_argument("--alpha", help='Margin alpha.', type=float, default=0.05)
    parser.add_argument("--prop_of_nb_edges", help='Proportion of non-basic edges to be added to train set.', type=float, default=0.90)
    parser.add_argument("--resume", help='Continue training from last checkpoint.', action='store_true')
    parser.add_argument("--weigh_pos_term", help='Use level weights for pos term only', action='store_true')
    parser.add_argument("--optimizer_method", help='[adam, sgd]', type=str, default='adam')
    parser.add_argument("--merged", help='Use dataset which has genus and species combined.', action='store_true')
    parser.add_argument("--weigh_neg_term", help='Weigh neg term in loss.', action='store_true')
    parser.add_argument("--weight_strategy", help='Use inverse freq or inverse sqrt freq. ["inv", "inv_sqrt"]',
                        type=str, default='inv')
    parser.add_argument("--model", help='NN model to use.', type=str, default='alexnet')
    parser.add_argument("--loss",
                        help='Loss function to use. [order_emb_loss, euc_emb_loss]',
                        type=str, required=True)
    parser.add_argument("--use_grayscale", help='Use grayscale images.', action='store_true')
    parser.add_argument("--class_weights", help='Re-weigh the loss function based on inverse class freq.',
                        action='store_true')
    parser.add_argument("--freeze_weights", help='This flag fine tunes only the last layer.', action='store_true')
    parser.add_argument("--pick_per_level", help='Pick negatives from each level in the graph.', action='store_true')
    parser.add_argument("--set_mode", help='If use training or testing mode (loads best model).', type=str,
                        required=True)
    parser.add_argument("--level_weights", help='List of weights for each level', nargs=4, default=None, type=float)
    parser.add_argument("--lr_step", help='List of epochs to make multiple lr by 0.1', nargs='*', default=[], type=int)
    parser.add_argument("--lr_decay", help='Decay lr by a factor.', default=1.0, type=float)
    parser.add_argument("--random_seed", help='Random seed for torch.', default=0, type=int)
    parser.add_argument("--load_cosine_emb", help='Path to cosine embeddings .np file', type=str, default=None)

    args = parser.parse_args()

    order_embedding_train_model(args)
