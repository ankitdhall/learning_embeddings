from __future__ import print_function
from __future__ import division
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import os

import json
import git
import argparse

import numpy as np
import random
random.seed(0)

import networkx as nx

import matplotlib
matplotlib.use('pdf')
#matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import cv2

from network.order_embeddings import OrderEmbedding, OrderEmbeddingLoss, EucConesLoss, Embedder
from tensorboardX import SummaryWriter
import torch.nn as nn


class ToyGraph:
    def __init__(self, levels=4, branching_factor=3):
        self.n_levels = levels
        self.branching_factor = branching_factor
        self.levels = [self.branching_factor**i for i in range(1, self.n_levels)]
        self.level_names = [str(i) for i in range(1, self.n_levels)]

        for level_id, level_name in enumerate(self.level_names):
            setattr(self, level_name, {'{}_{}'.format(level_name, str(i)): i for i in range(self.levels[level_id])})

        # make child_of_
        for level_id, level_name in enumerate(self.level_names[:-1]):
            setattr(self, 'child_of_' + level_name, {'{}_{}'.format(level_name, str(i)): ['{}_{}'.format(self.level_names[level_id+1], str(j+(self.branching_factor*i))) for j in range(self.branching_factor)] for i in range(self.levels[level_id])})

        self.n_classes = sum(self.levels)
        self.classes = [key for class_list in [getattr(self, level_name) for level_name in self.level_names] for key
                        in class_list]
        self.level_stop, self.level_start = [], []
        for level_id, level_len in enumerate(self.levels):
            if level_id == 0:
                self.level_start.append(0)
                self.level_stop.append(level_len)
            else:
                self.level_start.append(self.level_stop[level_id - 1])
                self.level_stop.append(self.level_stop[level_id - 1] + level_len)

        self.edges = set()
        for level_id, level_name in enumerate(self.level_names[:-1]):
            child_of_dict = getattr(self, 'child_of_' + level_name)
            for parent_node in child_of_dict:
                for child_node in child_of_dict[parent_node]:
                    u = getattr(self, level_name)[parent_node] + self.level_start[level_id]
                    v = getattr(self, self.level_names[level_id+1])[child_node] + self.level_start[level_id+1]
                    self.edges.add((u, v))


class ToyOrderEmbedding(OrderEmbedding):
    def __init__(self, labelmap, criterion, lr, batch_size, evaluator, experiment_name, embedding_dim,
                 neg_to_pos_ratio, alpha, proportion_of_nb_edges_in_train, lr_step=[], pick_per_level=False,
                 experiment_dir='../exp/', n_epochs=10, eval_interval=2, feature_extracting=True, load_wt=False,
                 optimizer_method='adam', lr_decay=1.0, random_seed=0):
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
        else:
            self.model = Embedder(embedding_dim=self.embedding_dim, labelmap=labelmap)
        self.model = self.model.to(self.device)
        self.model = nn.DataParallel(self.model)
        self.labelmap = labelmap

        self.G, self.G_train, self.G_val, self.G_test = nx.DiGraph(), nx.DiGraph(), nx.DiGraph(), nx.DiGraph()
        for edge in self.labelmap.edges:
            u, v = edge
            self.G.add_edge(u, v)

        self.G_tc = nx.transitive_closure(self.G)
        self.create_splits()

        self.criterion.set_negative_graph(self.G_train_neg, self.mapping_ix_to_node, self.mapping_node_to_ix)

        self.lr_decay = lr_decay

        self.check_graph_embedding_neg_graph = None
        self.check_reconstr_every = 1
        self.save_model_every = 5

        self.reconstruction_f1, self.reconstruction_threshold, self.reconstruction_accuracy, self.reconstruction_prec, self.reconstruction_recall = 0.0, 0.0, 0.0, 0.0, 0.0
        self.n_proc = 512 if torch.cuda.device_count() > 0 else 4
        print('Using {} processess!'.format(self.n_proc))

def embed_toy_model(arguments):
    if not os.path.exists(os.path.join(arguments.experiment_dir, arguments.experiment_name)):
        os.makedirs(os.path.join(arguments.experiment_dir, arguments.experiment_name))
    args_dict = vars(arguments)
    repo = git.Repo(search_parent_directories=True)
    args_dict['commit_hash'] = repo.head.object.hexsha
    args_dict['branch'] = repo.active_branch.name
    with open(os.path.join(arguments.experiment_dir, arguments.experiment_name, 'config_params.txt'), 'w') as file:
        file.write(json.dumps(args_dict, indent=4))

    print('Config parameters for this run are:\n{}'.format(json.dumps(vars(arguments), indent=4)))

    labelmap = ToyGraph(levels=arguments.tree_levels, branching_factor=arguments.tree_branching)

    batch_size = arguments.batch_size
    n_workers = arguments.n_workers

    eval_type = None

    use_criterion = None
    if arguments.loss == 'order_emb_loss':
        use_criterion = OrderEmbeddingLoss(labelmap=labelmap, neg_to_pos_ratio=arguments.neg_to_pos_ratio, alpha=arguments.alpha)
    elif arguments.loss == 'euc_cones_loss':
        use_criterion = EucConesLoss(labelmap=labelmap, neg_to_pos_ratio=arguments.neg_to_pos_ratio, alpha=arguments.alpha)
    else:
        print("== Invalid --loss argument")

    oe = ToyOrderEmbedding(labelmap=labelmap, criterion=use_criterion, lr=arguments.lr,
                           batch_size=batch_size, experiment_name=arguments.experiment_name,
                           embedding_dim=arguments.embedding_dim, neg_to_pos_ratio=arguments.neg_to_pos_ratio,
                           alpha=arguments.alpha, pick_per_level=arguments.pick_per_level,
                           proportion_of_nb_edges_in_train=arguments.prop_of_nb_edges, lr_step=arguments.lr_step,
                           experiment_dir=arguments.experiment_dir, n_epochs=arguments.n_epochs,
                           eval_interval=arguments.eval_interval, feature_extracting=False, evaluator=None,
                           load_wt=arguments.resume, optimizer_method=arguments.optimizer_method,
                           lr_decay=arguments.lr_decay, random_seed=arguments.random_seed)
    oe.prepare_model()
    f1, acc = oe.train()

    title = 'L={}, b={} \n F1 score: {:.4f} Accuracy: {:.4f}'.format(str(arguments.tree_levels-1),
                                                                     str(arguments.tree_branching), f1, acc)

    from network.viz_toy import VizualizeGraphRepresentation
    path_to_best = os.path.join(arguments.experiment_dir, arguments.experiment_name, 'weights', 'best_model.pth')
    viz = VizualizeGraphRepresentation(weights_to_load=path_to_best, title_text='', L=arguments.tree_levels, b=arguments.tree_branching)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", help='Input learning rate.', type=float, default=0.001)
    parser.add_argument("--batch_size", help='Batch size.', type=int, default=8)
    parser.add_argument("--experiment_name", help='Experiment name.', type=str, required=True)
    parser.add_argument("--experiment_dir", help='Experiment directory.', type=str, required=True)
    parser.add_argument("--n_epochs", help='Number of epochs to run training for.', type=int, required=True)
    parser.add_argument("--n_workers", help='Number of workers.', type=int, default=4)
    parser.add_argument("--eval_interval", help='Evaluate model every N intervals.', type=int, default=1)
    parser.add_argument("--embedding_dim", help='Dimensions of learnt embeddings.', type=int, default=10)
    parser.add_argument("--neg_to_pos_ratio", help='Number of negatives to sample for one positive.', type=int,
                        default=5)
    parser.add_argument("--alpha", help='Margin alpha.', type=float, default=0.05)
    parser.add_argument("--prop_of_nb_edges", help='Proportion of non-basic edges to be added to train set.',
                        type=float, default=0.0)
    parser.add_argument("--resume", help='Continue training from last checkpoint.', action='store_true')
    parser.add_argument("--optimizer_method", help='[adam, sgd]', type=str, default='adam')
    parser.add_argument("--loss",
                        help='Loss function to use. [order_emb_loss, euc_emb_loss]',
                        type=str, required=True)
    parser.add_argument("--pick_per_level", help='Pick negatives from each level in the graph.', action='store_true')
    parser.add_argument("--lr_step", help='List of epochs to make multiple lr by 0.1', nargs='*', default=[],
                        type=int)
    parser.add_argument("--lr_decay", help='Decay lr by a factor.', default=1.0, type=float)
    parser.add_argument("--tree_levels", help='tree levels', required=True, type=int)
    parser.add_argument("--tree_branching", help='branching factor', required=True, type=int)
    parser.add_argument("--random_seed", help='pytorch random seed', default=0, type=int)
    # cmd = """--pick_per_level --tree_levels 6 --tree_branching 2 --n_epochs 5 --lr 0.1 --loss euc_cones_loss --embedding_dim 2 --neg_to_pos_ratio 5 --alpha 0.01 --experiment_name toy_graph --batch_size 10 --optimizer adam --experiment_dir ../exp/embed_toy/"""

    # args = parser.parse_args(cmd.split(' '))
    args = parser.parse_args()

    embed_toy_model(args)
