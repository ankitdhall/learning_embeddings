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

import copy
import argparse
import json
import git

import torch
from torch import nn
from data.db import ETHECLabelMap, ETHECLabelMapMergedSmall

from network.finetuner import CIFAR10
import numpy as np
import random
random.seed(0)

import networkx as nx
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import torch.nn.functional as F


class ETHECHierarchy(torch.utils.data.Dataset):
    """
    Creates a PyTorch dataset for order-embeddings, without images.
    """

    def __init__(self, graph):
        """
        Constructor.
        :param graph: <networkx.DiGraph> Graph to be used.
        """
        self.G = graph
        self.num_edges = graph.size()
        self.edge_list = [e for e in graph.edges()]

    def __getitem__(self, item):
        """
        Fetch an entry based on index.
        :param item: <int> Index to fetch.
        :return: <dict> Consumable object (see schema.md)
                {'from': <int>, 'to': <int>}
        """
        return {'from': self.edge_list[item][0], 'to': self.edge_list[item][1]}

    def __len__(self):
        """
        Return number of entries in the database.
        :return: <int> length of database
        """
        return self.num_edges

    def are_connected(self, from_ix, to_ix):
        """
        Check if an edge exists in the graph
        :param to_ix: <int>
        :param from_ix: <int>
        :return: True if (from_ix, to_ix) exists in self.G
        """
        self.G.has_edge(from_ix, to_ix)


class Embedder(nn.Module):
    def __init__(self, embedding_dim, labelmap):
        super(Embedder, self).__init__()
        self.labelmap = labelmap
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(self.labelmap.n_classes, self.embedding_dim)
        print('Embeds {} objects'.format(self.labelmap.n_classes))

    def forward(self, inputs):
        embeds = self.embeddings(inputs)#.view((1, -1))
        return torch.abs(embeds)

class EmbeddingMetrics:
    def __init__(self, e_for_u_v_positive, e_for_u_v_negative, threshold, phase):
        self.e_for_u_v_positive = e_for_u_v_positive
        self.e_for_u_v_negative = e_for_u_v_negative
        self.threshold = threshold
        self.phase = phase

    def calculate_metrics(self):
        if self.e_for_u_v_negative is not None or self.phase != 'test':
            possible_thresholds = np.unique(np.concatenate((self.e_for_u_v_positive, self.e_for_u_v_negative), axis=None))
            best_acc, best_threshold = 0.0, 0.0
            for t_id in range(possible_thresholds.shape[0]):
                correct_positives = np.sum(self.e_for_u_v_positive <= possible_thresholds[t_id])
                correct_negatives = np.sum(self.e_for_u_v_negative > possible_thresholds[t_id])
                accuracy = (correct_positives+correct_negatives)/(self.e_for_u_v_positive.shape[0]+self.e_for_u_v_negative.shape[0])
                if accuracy > best_acc:
                    best_acc = accuracy
                    best_threshold = possible_thresholds[t_id]
            if self.phase == 'val':
                return best_acc, best_threshold
            elif self.phase == 'train':
                return best_acc, self.threshold
        else:
            correct_positives = np.sum(self.e_for_u_v_positive <= self.threshold)
            accuracy = correct_positives / self.e_for_u_v_positive.shape[0]
            return accuracy, self.threshold



class OrderEmbedding(CIFAR10):
    def __init__(self, data_loaders, labelmap, criterion, lr,
                 batch_size,
                 evaluator,
                 experiment_name,
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
        CIFAR10.__init__(self, data_loaders, labelmap, criterion, lr, batch_size, evaluator, experiment_name,
                         experiment_dir, n_epochs, eval_interval, feature_extracting, use_pretrained, load_wt,
                         model_name, optimizer_method)

        self.classes = labelmap.classes
        self.n_classes = labelmap.n_classes
        self.levels = labelmap.levels
        self.n_levels = len(self.levels)
        self.level_names = labelmap.level_names
        self.lr = lr
        self.batch_size = batch_size
        self.feature_extracting = feature_extracting
        self.optimizer_method = optimizer_method
        self.optimal_threshold = 0.0

        self.embedding_dim = 10

        self.model = Embedder(embedding_dim=10, labelmap=labelmap)
        self.labelmap = labelmap

        self.G, self.G_train, self.G_val, self.G_test = nx.DiGraph(), nx.DiGraph(), nx.DiGraph(), nx.DiGraph()
        for index, data_item in enumerate(self.dataloaders['train']):
            inputs, labels, level_labels = data_item['image'], data_item['labels'], data_item['level_labels']
            for level_id in range(len(self.labelmap.levels)-1):
                for sample_id in range(level_labels.shape[0]):
                    self.G.add_edge(level_labels[sample_id, level_id].item()+self.labelmap.level_start[level_id],
                                    level_labels[sample_id, level_id+1].item()+self.labelmap.level_start[level_id+1])

        self.G_tc = nx.transitive_closure(self.G)
        self.create_splits()
        self.neg_to_pos_ratio = 2

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
        # prepare test and val sub-graphs
        total_number_of_edges = self.G_tc.size()
        print('Has {} edges in transitive closure'.format(total_number_of_edges))
        edges_for_test_val = int(0.05*total_number_of_edges)

        # create val graph
        remove_edges = random.sample(range(total_number_of_edges), k=edges_for_test_val)
        edges_in_tc = [e for e in self.G_tc.edges()]
        for edge_ix in remove_edges:
            self.G_val.add_edge(edges_in_tc[edge_ix][0], edges_in_tc[edge_ix][1])
        for edge_ix in remove_edges:
            self.G_tc.remove_edge(edges_in_tc[edge_ix][0], edges_in_tc[edge_ix][1])

        # create test graph
        total_number_of_edges = self.G_tc.size()
        remove_edges = random.sample(range(total_number_of_edges), k=edges_for_test_val)
        edges_in_tc = [e for e in self.G_tc.edges()]
        for edge_ix in remove_edges:
            self.G_test.add_edge(edges_in_tc[edge_ix][0], edges_in_tc[edge_ix][1])
        for edge_ix in remove_edges:
            self.G_tc.remove_edge(edges_in_tc[edge_ix][0], edges_in_tc[edge_ix][1])

        print('Edges in train: {}, val: {}, test: {}'.format(self.G_tc.size(), self.G_val.size(), self.G_test.size()))
        self.G_train = self.G_tc

        # create dataloaders
        train_set = ETHECHierarchy(self.G_train)
        val_set = ETHECHierarchy(self.G_val)
        test_set = ETHECHierarchy(self.G_test)
        trainloader = torch.utils.data.DataLoader(train_set,
                                                  batch_size=self.batch_size,
                                                  num_workers=16,
                                                  shuffle=True)
        valloader = torch.utils.data.DataLoader(val_set,
                                                batch_size=self.batch_size,
                                                num_workers=16,
                                                shuffle=True)
        testloader = torch.utils.data.DataLoader(test_set,
                                                batch_size=self.batch_size,
                                                num_workers=16,
                                                shuffle=True)
        self.dataloaders = {'train': trainloader, 'val': valloader, 'test': testloader}
        self.graphs = {'train': self.G_train, 'val': self.G_val, 'test': self.G_test}
        self.dataset_length = {phase: len(self.dataloaders[phase].dataset) for phase in ['train', 'val', 'test']}

    def pass_samples(self, phase, save_to_tensorboard=True):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0

        predicted_from_embeddings = np.zeros((self.dataset_length[phase], self.embedding_dim))
        predicted_to_embeddings = np.zeros((self.dataset_length[phase], self.embedding_dim))
        e_positive, e_negative = np.zeros((self.dataset_length[phase])), np.zeros((self.neg_to_pos_ratio*self.dataset_length[phase]))
        correct_labels = np.zeros((self.dataset_length[phase], self.n_classes))

        # Iterate over data.
        for index, data_item in enumerate(self.dataloaders[phase]):
            inputs_from, inputs_to = data_item['from'], data_item['to']
            inputs_from = inputs_from.to(self.device)
            inputs_to = inputs_to.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                self.model = self.model.to(self.device)

                outputs_from, outputs_to, loss, e_for_u_v_positive, e_for_u_v_negative =\
                    self.criterion(self.model, inputs_from, inputs_to, phase, self.graphs[phase], self.neg_to_pos_ratio)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()

            # statistics
            running_loss += loss.item()

            outputs_from, outputs_to = outputs_from.cpu().detach(), outputs_to.cpu().detach()

            predicted_from_embeddings[self.batch_size * index:min(self.batch_size * (index + 1),
                                                             self.dataset_length[phase]), :] = outputs_from.data
            predicted_to_embeddings[self.batch_size * index:min(self.batch_size * (index + 1),
                                                                  self.dataset_length[phase]), :] = outputs_to.data
            e_positive[self.batch_size * index:min(self.batch_size * (index + 1),
                                                   self.dataset_length[phase])] = e_for_u_v_positive.data
            if phase != 'test':
                e_negative[self.neg_to_pos_ratio * self.batch_size * index:min(self.neg_to_pos_ratio*self.batch_size * (index + 1),
                                                       self.neg_to_pos_ratio*self.dataset_length[phase])] = e_for_u_v_negative.data
            else:
                e_negative = None

        metrics = EmbeddingMetrics(e_positive, e_negative, self.optimal_threshold, phase)

        accuracy, threshold = metrics.calculate_metrics()
        if phase == 'val':
            self.optimal_threshold = threshold
        epoch_loss = running_loss / self.dataset_length[phase]

        if save_to_tensorboard:
            self.writer.add_scalar('{}_loss'.format(phase), epoch_loss, self.epoch)
            self.writer.add_scalar('{}_accuracy'.format(phase), accuracy, self.epoch)
            self.writer.add_scalar('{}_thresh'.format(phase), self.optimal_threshold, self.epoch)

        # print('{} Loss: {:.4f} Score: {:.4f}'.format(phase, epoch_loss, micro_f1))
        print('{} Loss: {:.4f}, Accuracy: {:.4f}'.format(phase, epoch_loss, accuracy))

        # deep copy the model
        if phase == 'val':
            if self.epoch % 10 == 0:
                self.save_model(epoch_loss)
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                self.save_model(epoch_loss, filename='best_model')

    def save_model(self, loss, filename=None):
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'optimal_threshold': self.optimal_threshold
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
        print('Successfully loaded model epoch {} from {}'.format(self.epoch, self.path_to_save_model))


class OrderEmbeddingLoss(torch.nn.Module):
    def __init__(self, labelmap, alpha=1.0):
        print('Using order-embedding loss!')
        torch.nn.Module.__init__(self)
        self.labelmap = labelmap
        self.alpha = alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def E_operator(x, y):
        # print('xshape {} y shape {}'.format(x.shape, y.shape))
        # print('eop {}'.format(torch.sum(torch.clamp(y-x, min=0.0)**2, dim=1).shape))
        return torch.sum(torch.clamp(y-x, min=0.0)**2, dim=1)

    def positive_pair(self, x, y):
        # print('ppair shape {}'.format(self.E_operator(x, y).shape))
        return self.E_operator(x, y)

    def negative_pair(self, x, y):
        # print('npair shape {}'.format(torch.clamp(self.alpha-self.E_operator(x, y), min=0.0).shape))
        return torch.clamp(self.alpha-self.E_operator(x, y), min=0.0), self.E_operator(x, y)

    def forward(self, model, inputs_from, inputs_to, phase, G, neg_to_pos_ratio):
        loss = 0.0
        e_for_u_v_positive, e_for_u_v_negative = torch.tensor([]), torch.tensor([])
        predicted_from_embeddings = model(inputs_from)
        predicted_to_embeddings = model(inputs_to)

        reverse_G = nx.reverse(G)

        nodes_in_graph = set(list(G))

        e_for_u_v_positive = self.positive_pair(predicted_from_embeddings, predicted_to_embeddings)
        loss += torch.mean(e_for_u_v_positive)
        if phase != 'test':
            for _ in range(neg_to_pos_ratio):
                negative_from, negative_to = torch.zeros_like(inputs_from), torch.zeros_like(inputs_to)
                for sample_id in range(inputs_from.shape[0]):
                    if sample_id % 2 == 0:
                        list_of_edges_from_ui = [v for u, v in list(G.edges(inputs_from[sample_id].item()))]
                        corrupted_ix = random.choice(list(nodes_in_graph - set(list_of_edges_from_ui)))
                        negative_from[sample_id], negative_to[sample_id] = inputs_from[sample_id], corrupted_ix
                    else:
                        list_of_edges_to_vi = [v for u, v in list(reverse_G.edges(inputs_to[sample_id].item()))]
                        corrupted_ix = random.choice(list(nodes_in_graph - set(list_of_edges_to_vi)))
                        negative_from[sample_id], negative_to[sample_id] = corrupted_ix, inputs_to[sample_id]

                negative_from_embeddings, negative_to_embeddings = model(negative_from), model(negative_to)
                neg_term, current_e_for_u_v = self.negative_pair(negative_from_embeddings, negative_to_embeddings)
                e_for_u_v_negative = torch.cat((e_for_u_v_negative, current_e_for_u_v))
                loss += torch.mean(neg_term)

        return predicted_from_embeddings, predicted_to_embeddings, loss, e_for_u_v_positive, e_for_u_v_negative


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
                                                # RandomCrop((input_size, input_size)),
                                                transforms.RandomHorizontalFlip(),
                                                # ColorJitter(brightness=0.2, contrast=0.2),
                                                transforms.ToTensor(),
                                                # transforms.Normalize(mean=(143.2341, 162.8151, 177.2185),
                                                #                      std=(66.7762, 59.2524, 51.5077))
                                                ])
    val_test_data_transforms = transforms.Compose([transforms.ToPILImage(),
                                                   transforms.Resize((input_size, input_size)),
                                                   transforms.ToTensor(),
                                                   # transforms.Normalize(mean=(143.2341, 162.8151, 177.2185),
                                                   #                      std=(66.7762, 59.2524, 51.5077))
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
                            labelmap=labelmap, transform=train_data_transforms)
        val_set = ETHECDB(path_to_json='../database/ETHEC/val.json',
                          path_to_images=arguments.image_dir,
                          labelmap=labelmap, transform=val_test_data_transforms)
        test_set = ETHECDB(path_to_json='../database/ETHEC/test.json',
                           path_to_images=arguments.image_dir,
                           labelmap=labelmap, transform=val_test_data_transforms)
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
                                       labelmap=labelmap, transform=train_data_transforms)
        val_set = ETHECDBMergedSmall(path_to_json='../database/ETHEC/val.json',
                                     path_to_images=arguments.image_dir,
                                     labelmap=labelmap, transform=val_test_data_transforms)
        test_set = ETHECDBMergedSmall(path_to_json='../database/ETHEC/test.json',
                                      path_to_images=arguments.image_dir,
                                      labelmap=labelmap, transform=val_test_data_transforms)

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

    else:
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
    if arguments.loss == 'embedding_loss':
        use_criterion = OrderEmbeddingLoss(labelmap=labelmap)
    else:
        print("== Invalid --loss argument")

    oe = OrderEmbedding(data_loaders=data_loaders, labelmap=labelmap,
                        criterion=use_criterion,
                        lr=arguments.lr,
                        batch_size=batch_size, evaluator=eval_type,
                        experiment_name=arguments.experiment_name,  # 'cifar_test_ft_multi',
                        experiment_dir=arguments.experiment_dir,
                        eval_interval=arguments.eval_interval,
                        n_epochs=arguments.n_epochs,
                        feature_extracting=arguments.freeze_weights,
                        use_pretrained=True,
                        load_wt=arguments.resume,
                        model_name=arguments.model,
                        optimizer_method=arguments.optimizer_method,
                        use_grayscale=arguments.use_grayscale)
    oe.prepare_model()
    if arguments.set_mode == 'train':
        oe.train()
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
    parser.add_argument("--resume", help='Continue training from last checkpoint.', action='store_true')
    parser.add_argument("--optimizer_method", help='[adam, sgd]', type=str, default='adam')
    parser.add_argument("--merged", help='Use dataset which has genus and species combined.', action='store_true')
    parser.add_argument("--weight_strategy", help='Use inverse freq or inverse sqrt freq. ["inv", "inv_sqrt"]',
                        type=str, default='inv')
    parser.add_argument("--model", help='NN model to use.', type=str, required=True)
    parser.add_argument("--loss",
                        help='Loss function to use. [multi_label, multi_level, last_level, masked_loss, hsoftmax]',
                        type=str, required=True)
    parser.add_argument("--use_grayscale", help='Use grayscale images.', action='store_true')
    parser.add_argument("--class_weights", help='Re-weigh the loss function based on inverse class freq.',
                        action='store_true')
    parser.add_argument("--freeze_weights", help='This flag fine tunes only the last layer.', action='store_true')
    parser.add_argument("--set_mode", help='If use training or testing mode (loads best model).', type=str,
                        required=True)
    parser.add_argument("--level_weights", help='List of weights for each level', nargs=4, default=None, type=float)
    args = parser.parse_args()

    order_embedding_train_model(args)
