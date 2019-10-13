from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

import os
from network.experiment import Experiment
from network.evaluation import MultiLabelEvaluation, Evaluation, MultiLabelEvaluationSingleThresh, MultiLevelEvaluation

from data.db import ETHECLabelMap, ETHECDB

from network.loss import MultiLevelCELoss, MultiLabelSMLoss, LastLevelCELoss, MaskedCELoss, HierarchicalSoftmaxLoss, HierarchicalSoftmax

from PIL import Image
import numpy as np

import copy
import argparse
import json
import git
from tqdm import tqdm

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


def dataload(data_dirs, data_transforms, batch_size):
    print("Initializing Datasets and Dataloaders...")

    image_datasets = {x: datasets.ImageFolder(data_dirs[x], data_transforms[x]) for x in
                      ['train', 'val']}

    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       num_workers=4) for x in ['train', 'val']}
    return dataloaders_dict


class Finetuner(Experiment):
    def __init__(self, data_dir, data_transforms, classes, criterion, lr,
                 batch_size,
                 experiment_name,
                 experiment_dir='../exp/',
                 n_epochs=10,
                 eval_interval=2,
                 feature_extracting=True,
                 use_pretrained=True,
                 load_wt=False):

        model = models.alexnet(pretrained=use_pretrained)
        image_paths = {x: os.path.join(data_dir, x) for x in ['train', 'val']}
        data_loaders = dataload(image_paths, data_transforms, batch_size)

        Experiment.__init__(self, model, data_loaders, criterion, classes, experiment_name, n_epochs, eval_interval,
                            batch_size, experiment_dir, load_wt, Evaluation(experiment_dir, classes))

        self.lr = lr
        self.n_classes = len(classes)

        self.input_size = 224
        self.feature_extracting = feature_extracting

        self.set_parameter_requires_grad(feature_extracting)
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_features, self.n_classes)

        self.params_to_update = self.model.parameters()

        if self.feature_extracting:
            self.params_to_update = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.params_to_update.append(param)
                    print("Will update: {}".format(name))
        else:
            print("Fine-tuning")

        self.model.to(self.device)

    def train(self):
        self.run_model(optim.SGD(self.params_to_update, lr=self.lr, momentum=0.9))


class CIFAR10(Experiment):
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
                 optimizer_method='adam'):

        self.classes = labelmap.classes
        self.n_classes = labelmap.n_classes
        self.levels = labelmap.levels
        self.n_levels = len(self.levels)
        self.level_names = labelmap.level_names
        self.lr = lr
        self.batch_size = batch_size
        self.feature_extracting = feature_extracting
        self.optimal_thresholds = np.zeros(self.n_classes)
        self.optimizer_method = optimizer_method
        self.labelmap = labelmap

        if model_name == 'alexnet':
            model = models.alexnet(pretrained=use_pretrained)
        elif model_name == 'resnet18':
            model = models.resnet18(pretrained=use_pretrained)
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=use_pretrained)
        elif model_name == 'resnet101':
            model = models.resnet101(pretrained=use_pretrained)
        elif model_name == 'resnet152':
            model = models.resnet152(pretrained=use_pretrained)
        elif model_name == 'vgg':
            model = models.vgg11_bn(pretrained=use_pretrained)

        Experiment.__init__(self, model, data_loaders, criterion, self.classes, experiment_name, n_epochs,
                            eval_interval,
                            batch_size, experiment_dir, load_wt, evaluator)
        self.model_name = model_name

    def prepare_model(self, loading=False):
        self.dataset_length = {phase: len(self.dataloaders[phase].dataset) for phase in ['train', 'val', 'test']}

        self.set_parameter_requires_grad(self.feature_extracting)

        # modify last layers based on the model being used
        if not loading:
            if self.model_name in ['alexnet', 'vgg']:
                num_features = self.model.module.classifier[6].in_features
                if isinstance(self.criterion, LastLevelCELoss):
                    self.model.module.classifier[6] = nn.Linear(num_features, self.levels[-1])
                elif isinstance(self.criterion, HierarchicalSoftmaxLoss):
                    self.model.module.classifier[6] = HierarchicalSoftmax(labelmap=self.labelmap, input_size=num_features)
                else:
                    self.model.module.classifier[6] = nn.Linear(num_features, self.n_classes)
            elif 'resnet' in self.model_name:
                num_features = self.model.module.fc.in_features
                if isinstance(self.criterion, LastLevelCELoss):
                    self.model.module.fc = nn.Linear(num_features, self.levels[-1])
                elif isinstance(self.criterion, HierarchicalSoftmaxLoss):
                    self.model.module.fc = HierarchicalSoftmax(labelmap=self.labelmap, input_size=num_features)
                else:
                    self.model.module.fc = nn.Linear(num_features, self.n_classes)
        else:
            if self.model_name in ['alexnet', 'vgg']:
                num_features = self.model.module.module.classifier[6].in_features
                if isinstance(self.criterion, LastLevelCELoss):
                    self.model.module.module.classifier[6] = nn.Linear(num_features, self.levels[-1])
                elif isinstance(self.criterion, HierarchicalSoftmaxLoss):
                    self.model.module.module.classifier[6] = HierarchicalSoftmax(labelmap=self.labelmap, input_size=num_features)
                else:
                    self.model.module.module.classifier[6] = nn.Linear(num_features, self.n_classes)
            elif 'resnet' in self.model_name:
                num_features = self.model.module.module.fc.in_features
                if isinstance(self.criterion, LastLevelCELoss):
                    self.model.module.module.fc = nn.Linear(num_features, self.levels[-1])
                elif isinstance(self.criterion, HierarchicalSoftmaxLoss):
                    self.model.module.fc = HierarchicalSoftmax(labelmap=self.labelmap, input_size=num_features)
                else:
                    self.model.module.fc = nn.Linear(num_features, self.n_classes)

        self.n_train, self.n_val, self.n_test = torch.zeros(self.n_classes), torch.zeros(self.n_classes), \
                                                torch.zeros(self.n_classes)
        for phase in ['train', 'val', 'test']:
            for data_item in self.dataloaders[phase]:
                setattr(self, 'n_{}'.format(phase),
                        getattr(self, 'n_{}'.format(phase)) + torch.sum(data_item['labels'], 0))
        # print(self.n_train, torch.sum(self.n_train))
        # print(self.n_val, torch.sum(self.n_val))
        # print(self.n_test, torch.sum(self.n_test))

        self.samples_split = {'train': self.n_train, 'val': self.n_val, 'test': self.n_test}

        self.params_to_update = self.model.parameters()

        if self.feature_extracting:
            self.params_to_update = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.params_to_update.append(param)
                    print("Will update: {}".format(name))
        else:
            print("Fine-tuning")

    def pass_samples(self, phase, save_to_tensorboard=True):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        running_corrects = 0
        epoch_per_level_matches = np.zeros(self.n_levels)

        predicted_scores = np.zeros((self.dataset_length[phase], self.n_classes))
        correct_labels = np.zeros((self.dataset_length[phase], self.n_classes))

        # Iterate over data.
        for index, data_item in enumerate(tqdm(self.dataloaders[phase])):
            inputs, labels, level_labels = data_item['image'], data_item['labels'], data_item['level_labels']
            inputs = inputs.to(self.device)
            labels = labels.float().to(self.device)
            level_labels = level_labels.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                self.model = self.model.to(self.device)

                if isinstance(self.criterion, LastLevelCELoss) or isinstance(self.criterion, MaskedCELoss):
                    outputs = self.model(inputs)
                    if isinstance(self.criterion, MaskedCELoss):
                        outputs, loss = self.criterion(outputs, labels, level_labels, phase)
                    else:
                        outputs, loss = self.criterion(outputs, labels, level_labels)
                elif isinstance(self.criterion, HierarchicalSoftmaxLoss):
                    outputs, final_level_log_probs = self.model(inputs)
                    loss = self.criterion(final_level_log_probs, labels, level_labels)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels, level_labels)

                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # self.plot_grad_flow()
                    self.optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)

            outputs, labels = outputs.cpu().detach(), labels.cpu().detach()
            # exact matches
            n_exact_matches, per_level_matches = self.evaluate_hierarchical_matches(outputs, labels)
            running_corrects += n_exact_matches
            epoch_per_level_matches += per_level_matches

            predicted_scores[self.batch_size * index:min(self.batch_size * (index + 1),
                                                         self.dataset_length[phase]), :] = outputs.data
            correct_labels[self.batch_size * index:min(self.batch_size * (index + 1),
                                                       self.dataset_length[phase])] = labels.data

        # save predicted and correct labels to post-process
        if not save_to_tensorboard:
            np.save(os.path.join(self.log_dir, 'predicted_scores.npy'), predicted_scores)
            np.save(os.path.join(self.log_dir, 'correct_labels.npy'), correct_labels)

        metrics = self.eval.evaluate(predicted_scores, correct_labels, self.epoch, phase, save_to_tensorboard,
                                     self.samples_split)
        macro_f1, micro_f1, macro_p, micro_p, macro_r, micro_r = metrics['macro']['f1'], metrics['micro']['f1'], \
                                                                 metrics['macro']['precision'], \
                                                                 metrics['micro']['precision'], \
                                                                 metrics['macro']['recall'], \
                                                                 metrics['micro']['recall']
        if phase == 'eval':
            self.optimal_thresholds = self.eval.get_optimal_thresholds()

        epoch_loss = running_loss / self.dataset_length[phase]
        epoch_acc = running_corrects / self.dataset_length[phase]

        if save_to_tensorboard:
            self.writer.add_scalar('{}_loss'.format(phase), epoch_loss, self.epoch)
            self.writer.add_scalar('{}_accuracy'.format(phase), epoch_acc, self.epoch)
            self.writer.add_scalar('{}_micro_f1'.format(phase), micro_f1, self.epoch)
            self.writer.add_scalar('{}_macro_f1'.format(phase), macro_f1, self.epoch)
            self.writer.add_scalar('{}_micro_precision'.format(phase), micro_p, self.epoch)
            self.writer.add_scalar('{}_macro_precision'.format(phase), macro_p, self.epoch)
            self.writer.add_scalar('{}_micro_recall'.format(phase), micro_r, self.epoch)
            self.writer.add_scalar('{}_macro_recall'.format(phase), macro_r, self.epoch)

            for l_ix, level_matches in enumerate(epoch_per_level_matches.tolist()):
                self.writer.add_scalar('{}_{}_matches'.format(phase, self.level_names[l_ix]),
                                       level_matches / self.dataset_length[phase], self.epoch)

        print('{} Loss: {:.4f} Score: {:.4f}'.format(phase, epoch_loss, micro_f1))

        # deep copy the model
        if phase == 'val':
            if self.epoch % 10 == 0:
                self.save_model(epoch_loss)
            if micro_f1 >= self.best_score:
                self.best_score = micro_f1
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                self.save_model(epoch_loss, filename='best_model')

    def evaluate_hierarchical_matches(self, preds, labels):
        preds, labels = preds.cpu().detach().numpy(), labels.cpu().numpy()
        predicted_labels = preds > np.tile(self.optimal_thresholds, (labels.shape[0], 1))
        n_exact_matches = sum([1 for sample_ix in range(preds.shape[0])
                               if np.array_equal(preds[sample_ix, :], labels[sample_ix, :])])
        pred_labels_and_map = np.logical_and(np.array(predicted_labels), labels)

        level_matches = []
        level_start_ix = 0
        for level in self.levels:
            level_matches.append(np.sum(np.sum(pred_labels_and_map[:, level_start_ix:level_start_ix + level], axis=1)))
            level_start_ix += level

        return n_exact_matches, np.array(level_matches)

    def train(self):
        if self.optimizer_method == 'sgd':
            self.run_model(optim.SGD(self.params_to_update, lr=self.lr, momentum=0.9))
        elif self.optimizer_method == 'adam':
            self.run_model(optim.Adam(self.params_to_update, lr=self.lr))
        self.load_best_model()

    def test(self):
        if self.optimizer_method == 'sgd':
            self.run_model(optim.SGD(self.params_to_update, lr=self.lr, momentum=0.9))
        elif self.optimizer_method == 'adam':
            self.run_model(optim.Adam(self.params_to_update, lr=self.lr))
        self.load_best_model()

    def set_optimizer(self):
        if self.optimizer_method == 'sgd':
            self.optimizer = optim.SGD(self.params_to_update, lr=self.lr, momentum=0.9)
        elif self.optimizer_method == 'adam':
            self.optimizer = optim.Adam(self.params_to_update, lr=self.lr)

    def inference(self, data_item):
        self.model.eval()

        # Iterate over data.
        inputs, labels, level_labels = data_item['image'], data_item['labels'], data_item['level_labels']
        inputs = inputs.to(self.device)
        labels = labels.float().to(self.device)
        level_labels = level_labels.to(self.device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            self.model = self.model.to(self.device)

            if isinstance(self.criterion, LastLevelCELoss) or isinstance(self.criterion, MaskedCELoss):
                outputs = self.model(inputs)
                if isinstance(self.criterion, MaskedCELoss):
                    outputs, loss = self.criterion(outputs, labels, level_labels, phase='test')
                else:
                    outputs, loss = self.criterion(outputs, labels, level_labels)
            elif isinstance(self.criterion, HierarchicalSoftmaxLoss):
                outputs, final_level_log_probs = self.model(inputs)
                loss = self.criterion(final_level_log_probs, labels, level_labels)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels, level_labels)

            _, preds = torch.max(outputs, 1)

        outputs, labels = outputs.cpu().detach(), labels.cpu().detach()
        # print(outputs, labels)
        return outputs


class labelmap_CIFAR100:
    def __init__(self):
        self.family = {
            'aquatic mammals': 0,
            'fish': 1,
            'flowers': 2,
            'food containers': 3,
            'fruit and vegetables': 4,
            'household electrical device': 5,
            'household furniture': 6,
            'insects': 7,
            'large carnivores': 8,
            'large man-made outdoor things': 9,
            'large natural outdoor scenes': 10,
            'large omnivores and herbivores': 11,
            'medium-sized mammals': 12,
            'non-insect invertebrates': 13,
            'people': 14,
            'reptiles': 15,
            'small mammals': 16,
            'trees': 17,
            'vehicles 1': 18,
            'vehicles 2': 19
        }
        self.classes = {'apple': 20, 'aquarium_fish': 21, 'baby': 22, 'bear': 23, 'beaver': 24, 'bed': 25, 'bee': 26,
                        'beetle': 27,
                        'bicycle': 28, 'bottle': 29, 'bowl': 30, 'boy': 31, 'bridge': 32, 'bus': 33, 'butterfly': 34,
                        'camel': 35,
                        'can': 36, 'castle': 37, 'caterpillar': 38, 'cattle': 39, 'chair': 40, 'chimpanzee': 41,
                        'clock': 42,
                        'cloud': 43, 'cockroach': 44, 'couch': 45, 'crab': 46, 'crocodile': 47, 'cup': 48,
                        'dinosaur': 49,
                        'dolphin': 50, 'elephant': 51, 'flatfish': 52, 'forest': 53, 'fox': 54, 'girl': 55,
                        'hamster': 56,
                        'house': 57, 'kangaroo': 58, 'computer_keyboard': 59, 'lamp': 60, 'lawn_mower': 61,
                        'leopard': 62,
                        'lion': 63, 'lizard': 64, 'lobster': 65, 'man': 66, 'maple_tree': 67, 'motorcycle': 68,
                        'mountain': 69,
                        'mouse': 70, 'mushroom': 71, 'oak_tree': 72, 'orange': 73, 'orchid': 74, 'otter': 75,
                        'palm_tree': 76,
                        'pear': 77, 'pickup_truck': 78, 'pine_tree': 79, 'plain': 80, 'plate': 81, 'poppy': 82,
                        'porcupine': 83,
                        'possum': 84, 'rabbit': 85, 'raccoon': 86, 'ray': 87, 'road': 88, 'rocket': 89, 'rose': 90,
                        'sea': 91,
                        'seal': 92, 'shark': 93, 'shrew': 94, 'skunk': 95, 'skyscraper': 96, 'snail': 97, 'snake': 98,
                        'spider': 99, 'squirrel': 100, 'streetcar': 101, 'sunflower': 102, 'sweet_pepper': 103,
                        'table': 104,
                        'tank': 105, 'telephone': 106, 'television': 107, 'tiger': 108, 'tractor': 109, 'train': 110,
                        'trout': 111,
                        'tulip': 112, 'turtle': 113, 'wardrobe': 114, 'whale': 115, 'willow_tree': 116, 'wolf': 117,
                        'woman': 118, 'worm': 119
                        }
        self.classes_to_ix = {
            'aquatic mammals': 0,
            'fish': 1,
            'flowers': 2,
            'food containers': 3,
            'fruit and vegetables': 4,
            'household electrical device': 5,
            'household furniture': 6,
            'insects': 7,
            'large carnivores': 8,
            'large man-made outdoor things': 9,
            'large natural outdoor scenes': 10,
            'large omnivores and herbivores': 11,
            'medium-sized mammals': 12,
            'non-insect invertebrates': 13,
            'people': 14,
            'reptiles': 15,
            'small mammals': 16,
            'trees': 17,
            'vehicles 1': 18,
            'vehicles 2': 19,
            'apple': 20, 'aquarium_fish': 21, 'baby': 22, 'bear': 23, 'beaver': 24, 'bed': 25, 'bee': 26, 'beetle': 27,
            'bicycle': 28, 'bottle': 29, 'bowl': 30, 'boy': 31, 'bridge': 32, 'bus': 33, 'butterfly': 34, 'camel': 35,
            'can': 36, 'castle': 37, 'caterpillar': 38, 'cattle': 39, 'chair': 40, 'chimpanzee': 41, 'clock': 42,
            'cloud': 43, 'cockroach': 44, 'couch': 45, 'crab': 46, 'crocodile': 47, 'cup': 48, 'dinosaur': 49,
            'dolphin': 50, 'elephant': 51, 'flatfish': 52, 'forest': 53, 'fox': 54, 'girl': 55, 'hamster': 56,
            'house': 57, 'kangaroo': 58, 'computer_keyboard': 59, 'lamp': 60, 'lawn_mower': 61, 'leopard': 62,
            'lion': 63, 'lizard': 64, 'lobster': 65, 'man': 66, 'maple_tree': 67, 'motorcycle': 68, 'mountain': 69,
            'mouse': 70, 'mushroom': 71, 'oak_tree': 72, 'orange': 73, 'orchid': 74, 'otter': 75, 'palm_tree': 76,
            'pear': 77, 'pickup_truck': 78, 'pine_tree': 79, 'plain': 80, 'plate': 81, 'poppy': 82, 'porcupine': 83,
            'possum': 84, 'rabbit': 85, 'raccoon': 86, 'ray': 87, 'road': 88, 'rocket': 89, 'rose': 90, 'sea': 91,
            'seal': 92, 'shark': 93, 'shrew': 94, 'skunk': 95, 'skyscraper': 96, 'snail': 97, 'snake': 98,
            'spider': 99, 'squirrel': 100, 'streetcar': 101, 'sunflower': 102, 'sweet_pepper': 103, 'table': 104,
            'tank': 105, 'telephone': 106, 'television': 107, 'tiger': 108, 'tractor': 109, 'train': 110, 'trout': 111,
            'tulip': 112, 'turtle': 113, 'wardrobe': 114, 'whale': 115, 'willow_tree': 116, 'wolf': 117, 'woman': 118,
            'worm': 119
        }
        self.ix_to_classes = {self.classes_to_ix[k]: k for k in self.classes_to_ix}
        self.classes = [k for k in self.classes_to_ix]
        self.n_classes = 120
        self.levels = [20, 100]
        self.level_names = ['family', 'classes']
        self.map = {'beaver': ['aquatic mammals'], 'dolphin': ['aquatic mammals'], 'otter': ['aquatic mammals'],
                    'seal': ['aquatic mammals'], 'whale': ['aquatic mammals'], 'aquarium_fish': ['fish'],
                    'flatfish': ['fish'], 'ray': ['fish'], 'shark': ['fish'], 'trout': ['fish'], 'orchid': ['flowers'],
                    'poppy': ['flowers'], 'rose': ['flowers'], 'sunflower': ['flowers'], 'tulip': ['flowers'],
                    'bottle': ['food containers'], 'bowl': ['food containers'], 'can': ['food containers'],
                    'cup': ['food containers'], 'plate': ['food containers'], 'apple': ['fruit and vegetables'],
                    'mushroom': ['fruit and vegetables'], 'orange': ['fruit and vegetables'],
                    'pear': ['fruit and vegetables'], 'sweet_pepper': ['fruit and vegetables'],
                    'clock': ['household electrical device'], 'computer_keyboard': ['household electrical device'],
                    'lamp': ['household electrical device'], 'telephone': ['household electrical device'],
                    'television': ['household electrical device'], 'bed': ['household furniture'],
                    'chair': ['household furniture'], 'couch': ['household furniture'],
                    'table': ['household furniture'], 'wardrobe': ['household furniture'], 'bee': ['insects'],
                    'beetle': ['insects'], 'butterfly': ['insects'], 'caterpillar': ['insects'],
                    'cockroach': ['insects'], 'bear': ['large carnivores'], 'leopard': ['large carnivores'],
                    'lion': ['large carnivores'], 'tiger': ['large carnivores'], 'wolf': ['large carnivores'],
                    'bridge': ['large man-made outdoor things'], 'castle': ['large man-made outdoor things'],
                    'house': ['large man-made outdoor things'], 'road': ['large man-made outdoor things'],
                    'skyscraper': ['large man-made outdoor things'], 'cloud': ['large natural outdoor scenes'],
                    'forest': ['large natural outdoor scenes'], 'mountain': ['large natural outdoor scenes'],
                    'plain': ['large natural outdoor scenes'], 'sea': ['large natural outdoor scenes'],
                    'camel': ['large omnivores and herbivores'], 'cattle': ['large omnivores and herbivores'],
                    'chimpanzee': ['large omnivores and herbivores'], 'elephant': ['large omnivores and herbivores'],
                    'kangaroo': ['large omnivores and herbivores'], 'fox': ['medium-sized mammals'],
                    'porcupine': ['medium-sized mammals'], 'possum': ['medium-sized mammals'],
                    'raccoon': ['medium-sized mammals'], 'skunk': ['medium-sized mammals'],
                    'crab': ['non-insect invertebrates'], 'lobster': ['non-insect invertebrates'],
                    'snail': ['non-insect invertebrates'], 'spider': ['non-insect invertebrates'],
                    'worm': ['non-insect invertebrates'], 'baby': ['people'], 'boy': ['people'], 'girl': ['people'],
                    'man': ['people'], 'woman': ['people'], 'crocodile': ['reptiles'], 'dinosaur': ['reptiles'],
                    'lizard': ['reptiles'], 'snake': ['reptiles'], 'turtle': ['reptiles'], 'hamster': ['small mammals'],
                    'mouse': ['small mammals'], 'rabbit': ['small mammals'], 'shrew': ['small mammals'],
                    'squirrel': ['small mammals'], 'maple_tree': ['trees'], 'oak_tree': ['trees'],
                    'palm_tree': ['trees'], 'pine_tree': ['trees'], 'willow_tree': ['trees'], 'bicycle': ['vehicles 1'],
                    'bus': ['vehicles 1'], 'motorcycle': ['vehicles 1'], 'pickup_truck': ['vehicles 1'],
                    'train': ['vehicles 1'], 'lawn_mower': ['vehicles 2'], 'rocket': ['vehicles 2'],
                    'streetcar': ['vehicles 2'], 'tank': ['vehicles 2'], 'tractor': ['vehicles 2']
                    }
        self.child_of_family_ix = {0: [4, 30, 55, 72, 95], 1: [1, 32, 67, 73, 91], 2: [54, 62, 70, 82, 92],
                                   3: [9, 10, 16, 28, 61], 4: [0, 51, 53, 57, 83], 5: [22, 39, 40, 86, 87],
                                   6: [5, 20, 25, 84, 94], 7: [6, 7, 14, 18, 24], 8: [3, 42, 43, 88, 97],
                                   9: [12, 17, 37, 68, 76], 10: [23, 33, 49, 60, 71], 11: [15, 19, 21, 31, 38],
                                   12: [34, 63, 64, 66, 75], 13: [26, 45, 77, 79, 99], 14: [2, 11, 35, 46, 98],
                                   15: [27, 29, 44, 78, 93], 16: [36, 50, 65, 74, 80], 17: [47, 52, 56, 59, 96],
                                   18: [8, 13, 48, 58, 90], 19: [41, 69, 81, 85, 89]}

    def get_labels(self, class_index):
        class_index += 20
        family = self.map[self.ix_to_classes[class_index]][0]
        return [self.family[family], class_index]

    def labels_one_hot(self, class_index):
        indices = self.get_labels(class_index)
        retval = np.zeros(self.n_classes)
        retval[indices] = 1
        return retval

    def get_level_labels(self, class_index):
        level_labels = self.get_labels(class_index)
        return np.array([level_labels[0], level_labels[1] - self.levels[0]])


class labelmap_CIFAR10:
    def __init__(self):
        self.family = {'living': 0, 'non_living': 1}
        self.subfamily = {'non_land': 2, 'land': 3, 'vehicle': 4, 'craft': 5}
        self.classes_to_ix = {'living': 0, 'non_living': 1,
                              'non_land': 2, 'land': 3, 'vehicle': 4, 'craft': 5,
                              'plane': 6, 'car': 7, 'bird': 8, 'cat': 9, 'deer': 10, 'dog': 11, 'frog': 12, 'horse': 13,
                              'ship': 14, 'truck': 15}
        self.ix_to_classes = {self.classes_to_ix[k]: k for k in self.classes_to_ix}
        self.classes = [k for k in self.classes_to_ix]
        self.n_classes = 16
        self.levels = [2, 4, 10]
        self.level_names = ['family', 'subfamily', 'classes']
        self.map = {
            'plane': ['non_living', 'craft'],
            'car': ['non_living', 'vehicle'],
            'bird': ['living', 'non_land'],
            'cat': ['living', 'land'],
            'deer': ['living', 'land'],
            'dog': ['living', 'land'],
            'frog': ['living', 'non_land'],
            'horse': ['living', 'land'],
            'ship': ['non_living', 'craft'],
            'truck': ['non_living', 'vehicle']
        }
        self.child_of_family_ix, self.child_of_subfamily_ix = {}, {}
        self.child_of_family_ix = {0: [0, 1], 1: [2, 3]}
        self.child_of_subfamily_ix = {0: [2, 6], 1: [3, 4, 5, 7], 2: [1, 9], 3: [0, 8]}

    def get_labels(self, class_index):
        class_index += 6
        family, subfamily = self.map[self.ix_to_classes[class_index]]
        return [self.family[family], self.subfamily[subfamily], class_index]

    def labels_one_hot(self, class_index):
        indices = self.get_labels(class_index)
        retval = np.zeros(self.n_classes)
        retval[indices] = 1
        return retval

    def get_level_labels(self, class_index):
        level_labels = self.get_labels(class_index)
        return np.array(
            [level_labels[0], level_labels[1] - self.levels[0], level_labels[2] - (self.levels[0] + self.levels[1])])


class labelmap_CIFAR10_single:
    def __init__(self):
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')
        self.n_classes = 10


class Cifar10Hierarchical(torchvision.datasets.CIFAR10):
    def __init__(self, root, labelmap, train=True,
                 transform=None, target_transform=None,
                 download=False):
        self.labelmap = labelmap
        torchvision.datasets.CIFAR10.__init__(self, root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        #
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        multi_class_target = self.labelmap.labels_one_hot(target)

        return {'image': img, 'labels': torch.from_numpy(multi_class_target).float(), 'leaf_label': target,
                'level_labels': torch.from_numpy(self.labelmap.get_level_labels(target)).long()}


class Cifar100Hierarchical(torchvision.datasets.CIFAR100):
    def __init__(self, root, labelmap, train=True,
                 transform=None, target_transform=None,
                 download=False):
        self.labelmap = labelmap
        torchvision.datasets.CIFAR100.__init__(self, root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        multi_class_target = self.labelmap.labels_one_hot(target)

        return {'image': img, 'labels': torch.from_numpy(multi_class_target).float(), 'leaf_label': target,
                'level_labels': torch.from_numpy(self.labelmap.get_level_labels(target)).long()}


def train_cifar10(arguments):
    if not os.path.exists(os.path.join(arguments.experiment_dir, arguments.experiment_name)):
        os.makedirs(os.path.join(arguments.experiment_dir, arguments.experiment_name))
    args_dict = vars(arguments)
    repo = git.Repo(search_parent_directories=True)
    args_dict['commit_hash'] = repo.head.object.hexsha
    args_dict['branch'] = repo.active_branch.name
    with open(os.path.join(arguments.experiment_dir, arguments.experiment_name, 'config_params.txt'), 'w') as file:
        file.write(json.dumps(args_dict, indent=4))

    print('Config parameters for this run are:\n{}'.format(json.dumps(vars(arguments), indent=4)))

    input_size = 224
    data_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    labelmap = labelmap_CIFAR10()
    if arguments.dataset == 'cifar100':
        labelmap = labelmap_CIFAR100()

    batch_size = arguments.batch_size
    n_workers = arguments.n_workers

    if arguments.debug:
        print("== Running in DEBUG mode!")
        if arguments.dataset == 'cifar100':
            trainset = Cifar100Hierarchical(root='../database', labelmap=labelmap, train=False,
                                            download=True, transform=data_transforms)
        else:
            trainset = Cifar10Hierarchical(root='../database', labelmap=labelmap, train=False,
                                           download=True, transform=data_transforms)
        trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, list(range(100))),
                                                  batch_size=batch_size,
                                                  shuffle=True, num_workers=n_workers)

        valloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, list(range(100, 200))),
                                                batch_size=batch_size,
                                                shuffle=True, num_workers=n_workers)

        testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, list(range(200, 300))),
                                                 batch_size=batch_size,
                                                 shuffle=False, num_workers=n_workers)

        data_loaders = {'train': trainloader, 'val': valloader, 'test': testloader}

    else:
        if arguments.dataset == 'cifar100':
            trainset = Cifar100Hierarchical(root='../database', labelmap=labelmap, train=True,
                                            download=True, transform=data_transforms)
            testset = Cifar100Hierarchical(root='../database', labelmap=labelmap, train=False,
                                           download=True, transform=data_transforms)
        else:
            trainset = Cifar10Hierarchical(root='../database', labelmap=labelmap, train=True,
                                           download=True, transform=data_transforms)
            testset = Cifar10Hierarchical(root='../database', labelmap=labelmap, train=False,
                                          download=True, transform=data_transforms)
        # split the dataset into 80:10:10
        train_indices_from_train, val_indices_from_train, val_indices_from_test, test_indices_from_test = \
            cifar10_set_indices(trainset, testset, labelmap)

        trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, train_indices_from_train),
                                                  batch_size=batch_size,
                                                  shuffle=True, num_workers=n_workers)

        evalset_from_train = torch.utils.data.Subset(trainset, val_indices_from_train)
        evalset_from_test = torch.utils.data.Subset(testset, val_indices_from_test)
        valloader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([evalset_from_train, evalset_from_test]),
                                                batch_size=batch_size,
                                                shuffle=True, num_workers=n_workers)

        testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(testset, test_indices_from_test),
                                                 batch_size=batch_size,
                                                 shuffle=False, num_workers=n_workers)

        data_loaders = {'train': trainloader, 'val': valloader, 'test': testloader}

    weight = None
    if arguments.class_weights:
        n_train = torch.zeros(labelmap.n_classes)
        for data_item in data_loaders['train']:
            n_train += torch.sum(data_item['labels'], 0)
        weight = 1.0 / n_train

    eval_type = MultiLabelEvaluation(os.path.join(arguments.experiment_dir, arguments.experiment_name), labelmap)
    if arguments.evaluator == 'MLST':
        eval_type = MultiLabelEvaluationSingleThresh(os.path.join(arguments.experiment_dir, arguments.experiment_name),
                                                     labelmap)

    use_criterion = None
    if arguments.loss == 'multi_label':
        use_criterion = MultiLabelSMLoss(weight=weight)
    elif arguments.loss == 'multi_level':
        use_criterion = MultiLevelCELoss(labelmap=labelmap, weight=weight, level_weights=arguments.level_weights)
        eval_type = MultiLevelEvaluation(os.path.join(arguments.experiment_dir, arguments.experiment_name), labelmap)
    elif arguments.loss == 'last_level':
        use_criterion = LastLevelCELoss(labelmap=labelmap, weight=weight, level_weights=arguments.level_weights)
        eval_type = MultiLevelEvaluation(os.path.join(arguments.experiment_dir, arguments.experiment_name), labelmap)

    cifar_trainer = CIFAR10(data_loaders=data_loaders, labelmap=labelmap,
                            criterion=use_criterion,
                            lr=arguments.lr,
                            batch_size=batch_size, evaluator=eval_type,
                            experiment_name=arguments.experiment_name,  # 'cifar_test_ft_multi',
                            experiment_dir=arguments.experiment_dir,
                            eval_interval=arguments.eval_interval,
                            n_epochs=arguments.n_epochs,
                            feature_extracting=arguments.freeze_weights,
                            use_pretrained=True,
                            load_wt=False,
                            model_name=arguments.model,
                            optimizer_method=arguments.optimizer_method)
    cifar_trainer.prepare_model()
    if arguments.set_mode == 'train':
        cifar_trainer.train()
    elif arguments.set_mode == 'test':
        cifar_trainer.test()


def cifar10_set_indices(trainset, testset, labelmap=labelmap_CIFAR10()):
    indices = {d_set_name: {label_ix: [] for label_ix in range(len(labelmap.map))} for d_set_name in ['train', 'val']}
    for d_set, d_set_name in zip([trainset, testset], ['train', 'val']):
        for i in range(len(d_set)):
            indices[d_set_name][d_set[i]['leaf_label']].append(i)

    train_indices_from_train = []
    for label_ix in range(len(indices['train'])):
        train_indices_from_train += indices['train'][label_ix][:4800]

    val_indices_from_train = []
    val_indices_from_test = []
    for label_ix in range(len(indices['train'])):
        val_indices_from_train += indices['train'][label_ix][-200:]
    for label_ix in range(len(indices['val'])):
        val_indices_from_test += indices['val'][label_ix][:400]

    test_indices_from_test = []
    for label_ix in range(len(indices['val'])):
        test_indices_from_test += indices['val'][label_ix][-600:]

    print('Train set has: {}'.format(len(set(train_indices_from_train))))
    print('Val set has: {} + {}'.format(len(set(val_indices_from_train)), len(set(val_indices_from_test))))
    print('Test set has: {}'.format(len(set(test_indices_from_test))))

    return train_indices_from_train, val_indices_from_train, val_indices_from_test, test_indices_from_test


def cifar10_ind_exp():
    input_size = 224
    data_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    lmap = labelmap_CIFAR10()
    batch_size = 8

    trainset = Cifar10Hierarchical(root='../database', labelmap=lmap, train=True,
                                   download=True, transform=data_transforms)
    testset = Cifar10Hierarchical(root='../database', labelmap=lmap, train=False,
                                  download=True, transform=data_transforms)

    # split the dataset into 80:10:10
    train_indices_from_train, val_indices_from_train, val_indices_from_test, test_indices_from_test = \
        cifar10_set_indices(trainset, testset, lmap)

    torch.utils.data.ConcatDataset(datasets)

    trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, train_indices_from_train),
                                              batch_size=batch_size,
                                              shuffle=True, num_workers=4)

    evalset_from_train = torch.utils.data.Subset(trainset, val_indices_from_train)
    evalset_from_test = torch.utils.data.Subset(testset, val_indices_from_test)
    evalloader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([evalset_from_train, evalset_from_test]),
                                             batch_size=batch_size,
                                             shuffle=True, num_workers=4)

    testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(testset, test_indices_from_test),
                                             batch_size=batch_size,
                                             shuffle=False, num_workers=4)


def train_cifar10_single():
    input_size = 224
    data_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    lmap = labelmap_CIFAR10()
    batch_size = 8

    trainset = torchvision.datasets.CIFAR10(root='../database', train=False, download=True, transform=data_transforms)
    trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, list(range(1000))),
                                              batch_size=batch_size,
                                              shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='../database', train=False, download=True, transform=data_transforms)
    testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(testset, list(range(1000, 2000))),
                                             batch_size=batch_size,
                                             shuffle=False, num_workers=4)

    data_loaders = {'train': trainloader, 'val': testloader}

    cifar_trainer = CIFAR10(data_loaders=data_loaders, labelmap=lmap,
                            criterion=nn.CrossEntropyLoss(),
                            lr=0.001,
                            batch_size=batch_size,
                            experiment_name='cifar_test_ft',
                            experiment_dir='../exp/',
                            eval_interval=2,
                            n_epochs=10,
                            feature_extracting=True,
                            use_pretrained=True,
                            load_wt=True)


def train_alexnet_binary():
    input_size = 224
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    data_dir = '../database/hymenoptera_data'

    Finetuner(data_dir=data_dir, data_transforms=data_transforms, classes=('spider', 'bee'),
              criterion=nn.CrossEntropyLoss(),
              lr=0.001,
              batch_size=8,
              experiment_name='alexnet_ft',
              n_epochs=2,
              load_wt=False).train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", help='Use DEBUG mode.', action='store_true')
    parser.add_argument("--lr", help='Input learning rate.', type=float, default=0.001)
    parser.add_argument("--batch_size", help='Batch size.', type=int, default=8)
    parser.add_argument("--evaluator",
                        help='Evaluator type. If using `multi_level` option for --loss then is overidden.', type=str,
                        default='ML')
    parser.add_argument("--experiment_name", help='Experiment name.', type=str, required=True)
    parser.add_argument("--experiment_dir", help='Experiment directory.', type=str, required=True)
    parser.add_argument("--n_epochs", help='Number of epochs to run training for.', type=int, required=True)
    parser.add_argument("--n_workers", help='Number of workers.', type=int, default=4)
    parser.add_argument("--optimizer_method", help='[adam, sgd]', type=str, default='adam')
    parser.add_argument("--eval_interval", help='Evaluate model every N intervals.', type=int, default=1)
    parser.add_argument("--resume", help='Continue training from last checkpoint.', action='store_true')
    parser.add_argument("--model", help='NN model to use.', type=str, required=True)
    parser.add_argument("--loss", help='Loss function to use.', type=str, required=True)
    parser.add_argument("--dataset", help='Use cifar10 or cifar100 dataset', type=str, required=True)
    parser.add_argument("--class_weights", help='Re-weigh the loss function based on inverse class freq.',
                        action='store_true')
    parser.add_argument("--freeze_weights", help='This flag fine tunes only the last layer.', action='store_true')
    parser.add_argument("--set_mode", help='If use training or testing mode (loads best model).', type=str,
                        required=True)
    parser.add_argument("--level_weights", help='List of weights for each level', nargs=3, default=None, type=float)
    args = parser.parse_args()

    train_cifar10(args)
