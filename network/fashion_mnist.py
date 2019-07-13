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
from network.loss import MultiLevelCELoss, MultiLabelSMLoss, LastLevelCELoss
from network.finetuner import CIFAR10

from PIL import Image
import numpy as np

import copy
import argparse
import json
import git


class FMNIST(CIFAR10):
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

        CIFAR10.__init__(self, data_loaders, labelmap, criterion, lr, batch_size, evaluator, experiment_name,
                         experiment_dir, n_epochs, eval_interval, feature_extracting, use_pretrained,
                         load_wt, model_name, optimizer_method)

        if model_name in ['alexnet', 'vgg']:
            o_channels = self.model.features[0].out_channels
            k_size = self.model.features[0].kernel_size
            stride = self.model.features[0].stride
            pad = self.model.features[0].padding
            dil = self.model.features[0].dilation
            self.model.features[0] = nn.Conv2d(1, o_channels, kernel_size=k_size, stride=stride, padding=pad,
                                               dilation=dil)
        elif 'resnet' in model_name:
            o_channels = self.model.conv1.out_channels
            k_size = self.model.conv1.kernel_size
            stride = self.model.conv1.stride
            pad = self.model.conv1.padding
            dil = self.model.conv1.dilation
            self.model.conv1 = nn.Conv2d(1, o_channels, kernel_size=k_size, stride=stride, padding=pad, dilation=dil)

        self.model = nn.DataParallel(self.model)


def train_FMNIST(arguments):
    if not os.path.exists(os.path.join(arguments.experiment_dir, arguments.experiment_name)):
        os.makedirs(os.path.join(arguments.experiment_dir, arguments.experiment_name))
    args_dict = vars(args)
    repo = git.Repo(search_parent_directories=True)
    args_dict['commit_hash'] = repo.head.object.hexsha
    args_dict['branch'] = repo.active_branch.name
    with open(os.path.join(arguments.experiment_dir, arguments.experiment_name, 'config_params.txt'), 'w') as file:
        file.write(json.dumps(args_dict, indent=4))

    print('Config parameters for this run are:\n{}'.format(json.dumps(vars(args), indent=4)))

    input_size = 224
    data_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(0.5, 0.5)
        ])

    labelmap = labelmap_FMNIST()
    batch_size = arguments.batch_size
    n_workers = arguments.n_workers

    if arguments.debug:
        print("== Running in DEBUG mode!")
        trainset = FMNISTHierarchical(root='../database', labelmap=labelmap, train=False,
                                       download=True, transform=data_transforms)
        trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, list(range(100))), batch_size=batch_size,
                                                  shuffle=True, num_workers=n_workers)

        valloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, list(range(100, 200))),
                                                batch_size=batch_size,
                                                shuffle=True, num_workers=n_workers)

        testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, list(range(200, 300))),
                                                 batch_size=batch_size,
                                                 shuffle=False, num_workers=n_workers)

        data_loaders = {'train': trainloader, 'val': valloader, 'test': testloader}

    else:
        trainset = FMNISTHierarchical(root='../database', labelmap=labelmap, train=True,
                                       download=True, transform=data_transforms)
        testset = FMNISTHierarchical(root='../database', labelmap=labelmap, train=False,
                                      download=True, transform=data_transforms)

        # split the dataset into 80:10:10
        train_indices_from_train, val_indices_from_train, val_indices_from_test, test_indices_from_test = \
            FMNIST_set_indices(trainset, testset, labelmap)

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
        eval_type = MultiLabelEvaluationSingleThresh(os.path.join(arguments.experiment_dir, arguments.experiment_name), labelmap)

    use_criterion = None
    if arguments.loss == 'multi_label':
        use_criterion = MultiLabelSMLoss(weight=weight)
    elif arguments.loss == 'multi_level':
        use_criterion = MultiLevelCELoss(labelmap=labelmap, weight=weight, level_weights=arguments.level_weights)
        eval_type = MultiLevelEvaluation(os.path.join(arguments.experiment_dir, arguments.experiment_name), labelmap)
    elif arguments.loss == 'last_level':
        use_criterion = LastLevelCELoss(labelmap=labelmap, weight=weight, level_weights=arguments.level_weights)
        eval_type = MultiLevelEvaluation(os.path.join(arguments.experiment_dir, arguments.experiment_name), labelmap)

    FMNIST_trainer = FMNIST(data_loaders=data_loaders, labelmap=labelmap,
                            criterion=use_criterion,
                            lr=arguments.lr,
                            batch_size=batch_size, evaluator=eval_type,
                            experiment_name=arguments.experiment_name, # 'cifar_test_ft_multi',
                            experiment_dir=arguments.experiment_dir,
                            eval_interval=arguments.eval_interval,
                            n_epochs=arguments.n_epochs,
                            feature_extracting=arguments.freeze_weights,
                            use_pretrained=True,
                            load_wt=False,
                            model_name=arguments.model,
                            optimizer_method=arguments.optimizer_method)
    FMNIST_trainer.prepare_model()
    if arguments.set_mode == 'train':
        FMNIST_trainer.train()
    elif arguments.set_mode == 'test':
        FMNIST_trainer.test()


class labelmap_FMNIST:
    def __init__(self):
        self.classes_to_ix = {'tops': 0, 'bottoms': 1, 'accessories': 2, 'footwear': 3,
                              'T-shirt_top': 4, 'Trouser': 5, 'Pullover': 6, 'Dress': 7, 'Coat': 8, 'Sandal': 9,
                              'Shirt': 10, 'Sneaker': 11, 'Bag': 12, 'Ankle boot': 13}
        self.ix_to_classes = {self.classes_to_ix[k]: k for k in self.classes_to_ix}
        self.classes = [k for k in self.classes_to_ix]
        self.family = {'tops': 0, 'bottoms': 1, 'accessories': 2, 'footwear': 3}
        self.n_classes = 14
        self.levels = [4, 10]
        self.level_names = ['family', 'classes']
        self.map = {
            'T-shirt_top': ['tops'],
            'Trouser': ['bottoms'],
            'Pullover': ['tops'],
            'Dress': ['tops'],
            'Coat': ['tops'],
            'Sandal': ['footwear'],
            'Shirt': ['tops'],
            'Sneaker': ['footwear'],
            'Bag': ['accessories'],
            'Ankle boot': ['footwear']
        }
        self.child_of_family_ix = {0: [0, 2, 3, 4, 6], 1: [1], 2: [8], 3: [5, 7, 9]}

    def get_labels(self, class_index):
        class_index += 4
        family = self.map[self.classes[class_index]][0]
        return [self.family[family], class_index]

    def labels_one_hot(self, class_index):
        indices = self.get_labels(class_index)
        retval = np.zeros(self.n_classes)
        retval[indices] = 1
        return retval

    def get_level_labels(self, class_index):
        level_labels = self.get_labels(class_index)
        return np.array([level_labels[0], level_labels[1]-self.levels[0]])


class FMNISTHierarchical(torchvision.datasets.FashionMNIST):
    def __init__(self, root, labelmap, train=True,
                 transform=None, target_transform=None,
                 download=False):
        self.labelmap = labelmap
        torchvision.datasets.FashionMNIST.__init__(self, root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        multi_class_target = self.labelmap.labels_one_hot(target)
        return {'image': img, 'labels': torch.from_numpy(multi_class_target).float(), 'leaf_class': target,
                'level_labels': torch.from_numpy(self.labelmap.get_level_labels(target)).long()}


def FMNIST_set_indices(trainset, testset, labelmap=labelmap_FMNIST()):
    indices = {d_set_name: {label_ix: [] for label_ix in range(len(labelmap.map))} for d_set_name in ['train', 'val']}
    for d_set, d_set_name in zip([trainset, testset], ['train', 'val']):
        for i in range(len(d_set)):
            indices[d_set_name][d_set[i]['leaf_class']].append(i)

    train_indices_from_train = []
    for label_ix in range(len(indices['train'])):
        train_indices_from_train += indices['train'][label_ix][:5000]

    val_indices_from_train = []
    val_indices_from_test = []
    for label_ix in range(len(indices['train'])):
        val_indices_from_train += indices['train'][label_ix][-1000:]

    test_indices_from_test = []
    for label_ix in range(len(indices['val'])):
        test_indices_from_test += indices['val'][label_ix]

    print('Train set has: {}'.format(len(set(train_indices_from_train))))
    print('Val set has: {} + {}'.format(len(set(val_indices_from_train)), len(set(val_indices_from_test))))
    print('Test set has: {}'.format(len(set(test_indices_from_test))))

    return train_indices_from_train, val_indices_from_train, val_indices_from_test, test_indices_from_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", help='Use DEBUG mode.', action='store_true')
    parser.add_argument("--lr", help='Input learning rate.', type=float, default=0.001)
    parser.add_argument("--batch_size", help='Batch size.', type=int, default=8)
    parser.add_argument("--evaluator", help='Evaluator type.', type=str, default='ML')
    parser.add_argument("--experiment_name", help='Experiment name.', type=str, required=True)
    parser.add_argument("--experiment_dir", help='Experiment directory.', type=str, required=True)
    parser.add_argument("--n_epochs", help='Number of epochs to run training for.', type=int, required=True)
    parser.add_argument("--n_workers", help='Number of workers.', type=int, default=4)
    parser.add_argument("--optimizer_method", help='[adam, sgd]', type=str, default='adam')
    parser.add_argument("--eval_interval", help='Evaluate model every N intervals.', type=int, default=1)
    parser.add_argument("--resume", help='Continue training from last checkpoint.', action='store_true')
    parser.add_argument("--model", help='NN model to use.', type=str, required=True)
    parser.add_argument("--freeze_weights", help='This flag fine tunes only the last layer.', action='store_true')
    parser.add_argument("--class_weights", help='Re-weigh the loss function based on inverse class freq.', action='store_true')
    parser.add_argument("--set_mode", help='If use training or testing mode (loads best model).', type=str, required=True)
    parser.add_argument("--loss", help='Loss function to use.', type=str, required=True)
    parser.add_argument("--level_weights", help='List of weights for each level', nargs=2, default=None, type=float)
    args = parser.parse_args()

    train_FMNIST(args)
