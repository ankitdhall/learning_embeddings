from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

import os
from experiment import Experiment

from PIL import Image
import numpy as np


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
                            batch_size, experiment_dir, load_wt)

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

        self.train()

    def train(self):
        self.run_model(optim.SGD(self.params_to_update, lr=self.lr, momentum=0.9))


class CIFAR10(Experiment):
    def __init__(self, data_loaders, labelmap, criterion, lr,
                 batch_size,
                 experiment_name,
                 experiment_dir='../exp/',
                 n_epochs=10,
                 eval_interval=2,
                 feature_extracting=True,
                 use_pretrained=True,
                 load_wt=False):

        self.classes = labelmap.classes
        self.n_classes = labelmap.n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.feature_extracting = feature_extracting

        model = models.alexnet(pretrained=use_pretrained)
        Experiment.__init__(self, model, data_loaders, criterion, self.classes, experiment_name, n_epochs, eval_interval,
                            batch_size, experiment_dir, load_wt)

        self.set_parameter_requires_grad(self.feature_extracting)
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

        self.train()

    def train(self):
        self.run_model(optim.SGD(self.params_to_update, lr=self.lr, momentum=0.9))

class labelmap_CIFAR10:
    def __init__(self):
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')
        self.family = {'living': 10, 'non_living': 11}
        self.subfamily = {'non_land': 12, 'land': 13, 'vehicle': 14, 'craft': 15}
        self.n_classes = 16
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

    def get_labels(self, class_index):
        family, subfamily = self.map[self.classes[class_index]]
        return [class_index, self.family[family], self.subfamily[subfamily]]

    def labels_one_hot(self, class_index):
        indices = self.get_labels(class_index)
        retval = np.zeros(self.n_classes)
        retval[indices] = 1
        return retval

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

        if self.target_transform is not None:
            target = self.target_transform(target)

        multi_class_target = self.labelmap.labels_one_hot(target)
        return img, multi_class_target


def train_cifar10():
    input_size = 224
    data_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    lmap = labelmap_CIFAR10_single()
    batch_size = 8

    trainset = Cifar10Hierarchical(root='../database', labelmap=lmap, train=False,
                                   download=True, transform=data_transforms)
    trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, list(range(1000))), batch_size=batch_size,
                                              shuffle=True, num_workers=4)

    testset = Cifar10Hierarchical(root='../database', labelmap=lmap, train=False,
                                  download=True, transform=data_transforms)
    testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(testset, list(range(1000, 2000))), batch_size=batch_size,
                                             shuffle=False, num_workers=4)

    data_loaders = {'train': trainloader, 'val': testloader}

    cifar_trainer = CIFAR10(data_loaders=data_loaders, labelmap=lmap,
                            criterion=nn.MultiLabelSoftMarginLoss(),
                            lr=0.01,
                            batch_size=batch_size,
                            experiment_name='cifar_test_ft_multi',
                            experiment_dir='../exp/',
                            eval_interval=2,
                            n_epochs=10,
                            feature_extracting=True,
                            use_pretrained=True,
                            load_wt=True)

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
    trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, list(range(1000))), batch_size=batch_size,
                                              shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='../database', train=False, download=True, transform=data_transforms)
    testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(testset, list(range(1000, 2000))), batch_size=batch_size,
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
              load_wt=False)


if __name__ == '__main__':
    # train_alexnet_binary()
    train_cifar10_single()
