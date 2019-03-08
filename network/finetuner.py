from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tensorboardX import SummaryWriter
import datetime
from evaluation import Evaluation

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


class Experiment:

    def __init__(self, model, dataloaders, criterion, classes, experiment_name, n_epochs, batch_size, exp_dir):
        self.exp_dir = exp_dir
        self.classes = classes
        self.criterion = criterion
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.n_epochs = n_epochs
        self.dataloaders = dataloaders
        print('Training set has {} samples. Validation set has {} samples.'.format(
            len(self.dataloaders['train'].dataset),
            len(self.dataloaders['val'].dataset)))

        self.log_dir = os.path.join(self.exp_dir, '{}_{}').format(datetime.datetime.today().strftime('%Y_%m_%d_%H_%M_%S'),
                                                                  experiment_name)
        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'tensorboard'))

    def set_parameter_requires_grad(self, feature_extracting):
        if feature_extracting:
            for param in self.model.parameters():
                param.requires_grad = False

    def train_model(self, optimizer):
        self.optimizer = optimizer

        since = time.time()

        val_acc_history = []

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        eval = Evaluation(self.log_dir, self.classes)

        for epoch in range(self.n_epochs):
            print('Epoch {}/{}'.format(epoch, self.n_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                predicted_scores = np.zeros((len(self.dataloaders[phase].dataset), self.n_classes))
                correct_labels = np.zeros((len(self.dataloaders[phase].dataset)))

                # Iterate over data.
                for index, data_item in enumerate(self.dataloaders[phase]):
                    inputs, labels = data_item
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    predicted_scores[self.batch_size*index:min(self.batch_size*(index+1),
                                                               len(self.dataloaders[phase].dataset)), :] = outputs.data
                    correct_labels[self.batch_size*index:min(self.batch_size*(index+1),
                                                             len(self.dataloaders[phase].dataset))] = labels.data

                eval.evaluate(predicted_scores, correct_labels, epoch, phase)

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(self.dataloaders[phase].dataset)

                self.writer.add_scalar('{}_loss'.format(phase), epoch_loss, epoch)
                self.writer.add_scalar('{}_accuracy'.format(phase), epoch_acc, epoch)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)

        self.writer.close()
        return self.model, val_acc_history


class Finetuner(Experiment):
    def __init__(self, data_dir, data_transforms, classes, criterion, lr,
                 batch_size,
                 experiment_name,
                 experiment_dir='../exp/',
                 n_epochs=10,
                 feature_extracting=True,
                 use_pretrained=True):

        model = models.alexnet(pretrained=use_pretrained)
        image_paths = {x: os.path.join(data_dir, x) for x in ['train', 'val']}
        data_loaders = dataload(image_paths, data_transforms, batch_size)

        Experiment.__init__(self, model, data_loaders, criterion, classes, experiment_name, n_epochs, batch_size, experiment_dir)

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
        self.train_model(optim.SGD(self.params_to_update, lr=self.lr, momentum=0.9))


class CIFAR10(Experiment):
    def __init__(self, data_loaders, classes, criterion, lr,
                 batch_size,
                 experiment_name,
                 experiment_dir='../exp/',
                 n_epochs=10,
                 feature_extracting=True,
                 use_pretrained=True):

        self.classes = classes
        self.lr = lr
        self.batch_size = batch_size
        self.feature_extracting = feature_extracting

        model = models.alexnet(pretrained=use_pretrained)
        Experiment.__init__(self, model, data_loaders, criterion, classes, experiment_name, n_epochs, experiment_dir)

        self.set_parameter_requires_grad(self.feature_extracting)
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_features, len(self.classes))

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
        self.train_model(optim.SGD(self.params_to_update, lr=self.lr, momentum=0.9))


def train_cifar10():
    input_size = 224
    data_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    trainset = torchvision.datasets.CIFAR10(root='../database', train=True,
                                            download=True, transform=data_transforms)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='../database', train=False,
                                           download=True, transform=data_transforms)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=4)

    data_loaders = {'train': trainloader, 'val': testloader}
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    cifar_trainer = CIFAR10(data_loaders=data_loaders, classes=classes,
                            criterion=nn.CrossEntropyLoss(),
                            lr=0.001,
                            batch_size=8,
                            experiment_name='alexnet_ft',
                            n_epochs=10)


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
              n_epochs=10)


if __name__ == '__main__':
    train_alexnet_binary()
