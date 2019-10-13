from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

import copy
from tensorboardX import SummaryWriter
import datetime
from evaluation import Evaluation
import time
import numpy as np
import os
import collections

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Experiment:

    def __init__(self, model, dataloaders, criterion, classes, experiment_name, n_epochs, eval_interval, batch_size,
                 exp_dir, load_wt, evaluator):
        self.epoch = 0
        self.exp_dir = exp_dir
        self.load_wt = load_wt

        self.eval = evaluator

        self.classes = classes
        self.criterion = criterion
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using device: {}'.format(self.device))
        if torch.cuda.device_count() > 1:
            print("== Using", torch.cuda.device_count(), "GPUs!")
        self.model = model.to(self.device)
        self.n_epochs = n_epochs
        self.eval_interval = eval_interval
        self.dataloaders = dataloaders
        print('Training set has {} samples. Validation set has {} samples. Test set has {} samples'.format(
            len(self.dataloaders['train'].dataset),
            len(self.dataloaders['val'].dataset),
            len(self.dataloaders['test'].dataset)))

        self.log_dir = os.path.join(self.exp_dir, '{}').format(experiment_name)
        self.path_to_save_model = os.path.join(self.log_dir, 'weights')
        self.make_dir_if_non_existent(self.path_to_save_model)

        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'tensorboard'))

    @staticmethod
    def make_dir_if_non_existent(dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def set_parameter_requires_grad(self, feature_extracting):
        if feature_extracting:
            for param in self.model.parameters():
                param.requires_grad = False

    def plot_grad_flow(self):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in self.model.named_parameters():
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([matplotlib.lines.Line2D([0], [0], color="c", lw=4),
                    matplotlib.lines.Line2D([0], [0], color="b", lw=4),
                    matplotlib.lines.Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.savefig(os.path.join(self.log_dir, 'gradient_flow.png'))

    def pass_samples(self, phase, save_to_tensorboard=True):
        running_loss = 0.0
        running_corrects = 0

        predicted_scores = np.zeros((len(self.dataloaders[phase].dataset), self.n_classes))
        correct_labels = np.zeros((len(self.dataloaders[phase].dataset)))

        # Iterate over data.
        for index, data_item in enumerate(self.dataloaders[phase]):
            inputs, labels = data_item
            inputs = inputs.to(self.device)
            labels = labels.float().to(self.device)

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
                    # self.plot_grad_flow()
                    self.optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            predicted_scores[self.batch_size * index:min(self.batch_size * (index + 1),
                                                         len(self.dataloaders[phase].dataset)), :] = outputs.data
            correct_labels[self.batch_size * index:min(self.batch_size * (index + 1),
                                                       len(self.dataloaders[phase].dataset))] = labels.data

        mAP, _, _, _, _ = self.eval.evaluate(predicted_scores, correct_labels, self.epoch, phase, save_to_tensorboard, )

        epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(self.dataloaders[phase].dataset)

        if save_to_tensorboard:
            self.writer.add_scalar('{}_loss'.format(phase), epoch_loss, self.epoch)
            self.writer.add_scalar('{}_accuracy'.format(phase), epoch_acc, self.epoch)
            self.writer.add_scalar('{}_mAP'.format(phase), mAP, self.epoch)

        print('{} Loss: {:.4f} Score: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        # deep copy the model
        if phase == 'val':
            self.save_model(epoch_loss)
            if epoch_acc > self.best_score:
                self.best_score = epoch_acc
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                self.save_model(epoch_loss, filename='best_model')

    def run_model(self, optimizer):
        self.optimizer = optimizer

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
                if self.epoch % 10 == 0:
                    self.eval.enable_plotting()
                self.pass_samples(phase='val')
                self.pass_samples(phase='test')
                self.eval.disable_plotting()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val score: {:4f}'.format(self.best_score))

        # load best model weights
        self.model.load_state_dict(self.best_model_wts)

        self.writer.close()
        return self.model

    def save_model(self, loss, filename=None):
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }, os.path.join(self.path_to_save_model, '{}.pth'.format(filename if filename else self.epoch)))
        print('Successfully saved model epoch {} to {} as {}.pth'.format(self.epoch, self.path_to_save_model,
                                                                         filename if filename else self.epoch))

    def load_model(self, epoch_to_load):
        checkpoint = torch.load(os.path.join(self.path_to_save_model, '{}.pth'.format(epoch_to_load)), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        print('Successfully loaded model epoch {} from {}'.format(self.epoch, self.path_to_save_model))

    def find_existing_weights(self):
        weights = sorted([filename for filename in os.listdir(self.path_to_save_model)])
        if len(weights) < 2:
            print('Could not find weights to load from, will train from scratch.')
        else:
            self.load_model(epoch_to_load=weights[-2].split('.')[0])

    def load_best_model(self, only_load=False):
        if only_load:
            self.load_model(epoch_to_load='best_model')
        else:
            self.load_model(epoch_to_load='best_model')
            self.eval.enable_plotting()
            self.pass_samples(phase='test', save_to_tensorboard=False)
            self.eval.disable_plotting()


class WeightedResampler(torch.utils.data.sampler.WeightedRandomSampler):
    def __init__(self, dataset, start=None, stop=None, weight_strategy='inv'):
        dset_len = len(dataset)
        if start is None and stop is None:
            label_ids = [dataset[ind]['leaf_label'] for ind in range(len(dataset))]
        elif start is not None and stop is not None:
            label_ids = [dataset[ind]['leaf_label'] for ind in range(start, stop)]
            dset_len = stop-start

        label_counts = collections.Counter(label_ids)
        dense_label_counts = np.zeros(dataset.labelmap.levels[-1], dtype=np.float32)
        dense_label_counts[list(label_counts.keys())] = list(label_counts.values())

        if np.any(dense_label_counts == 0):
            print("[warning] Found labels with zero samples")

        if weight_strategy == 'inv':
            label_weights = 1.0 / dense_label_counts
        elif weight_strategy == 'inv_sqrt':
            label_weights = 1.0 / np.sqrt(dense_label_counts)

        label_weights[dense_label_counts == 0] = 0.0

        weights = label_weights[label_ids]
        torch.utils.data.sampler.WeightedRandomSampler.__init__(self, weights, dset_len, replacement=True)
