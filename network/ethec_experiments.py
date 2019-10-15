from __future__ import print_function
from __future__ import division
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

import os
from network.experiment import Experiment, WeightedResampler
from network.evaluation import MultiLabelEvaluation, Evaluation, MultiLabelEvaluationSingleThresh, MultiLevelEvaluation
from network.finetuner import CIFAR10

from data.db import ETHECLabelMap, ETHECDB, ETHECDBMerged, ETHECLabelMapMerged, ETHECLabelMapMergedSmall, ETHECDBMergedSmall
from data.db import Butterfly200LabelMap
from network.loss import MultiLevelCELoss, MultiLabelSMLoss, LastLevelCELoss, MaskedCELoss, HierarchicalSoftmaxLoss

from PIL import Image
import numpy as np
np.random.seed(0)

import copy
import argparse
import json
import git
import time

import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt

class CNN2DFeat(torch.nn.Module):
    def __init__(self, original_model, labelmap):
        torch.nn.Module.__init__(self)
        self.original_model = original_model
        self.labelmap = labelmap
        num_features = self.original_model.fc.in_features
        self.original_model.fc = nn.Linear(num_features, 2, bias=False)

        self.final_linears = nn.ModuleList([nn.Linear(2, level, bias=False) for level in self.labelmap.levels])

    def forward(self, x):
        x = self.original_model(x)
        x = [self.final_linears[level_id](x) for level_id in range(len(self.labelmap.levels))]
        x = torch.cat(tuple(x), dim=1)
        return x


class ETHEC2D(CIFAR10):
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

        self.dataset_length = {phase: len(self.dataloaders[phase].dataset) for phase in ['train', 'val', 'test']}

        self.set_parameter_requires_grad(self.feature_extracting)
        self.model = CNN2DFeat(self.model, self.labelmap)

        self.n_train, self.n_val, self.n_test = torch.zeros(self.n_classes), torch.zeros(self.n_classes), \
                                                torch.zeros(self.n_classes)
        for phase in ['train', 'val', 'test']:
            for data_item in self.dataloaders[phase]:
                setattr(self, 'n_{}'.format(phase),
                        getattr(self, 'n_{}'.format(phase)) + torch.sum(data_item['labels'], 0))

        self.samples_split = {'train': self.n_train, 'val': self.n_val, 'test': self.n_test}

        self.model = nn.DataParallel(self.model)

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
                self.plot_label_representations(to_load=False)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val score: {:4f}'.format(self.best_score))

        # load best model weights
        self.model.load_state_dict(self.best_model_wts)

        self.writer.close()
        return self.model

    def load_model(self, epoch_to_load):
        checkpoint = torch.load(os.path.join(self.path_to_save_model, '{}.pth'.format(epoch_to_load)), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.epoch = checkpoint['epoch']
        print('Successfully loaded model epoch {} from {}'.format(self.epoch, self.path_to_save_model))

    def plot_label_representations(self, to_load=True):
        if to_load:
            self.load_model(epoch_to_load='best_model')
        colors = ['c', 'm', 'y', 'k']
        embeddings_x, embeddings_y, annotation, color_list = {}, {}, {}, {}

        connected_to = {}

        fig, ax = plt.subplots()

        for level_id in range(len(self.labelmap.levels)):
            level_start, level_stop = self.labelmap.level_start[level_id], self.labelmap.level_stop[level_id]
            weights = self.model.module.final_linears[level_id].weight.data.cpu().numpy()
            for label_ix in range(weights.shape[0]):
                emb_id = label_ix + level_start

                embeddings_x[emb_id] = weights[label_ix][0]
                embeddings_y[emb_id] = weights[label_ix][1]
                annotation[emb_id] = '{}'.format(getattr(self.labelmap,
                                                         '{}_ix_to_str'.format(self.labelmap.level_names[level_id]))[label_ix]
                                                 )
                color_list[emb_id] = colors[level_id]

                if level_id < len(self.labelmap.levels)-1:
                    connected_to[emb_id] = getattr(self.labelmap, 'child_of_{}_ix'.format(self.labelmap.level_names[level_id]))[label_ix]
                    connected_to[emb_id] = [i+self.labelmap.level_start[level_id+1] for i in connected_to[emb_id]]

                ax.scatter(weights[label_ix][0], weights[label_ix][1], c=colors[level_id], alpha=1)
                # ax.annotate(annotation[emb_id], (weights[label_ix][0], weights[label_ix][1]))

        for from_node in connected_to:
            for to_node in connected_to[from_node]:
                if to_node in embeddings_x:
                    plt.plot([embeddings_x[from_node], embeddings_x[to_node]], [embeddings_y[from_node], embeddings_y[to_node]],
                             'b-', alpha=0.2)

        emb_info = {'x': embeddings_x, 'y': embeddings_y, 'annotation': annotation, 'color': color_list, 'connected_to': connected_to}
        np.save(os.path.join(self.log_dir, '{0:04d}_embedding_info.npy'.format(self.epoch)), emb_info)

        # if self.title_text:
        #     fig.suptitle(self.title_text, family='sans-serif')
        fig.set_size_inches(8, 7)
        ax.axis('equal')
        fig.savefig(os.path.join(self.log_dir, '{0:04d}_embedding.pdf'.format(self.epoch)), dpi=200)
        fig.savefig(os.path.join(self.log_dir, '{0:04d}_embedding.png'.format(self.epoch)), dpi=200)
        print('Successfully saved embedding to disk!')

class ETHECExperiment(CIFAR10):
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

        CIFAR10.__init__(self, data_loaders, labelmap, criterion, lr, batch_size, evaluator, experiment_name,
                         experiment_dir, n_epochs, eval_interval, feature_extracting, use_pretrained,
                         load_wt, model_name, optimizer_method)

        if use_grayscale:
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
                self.model.conv1 = nn.Conv2d(1, o_channels, kernel_size=k_size, stride=stride, padding=pad,
                                             dilation=dil)

        self.model = nn.DataParallel(self.model)


def ETHEC_train_model(arguments):
    if not os.path.exists(os.path.join(arguments.experiment_dir, arguments.experiment_name)):
        os.makedirs(os.path.join(arguments.experiment_dir, arguments.experiment_name))
    args_dict = vars(arguments)
    repo = git.Repo(search_parent_directories=True)
    args_dict['commit_hash'] = repo.head.object.hexsha
    args_dict['branch'] = repo.active_branch.name
    with open(os.path.join(arguments.experiment_dir, arguments.experiment_name, 'config_params.txt'), 'w') as file:
        file.write(json.dumps(args_dict, indent=4))

    print('Config parameters for this run are:\n{}'.format(json.dumps(vars(arguments), indent=4)))

    initial_crop = 512
    input_size = 448
    labelmap = Butterfly200LabelMap()
    if arguments.merged:
        labelmap = Butterfly200LabelMap()
    if arguments.debug:
        labelmap = ETHECLabelMapMergedSmall()

    train_data_transforms = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize((initial_crop, initial_crop)),
                                                transforms.RandomCrop((input_size, input_size)),
                                                transforms.RandomHorizontalFlip(),
                                                # ColorJitter(brightness=0.2, contrast=0.2),
                                                transforms.ToTensor(),
                                                # transforms.Normalize(mean=(143.2341, 162.8151, 177.2185),
                                                #                      std=(66.7762, 59.2524, 51.5077))
                                                ])
    val_test_data_transforms = transforms.Compose([transforms.ToPILImage(),
                                                   transforms.Resize((initial_crop, initial_crop)),
                                                   transforms.CenterCrop((input_size, input_size)),
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
        train_set = ETHECDB(path_to_json='../database/butterfly200/train.json',
                            path_to_images=arguments.image_dir,
                            labelmap=labelmap, transform=train_data_transforms)
        val_set = ETHECDB(path_to_json='../database/butterfly200/val.json',
                          path_to_images=arguments.image_dir,
                          labelmap=labelmap, transform=val_test_data_transforms)
        test_set = ETHECDB(path_to_json='../database/butterfly200/test.json',
                           path_to_images=arguments.image_dir,
                           labelmap=labelmap, transform=val_test_data_transforms)
    elif not arguments.debug:
        train_set = ETHECDBMerged(path_to_json='../database/butterfly200/train.json',
                                  path_to_images=arguments.image_dir,
                                  labelmap=labelmap, transform=train_data_transforms)
        val_set = ETHECDBMerged(path_to_json='../database/butterfly200/val.json',
                                path_to_images=arguments.image_dir,
                                labelmap=labelmap, transform=val_test_data_transforms)
        test_set = ETHECDBMerged(path_to_json='../database/butterfly200/test.json',
                                 path_to_images=arguments.image_dir,
                                 labelmap=labelmap, transform=val_test_data_transforms)
    else:
        labelmap = ETHECLabelMapMergedSmall(single_level=False)
        train_set = ETHECDBMergedSmall(path_to_json='../database/butterfly200/train.json',
                                       path_to_images=arguments.image_dir,
                                       labelmap=labelmap, transform=train_data_transforms)
        val_set = ETHECDBMergedSmall(path_to_json='../database/butterfly200/val.json',
                                     path_to_images=arguments.image_dir,
                                     labelmap=labelmap, transform=val_test_data_transforms)
        test_set = ETHECDBMergedSmall(path_to_json='../database/butterfly200/test.json',
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
    if arguments.loss == 'multi_label':
        use_criterion = MultiLabelSMLoss(weight=weight)
    elif arguments.loss == 'multi_level':
        use_criterion = MultiLevelCELoss(labelmap=labelmap, weight=weight, level_weights=arguments.level_weights)
        eval_type = MultiLevelEvaluation(os.path.join(arguments.experiment_dir, arguments.experiment_name), labelmap)
    elif arguments.loss == 'last_level':
        use_criterion = LastLevelCELoss(labelmap=labelmap, weight=weight, level_weights=arguments.level_weights)
        eval_type = MultiLevelEvaluation(os.path.join(arguments.experiment_dir, arguments.experiment_name), labelmap)
    elif arguments.loss == 'masked_loss':
        use_criterion = MaskedCELoss(labelmap=labelmap, level_weights=arguments.level_weights)
        eval_type = MultiLevelEvaluation(os.path.join(arguments.experiment_dir, arguments.experiment_name), labelmap)
    elif arguments.loss == 'hsoftmax':
        use_criterion = HierarchicalSoftmaxLoss(labelmap=labelmap, level_weights=arguments.level_weights)
        eval_type = MultiLevelEvaluation(os.path.join(arguments.experiment_dir, arguments.experiment_name), labelmap)
    else:
        print("== Invalid --loss argument")

    if arguments.use_2d:
        ETHEC_trainer = ETHEC2D(data_loaders=data_loaders, labelmap=labelmap,
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
    else:
        ETHEC_trainer = ETHECExperiment(data_loaders=data_loaders, labelmap=labelmap,
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
    ETHEC_trainer.prepare_model()
    #if arguments.use_2d and arguments.resume:
    #    ETHEC_trainer.plot_label_representations()
    #    return
    if arguments.set_mode == 'train':
        ETHEC_trainer.train()
    elif arguments.set_mode == 'test':
        ETHEC_trainer.test()


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
    parser.add_argument("--loss", help='Loss function to use. [multi_label, multi_level, last_level, masked_loss, hsoftmax]', type=str, required=True)
    parser.add_argument("--use_grayscale", help='Use grayscale images.', action='store_true')
    parser.add_argument("--class_weights", help='Re-weigh the loss function based on inverse class freq.', action='store_true')
    parser.add_argument("--freeze_weights", help='This flag fine tunes only the last layer.', action='store_true')
    parser.add_argument("--set_mode", help='If use training or testing mode (loads best model).', type=str,
                        required=True)
    parser.add_argument("--level_weights", help='List of weights for each level', nargs=4, default=None, type=float)
    parser.add_argument("--use_2d", help='Use model with 2d features', action='store_true')
    args = parser.parse_args()

    ETHEC_train_model(args)
