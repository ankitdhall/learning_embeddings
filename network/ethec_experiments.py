from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

import os
from network.experiment import Experiment, WeightedResampler
from network.evaluation import MLEvaluation, Evaluation, MLEvaluationSingleThresh
from network.finetuner import CIFAR10

from data.db import ETHECLabelMap, Rescale, ToTensor, Normalize, ColorJitter, RandomHorizontalFlip, RandomCrop, ToPILImage, ETHECDB
from network.loss import MultiLevelCELoss, MultiLabelSMLoss

from PIL import Image
import numpy as np

import copy
import argparse
import json


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
                 model_name=None):
        CIFAR10.__init__(self, data_loaders, labelmap, criterion, lr, batch_size, evaluator, experiment_name,
                         experiment_dir, n_epochs, eval_interval, feature_extracting, use_pretrained,
                         load_wt, model_name)


def ETHEC_train_model(arguments):
    if not os.path.exists(os.path.join(arguments.experiment_dir, arguments.experiment_name)):
        os.makedirs(os.path.join(arguments.experiment_dir, arguments.experiment_name))
    with open(os.path.join(arguments.experiment_dir, arguments.experiment_name, 'config_params.txt'), 'w') as file:
        file.write(json.dumps(vars(args), indent=4))

    print('Config parameters for this run are:\n{}'.format(json.dumps(vars(args), indent=4)))

    initial_crop = 324
    input_size = 224
    labelmap = ETHECLabelMap()
    train_data_transforms = transforms.Compose([ToPILImage(),
                                                Rescale((initial_crop, initial_crop)),
                                                RandomCrop((input_size, input_size)),
                                                RandomHorizontalFlip(),
                                                # ColorJitter(brightness=0.2, contrast=0.2),
                                                ToTensor(),
                                                Normalize(mean=(143.2341, 162.8151, 177.2185),
                                                          std=(66.7762, 59.2524, 51.5077))])
    val_test_data_transforms = transforms.Compose([ToPILImage(),
                                                   Rescale((input_size, input_size)),
                                                   ToTensor(),
                                                   Normalize(mean=(143.2341, 162.8151, 177.2185),
                                                             std=(66.7762, 59.2524, 51.5077))])

    train_set = ETHECDB(path_to_json='../database/ETHEC/train.json',
                        path_to_images=args.image_dir,
                        labelmap=labelmap, transform=train_data_transforms)
    val_set = ETHECDB(path_to_json='../database/ETHEC/val.json',
                      path_to_images=args.image_dir,
                      labelmap=labelmap, transform=val_test_data_transforms)
    test_set = ETHECDB(path_to_json='../database/ETHEC/test.json',
                       path_to_images=args.image_dir,
                       labelmap=labelmap, transform=val_test_data_transforms)
    print('Dataset has following splits: train: {}, val: {}, test: {}'.format(len(train_set), len(val_set),
                                                                              len(test_set)))

    batch_size = arguments.batch_size
    n_workers = arguments.n_workers

    if arguments.debug:
        print("== Running in DEBUG mode!")
        trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_set, list(range(100))),
                                                  batch_size=batch_size,
                                                  num_workers=n_workers,
                                                  sampler=WeightedResampler(train_set, start=0, stop=100))

        valloader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_set, list(range(100, 200))),
                                                batch_size=batch_size,
                                                shuffle=False, num_workers=n_workers)

        testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_set, list(range(200, 300))),
                                                 batch_size=batch_size,
                                                 shuffle=False, num_workers=n_workers)

        data_loaders = {'train': trainloader, 'val': valloader, 'test': testloader}

    else:
        trainloader = torch.utils.data.DataLoader(train_set,
                                                  batch_size=batch_size,
                                                  num_workers=n_workers,
                                                  sampler=WeightedResampler(train_set))
        valloader = torch.utils.data.DataLoader(val_set,
                                                batch_size=batch_size,
                                                shuffle=False, num_workers=n_workers)
        testloader = torch.utils.data.DataLoader(test_set,
                                                 batch_size=batch_size,
                                                 shuffle=False, num_workers=n_workers)

        data_loaders = {'train': trainloader, 'val': valloader, 'test': testloader}

    eval_type = MLEvaluation(os.path.join(arguments.experiment_dir, arguments.experiment_name), labelmap)
    if arguments.evaluator == 'MLST':
        eval_type = MLEvaluationSingleThresh(os.path.join(arguments.experiment_dir, arguments.experiment_name),
                                             labelmap)

    use_criterion = None
    if arguments.loss == 'multi_label':
        use_criterion = MultiLabelSMLoss()
    elif arguments.loss == 'multi_level':
        use_criterion = MultiLevelCELoss(labelmap=labelmap)

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
                                    load_wt=False,
                                    model_name=arguments.model)
    ETHEC_trainer.prepare_model()
    if arguments.set_mode == 'train':
        ETHEC_trainer.train()
    elif arguments.set_mode == 'test':
        ETHEC_trainer.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", help='Use DEBUG mode.', action='store_true')
    parser.add_argument("--lr", help='Input learning rate.', type=float, default=0.01)
    parser.add_argument("--batch_size", help='Batch size.', type=int, default=8)
    parser.add_argument("--evaluator", help='Evaluator type.', type=str, default='ML')
    parser.add_argument("--experiment_name", help='Experiment name.', type=str, required=True)
    parser.add_argument("--experiment_dir", help='Experiment directory.', type=str, required=True)
    parser.add_argument("--image_dir", help='Image parent directory.', type=str, required=True)
    parser.add_argument("--n_epochs", help='Number of epochs to run training for.', type=int, required=True)
    parser.add_argument("--n_workers", help='Number of workers.', type=int, default=4)
    parser.add_argument("--eval_interval", help='Evaluate model every N intervals.', type=int, default=1)
    parser.add_argument("--resume", help='Continue training from last checkpoint.', action='store_true')
    parser.add_argument("--model", help='NN model to use. Use one of [`multi_label`, `multi_level`]',
                        type=str, required=True)
    parser.add_argument("--loss", help='Loss function to use.', type=str, required=True)
    parser.add_argument("--freeze_weights", help='This flag fine tunes only the last layer.', action='store_true')
    parser.add_argument("--set_mode", help='If use training or testing mode (loads best model).', type=str,
                        required=True)
    args = parser.parse_args()

    ETHEC_train_model(args)
