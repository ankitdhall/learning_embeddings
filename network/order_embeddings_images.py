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
import time
from tqdm import tqdm

import copy
import argparse
import json
import git

import torch
from torch import nn
import torch.nn.functional as F
from data.db import ETHECLabelMap, ETHECLabelMapMergedSmall

from network.finetuner import CIFAR10
import numpy as np
import random
random.seed(0)

import networkx as nx
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from network.order_embeddings import OrderEmbedding, OrderEmbeddingLoss
from network.inference import Inference


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class FeatNet(nn.Module):
    """
    Fully connected NN to learn features on top of image features in the joint embedding space.
    """
    def __init__(self, input_dim=2048, output_dim=10):
        """
        Constructor to prepare layers for the embedding.
        """
        super(FeatNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        """
        Forward pass through the model.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.abs(self.fc3(x))
        return x


class ImageEmb:
    def __init__(self, path_to_exp='../exp/ethec_resnet50_lr_1e-5_1_1_1_1/',
                 image_dir='/media/ankit/DataPartition/IMAGO_build_test_resized'):
        self.path_to_exp = path_to_exp
        self.image_dir = image_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def load_model(self):
        inf_obj = Inference(path_to_exp=self.path_to_exp, image_dir=self.image_dir, mode=None, perform_inference=False)
        self.model = inf_obj.get_model()
        print(self.model)

    def calc_emb(self):
        input_size = 224
        val_test_data_transforms = transforms.Compose([transforms.ToPILImage(),
                                                       transforms.Resize((input_size, input_size)),
                                                       transforms.ToTensor(),
                                                       ])
        labelmap = ETHECLabelMapMergedSmall()
        train_set = ETHECDBMergedSmall(path_to_json='../database/ETHEC/train.json',
                                       path_to_images=self.image_dir,
                                       labelmap=labelmap, transform=val_test_data_transforms)
        val_set = ETHECDBMergedSmall(path_to_json='../database/ETHEC/val.json',
                                     path_to_images=self.image_dir,
                                     labelmap=labelmap, transform=val_test_data_transforms)
        test_set = ETHECDBMergedSmall(path_to_json='../database/ETHEC/test.json',
                                      path_to_images=self.image_dir,
                                      labelmap=labelmap, transform=val_test_data_transforms)
        trainloader = torch.utils.data.DataLoader(train_set,
                                                 batch_size=1,
                                                 shuffle=False, num_workers=0)
        valloader = torch.utils.data.DataLoader(val_set,
                                                 batch_size=1,
                                                 shuffle=False, num_workers=0)
        testloader = torch.utils.data.DataLoader(test_set,
                                                 batch_size=1,
                                                 shuffle=False, num_workers=0)
        self.model.module.fc = Identity()

        path_to_save_emb = '../database/ETHEC/embeddings'
        if not os.path.exists(path_to_save_emb):
            os.makedirs(path_to_save_emb)

        for loader, loader_name in zip([trainloader, valloader, testloader], ['train', 'val', 'test']):
            embeddings = {}
            print('{} items in {} loader'.format(len(loader), loader_name))
            for index, data_item in enumerate(tqdm(loader)):
                outputs = self.model(data_item['image']).detach()
                embeddings[data_item['image_filename'][0]] = outputs[0].numpy()
            print(embeddings)
            with open(os.path.join(path_to_save_emb, '{}.json'.format(loader_name)), 'w') as fp:
                json.dump(embeddings, fp)


ie_model = ImageEmb()
ie_model.load_model()
ie_model.calc_emb()
