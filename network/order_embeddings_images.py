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

from network.order_embeddings import OrderEmbedding, OrderEmbeddingLoss
from network.inference import Inference


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
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
        test_set = ETHECDBMergedSmall(path_to_json='../database/ETHEC/test.json',
                                      path_to_images=self.image_dir,
                                      labelmap=labelmap, transform=val_test_data_transforms)
        testloader = torch.utils.data.DataLoader(test_set,
                                                 batch_size=1,
                                                 shuffle=False, num_workers=0)
        self.model.module.fc = Identity()

        print('{} items in testloader'.format(len(testloader)))
        for index, data_item in enumerate(testloader):
            print(index, data_item)
            outputs = self.model(data_item['image'])
            print(outputs)
            break

ie_model = ImageEmb()
ie_model.load_model()
ie_model.calc_emb()
