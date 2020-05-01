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
from network.ethec_experiments import ETHECExperiment

from data.db import ETHECLabelMap, ETHECDB, ETHECDBMerged, ETHECLabelMapMerged, ETHECLabelMapMergedSmall, ETHECDBMergedSmall
from network.loss import MultiLevelCELoss, MultiLabelSMLoss, LastLevelCELoss, MaskedCELoss, HierarchicalSoftmaxLoss

from PIL import Image
import numpy as np

import copy
import argparse
import json
import git

import cv2
import time

import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from network.summarize import Summarize

import matplotlib.image as mpimg

from lime import lime_image
from skimage.segmentation import mark_boundaries

import random


class Inference:
    def __init__(self, path_to_exp, mode, image_dir=None, data_ix=None, perform_inference=True):
        with open(os.path.join(path_to_exp, 'config_params.txt'), 'r') as file:
            arguments = json.loads(file.read())

        print('Config parameters for this run are:\n{}'.format(json.dumps(arguments, indent=4)))
        if image_dir:
            arguments['image_dir'] = image_dir
        arguments['experiment_dir'] = os.path.join(path_to_exp, '..')
        arguments['batch_size'] = 64
        arguments['resume'] = True

        if 'use_grayscale' not in arguments:
            arguments['use_grayscale'] = False
        if 'level_weights' not in arguments:
            arguments['level_weights'] = None

        # initial_crop = 324
        input_size = 224
        labelmap = ETHECLabelMap()
        if arguments['merged']:
            labelmap = ETHECLabelMapMerged()
        if arguments['debug']:
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
        if arguments['use_grayscale']:
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

        if not arguments['merged']:
            train_set = ETHECDB(path_to_json='../database/ETHEC/train.json',
                                path_to_images=arguments['image_dir'],
                                labelmap=labelmap, transform=val_test_data_transforms)
            val_set = ETHECDB(path_to_json='../database/ETHEC/val.json',
                              path_to_images=arguments['image_dir'],
                              labelmap=labelmap, transform=val_test_data_transforms)
            test_set = ETHECDB(path_to_json='../database/ETHEC/test.json',
                               path_to_images=arguments['image_dir'],
                               labelmap=labelmap, transform=val_test_data_transforms)
        elif not arguments['debug']:
            train_set = ETHECDBMerged(path_to_json='../database/ETHEC/train.json',
                                      path_to_images=arguments['image_dir'],
                                      labelmap=labelmap, transform=val_test_data_transforms)
            val_set = ETHECDBMerged(path_to_json='../database/ETHEC/val.json',
                                    path_to_images=arguments['image_dir'],
                                    labelmap=labelmap, transform=val_test_data_transforms)
            test_set = ETHECDBMerged(path_to_json='../database/ETHEC/test.json',
                                     path_to_images=arguments['image_dir'],
                                     labelmap=labelmap, transform=val_test_data_transforms)
        else:
            labelmap = ETHECLabelMapMergedSmall(single_level=False)
            train_set = ETHECDBMergedSmall(path_to_json='../database/ETHEC/train.json',
                                           path_to_images=arguments['image_dir'],
                                           labelmap=labelmap, transform=val_test_data_transforms)
            val_set = ETHECDBMergedSmall(path_to_json='../database/ETHEC/val.json',
                                         path_to_images=arguments['image_dir'],
                                         labelmap=labelmap, transform=val_test_data_transforms)
            test_set = ETHECDBMergedSmall(path_to_json='../database/ETHEC/test.json',
                                          path_to_images=arguments['image_dir'],
                                          labelmap=labelmap, transform=val_test_data_transforms)

        print('Dataset has following splits: train: {}, val: {}, test: {}'.format(len(train_set), len(val_set),
                                                                                  len(test_set)))

        batch_size = arguments['batch_size']
        n_workers = arguments['n_workers']

        testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(test_set, list(range(0, 1))),
                                                 batch_size=batch_size,
                                                 shuffle=False, num_workers=n_workers)

        data_loaders = {'train': testloader, 'val': testloader, 'test': testloader}

        weight = None

        eval_type = MultiLabelEvaluation(os.path.join(arguments['experiment_dir'], arguments['experiment_name']), labelmap)
        if arguments['evaluator'] == 'MLST':
            eval_type = MultiLabelEvaluationSingleThresh(os.path.join(arguments['experiment_dir'], arguments['experiment_name']),
                                                         labelmap)

        use_criterion = None
        if arguments['loss'] == 'multi_label':
            use_criterion = MultiLabelSMLoss(weight=weight)
        elif arguments['loss'] == 'multi_level':
            use_criterion = MultiLevelCELoss(labelmap=labelmap, weight=weight, level_weights=arguments['level_weights'])
            eval_type = MultiLevelEvaluation(os.path.join(arguments['experiment_dir'], arguments['experiment_name']), labelmap)
        elif arguments['loss'] == 'last_level':
            use_criterion = LastLevelCELoss(labelmap=labelmap, weight=weight, level_weights=arguments['level_weights'])
            eval_type = MultiLevelEvaluation(os.path.join(arguments['experiment_dir'], arguments['experiment_name']), labelmap)
        elif arguments['loss'] == 'masked_loss':
            use_criterion = MaskedCELoss(labelmap=labelmap, level_weights=arguments['level_weights'])
            eval_type = MultiLevelEvaluation(os.path.join(arguments['experiment_dir'], arguments['experiment_name']), labelmap)
        elif arguments['loss'] == 'hsoftmax':
            use_criterion = HierarchicalSoftmaxLoss(labelmap=labelmap, level_weights=arguments['level_weights'])
            eval_type = MultiLevelEvaluation(os.path.join(arguments['experiment_dir'], arguments['experiment_name']), labelmap)
        else:
            print("== Invalid --loss argument")

        ETHEC_trainer = ETHECExperiment(data_loaders=data_loaders, labelmap=labelmap,
                                        criterion=use_criterion,
                                        lr=arguments['lr'],
                                        batch_size=batch_size, evaluator=eval_type,
                                        experiment_name=arguments['experiment_name'],  # 'cifar_test_ft_multi',
                                        experiment_dir=arguments['experiment_dir'],
                                        eval_interval=arguments['eval_interval'],
                                        n_epochs=arguments['n_epochs'],
                                        feature_extracting=arguments['freeze_weights'],
                                        use_pretrained=True,
                                        load_wt=False,
                                        model_name=arguments['model'],
                                        optimizer_method=arguments['optimizer_method'],
                                        use_grayscale=arguments['use_grayscale'])
        ETHEC_trainer.prepare_model(loading=True)
        ETHEC_trainer.set_optimizer()

        self.ETHEC_trainer = ETHEC_trainer
        self.path_to_exp = path_to_exp
        self.test_set, self.val_set, self.train_set = test_set, val_set, train_set
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.labelmap = labelmap
        self.model_name=arguments['model']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.viz_these_samples_ix = data_ix
        if not data_ix:
            self.viz_these_samples_ix = list(range(len(self.test_set))) # [231, 890] # 890

        if perform_inference:
            if mode == 'lime':
                self.run_LIME()
            elif mode == 'tsne':
                self.run_tsne()
            else:
                print('Invalid option: {}'.format(mode))

    def get_model(self):
        self.ETHEC_trainer.load_best_model(only_load=True)
        return self.ETHEC_trainer.model

    def run_tsne(self):
        self.ETHEC_trainer.load_best_model(only_load=True)

        outputs = []
        def hook(module, input, output):
            outputs.append(input)

        # modify last layers based on the model being used
        if self.model_name in ['alexnet', 'vgg']:
            self.ETHEC_trainer.model.module.classifier[6].register_forward_hook(hook)
        elif 'resnet' in self.model_name:
            print(self.ETHEC_trainer.model)
            self.ETHEC_trainer.model.module.fc.register_forward_hook(hook)

        for set_name in ['train', 'test', 'val']:
            chosen_set = getattr(self, '{}_set'.format(set_name))
            sample_ix = list(range(len(chosen_set)))
            print(sample_ix)
            level_labels_array, representations = [], []
            testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(chosen_set, sample_ix),
                                                 batch_size=1,
                                                 shuffle=False, num_workers=0)

            for index, data_item in enumerate(testloader):
                inputs, labels, level_labels = data_item['image'], data_item['labels'], data_item['level_labels']
                self.ETHEC_trainer.model(inputs.to(self.device))
                level_labels_array.append(level_labels[0, :].detach().cpu().numpy())
                for j in range(len(outputs)):
                    # print(outputs[0][0].shape, outputs[1][0].shape)
                    for i in range(outputs[j][0].shape[0]):
                        # print(outputs[0][0].data[i, :].numpy())
                        # print(level_labels[i, :])
                        representations.append(outputs[j][0].detach().data[i, :].cpu().numpy())
                outputs = []

            path_to_embeddings = os.path.join(self.path_to_exp, 'embeddings')
            if not os.path.exists(path_to_embeddings):
                os.makedirs(path_to_embeddings)

            np.save(os.path.join(path_to_embeddings, '{}_representations.npy'.format(set_name)), np.array(representations))
            np.save(os.path.join(path_to_embeddings, '{}_level_labels.npy'.format(set_name)), np.array(level_labels_array))

    def run_LIME(self):

        tforms = transforms.Compose([# transforms.ToPILImage(),
                            # transforms.Resize((input_size, input_size)),
                            transforms.CenterCrop(224)
                            # transforms.ToTensor()
                            ])
        to_tensor_tform = transforms.Compose([
                            transforms.ToTensor()
                            ])

        def get_image(path):
            with open(os.path.abspath(path), 'rb') as f:
                with Image.open(f) as img:
                    return img.convert('RGB')

        def get_input_tensors(img):
            # unsqeeze converts single image to batch of 1
            retval = tforms(img)
            return retval #.unsqueeze(0)

        self.ETHEC_trainer.load_best_model()

        explainer = lime_image.LimeImageExplainer()

        level_id = 1
        item = None

        save_images_in = os.path.join(self.path_to_exp, 'analysis')
        if not os.path.exists(save_images_in):
             os.makedirs(os.path.join(self.path_to_exp, 'analysis'))

        summarizer = Summarize(save_images_in)

        for sample_ix in self.viz_these_samples_ix:
            testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(self.test_set, [sample_ix]),
                                                     batch_size=self.batch_size,
                                                     shuffle=False, num_workers=self.n_workers)

            for index, item in enumerate(testloader):
                print(os.path.join(save_images_in, str(sample_ix)))
                if not os.path.exists(os.path.join(save_images_in, str(sample_ix))):
                    os.makedirs(os.path.join(save_images_in, str(sample_ix)))
                print(item['path_to_image'])
                print(item['level_labels'])

                img = get_image(item['path_to_image'][0])
                print(item['level_labels'])
                print('=' * 30 + 'Ground truth' + '=' * 30)
                print(item['level_labels'][0][0].data.item(),
                      self.labelmap.family_ix_to_str[item['level_labels'][0][0].data.item()])
                print(item['level_labels'][0][1].data.item(),
                      self.labelmap.subfamily_ix_to_str[item['level_labels'][0][1].data.item()])
                print(item['level_labels'][0][2].data.item(),
                      self.labelmap.genus_ix_to_str[item['level_labels'][0][2].data.item()])
                print(item['level_labels'][0][3].data.item(),
                      self.labelmap.genus_specific_epithet_ix_to_str[item['level_labels'][0][3].data.item()])

                summarizer.make_heading('Data ID: {}'.format(sample_ix), heading_level=2)
                summarizer.make_heading('Ground truth', heading_level=3)
                summarizer.make_text(text='{} {}'.format(item['level_labels'][0][0].data.item(),
                                                         self.labelmap.family_ix_to_str[item['level_labels'][0][0].data.item()]),
                                     bullet=False)
                summarizer.make_text(text='{} {}'.format(item['level_labels'][0][1].data.item(),
                                                         self.labelmap.subfamily_ix_to_str[item['level_labels'][0][1].data.item()]),
                                     bullet=False)
                summarizer.make_text(text='{} {}'.format(item['level_labels'][0][2].data.item(),
                                                         self.labelmap.genus_ix_to_str[item['level_labels'][0][2].data.item()]),
                                     bullet=False)
                summarizer.make_text(text='{} {}'.format(item['level_labels'][0][3].data.item(),
                                                         self.labelmap.genus_specific_epithet_ix_to_str[item['level_labels'][0][3].data.item()]),
                                     bullet=False)
                summarizer.make_hrule()

                for level_id in range(4):

                    def batch_predict(images):
                        item['image'] = torch.stack(tuple(to_tensor_tform(i) for i in images), dim=0)
                        item['level_labels'] = torch.stack(tuple(item['level_labels'][0] for i in range(images.shape[0])), dim=0)
                        item['labels'] = torch.stack(tuple(item['labels'][0] for i in range(images.shape[0])), dim=0)

                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                        # print(item)
                        logits = self.ETHEC_trainer.inference(item)
                        probs = torch.nn.functional.softmax(logits[:, self.labelmap.level_start[level_id]:self.labelmap.level_stop[level_id]],
                                                            dim=1)
                        return probs.detach().cpu().numpy()


                    tmp = time.time()
                    explanation = explainer.explain_instance(np.array(get_input_tensors(img)),
                                                             batch_predict,  # classification function
                                                             top_labels=5,
                                                             hide_color=0,
                                                             num_samples=100, # number of images that will be sent to classification function
                                                             )
                    # print('time taken: {}'.format(time.time() - tmp))
                    summarizer.make_heading('{} - Top 5'.format(self.labelmap.level_names[level_id]), heading_level=3)
                    summarizer.make_text('time taken: {}'.format(time.time() - tmp))

                    x_labels, data = [], []
                    for i in range(5):
                        temp, mask = explanation.get_image_and_mask(explanation.top_labels[i], positive_only=False, num_features=10,
                                                                    hide_rest=False)
                        print('Predicted: {} {}'.format(explanation.top_labels[i], getattr(self.labelmap, '{}_ix_to_str'.format(self.labelmap.level_names[level_id]))[explanation.top_labels[i]]))
                        # summarizer.make_text('Predicted: {} {}'.format(explanation.top_labels[i], getattr(self.labelmap, '{}_ix_to_str'.format(self.labelmap.level_names[level_id]))[explanation.top_labels[i]]))
                        x_labels.append('<span style="color:{}">Predicted: {} {}</span>'.format('green' if explanation.top_labels[i] == item['level_labels'][0][level_id].data.item() else 'red', explanation.top_labels[i], getattr(self.labelmap, '{}_ix_to_str'.format(self.labelmap.level_names[level_id]))[explanation.top_labels[i]]))
                        img_boundry1 = mark_boundaries(temp / 255.0, mask)
                        save_img_to = os.path.join(save_images_in, str(sample_ix), '{}_{}.png'.format(level_id, i))
                        mpimg.imsave(save_img_to, img_boundry1)
                        # summarizer.make_image(location=save_img_to, alt_text='.')
                        data.append('![{}]({})'.format('text', os.path.relpath(save_img_to, save_images_in)))

                    summarizer.make_table(data=[data], x_labels=x_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_exp", help='Location where experiment is stored.', type=str, required=True)
    parser.add_argument("--image_dir", help='Path to images.', type=str, default=None)
    parser.add_argument("--mode", help='[lime, tsne]', type=str, required=True)
    args = parser.parse_args()

    Inference(args.path_to_exp, args.mode, args.image_dir)
