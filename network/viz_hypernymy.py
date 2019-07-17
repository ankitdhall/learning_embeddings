from __future__ import print_function
from __future__ import division
from torchvision import transforms

import os

from data.db import ETHECLabelMapMerged
from tqdm import tqdm

import torch
from torch import nn
from data.db import ETHECLabelMapMergedSmall

import numpy as np
import random
random.seed(0)

import networkx as nx

import math
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection

from network.oe import Embedder, FeatNet
from network.oe import load_combined_graphs, EuclideanConesWithImagesHypernymLoss, OrderEmbeddingWithImagesHypernymLoss
from network.oe import my_collate, ETHECHierarchyWithImages

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt


class VizualizeGraphRepresentation:
    def __init__(self, debug=False,
                 dim=2, loss_fn='ec', title_text='',
                 # weights_to_load='/home/ankit/learning_embeddings/exp/ethec_debug/ec_debug/d10/oe10d_debug/weights/best_model.pth'):
                 weights_to_load='/home/ankit/Desktop/emb_weights/joint_2xlr/best_model_model.pth'):
        torch.manual_seed(0)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.title_text = title_text

        labelmap = ETHECLabelMapMerged()
        if debug:
            labelmap = ETHECLabelMapMergedSmall()
            path_to_folder = '../database/ETHEC/ETHECSmall_embeddings/graphs'
        else:
            path_to_folder = '../database/ETHEC/ETHEC_embeddings/graphs'

        G = nx.read_gpickle(os.path.join(path_to_folder, 'G'))

        self.G = G
        self.G_tc = nx.transitive_closure(self.G)
        self.labelmap = labelmap

        if loss_fn == 'ec':
            self.model = Embedder(embedding_dim=dim, labelmap=labelmap, normalize=False, K=3.0)
        elif loss_fn == 'oe':
            self.model = Embedder(embedding_dim=dim, labelmap=labelmap, normalize=False)#, K=3.0)
        self.model =nn.DataParallel(self.model)
        self.loss_fn = loss_fn

        self.weights_to_load = weights_to_load
        self.load_model()

        self.title_text = title_text
        if self.title_text == '':
            self.title_text = 'F1 score: {:.4f} Accuracy: {:.4f} \n Precision: {:.4f} Recall: {:.4f} | Threshold: {:.4f}'.format(
                self.reconstruction_f1, self.reconstruction_accuracy, self.reconstruction_prec,
                self.reconstruction_recall, self.reconstruction_threshold)

        # run vizualize
        self.vizualize()

    def load_model(self):
        checkpoint = torch.load(self.weights_to_load,
                                map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.epoch = checkpoint['epoch']
        self.optimal_threshold = checkpoint['optimal_threshold']

        self.reconstruction_f1, self.reconstruction_threshold, self.reconstruction_accuracy, self.reconstruction_prec, self.reconstruction_recall = \
            checkpoint['reconstruction_scores']['f1'], checkpoint['reconstruction_scores']['threshold'], \
            checkpoint['reconstruction_scores']['accuracy'], checkpoint['reconstruction_scores']['precision'], \
            checkpoint['reconstruction_scores']['recall']

        print('Using optimal threshold = {}'.format(self.optimal_threshold))
        print('Successfully loaded model and img_feat_net epoch {} from {}'.format(self.epoch, self.weights_to_load))

    def get_wedge(self, emb, radius=30):
        if self.loss_fn == 'ec':
            psi_x = math.asin((3 / math.sqrt(emb[0] ** 2 + emb[1] ** 2))) * 57.2958
            angle_of_vec = math.atan2(emb[1], emb[0]) * 57.2958
            patches = [Wedge((emb[0], emb[1]), radius,
                             angle_of_vec - psi_x, angle_of_vec + psi_x, color='k')]
        else:
            patches = [Wedge((emb[0], emb[1]), radius, 0, 90, color='k')]
        p = PatchCollection(patches, alpha=0.1)
        return p

    def vizualize(self, save_to_disk=True, filename='embeddings'):
        phase = 'test'
        self.model.eval()

        colors = ['c', 'm', 'y', 'k']
        embeddings_x, embeddings_y, annotation, color_list = {}, {}, {}, {}

        connected_to = {}

        fig, ax = plt.subplots()

        for level_id in range(3): #len(self.labelmap.levels)):
            level_start, level_stop = self.labelmap.level_start[level_id], self.labelmap.level_stop[level_id]
            level_color = colors[level_id]
            for label_ix in range(self.labelmap.levels[level_id]):
                emb_id = label_ix + level_start
                emb = self.model(torch.tensor([emb_id]).to(self.device)).cpu().detach()
                emb = emb[0].numpy()

                embeddings_x[emb_id] = emb[0]
                embeddings_y[emb_id] = emb[1]
                annotation[emb_id] = '{}'.format(getattr(self.labelmap,
                                                      '{}_ix_to_str'.format(self.labelmap.level_names[level_id]))[label_ix]
                                                 )
                color_list[emb_id] = level_color

                connected_to[emb_id] = [v for u, v in list(self.G.edges(emb_id))]

                if level_id == 3:
                    ax.scatter(emb[0], emb[1], c=level_color, alpha=0.5, linewidth='0')
                else:
                    ax.scatter(emb[0], emb[1], c=level_color, alpha=1)
                # ax.annotate(annotation[emb_id], (emb[0], emb[1]))

                if level_id in [0, 1]:
                    p = self.get_wedge(emb, radius=50)
                    ax.add_collection(p)


        # fig, ax = plt.subplots()
        # ax.scatter(embeddings_x, embeddings_y, c=color_list)

        for from_node in connected_to:
            for to_node in connected_to[from_node]:
                if to_node in embeddings_x:
                    plt.plot([embeddings_x[from_node], embeddings_x[to_node]], [embeddings_y[from_node], embeddings_y[to_node]],
                             'b-', alpha=0.2)

        # for i, txt in enumerate(annotation):
        #     ax.annotate(txt, (embeddings_x[i], embeddings_y[i]))

        ax.axis('equal')
        if self.title_text:
            fig.suptitle(self.title_text, family='sans-serif')
        if save_to_disk:
            fig.set_size_inches(8, 7)
            fig.savefig(os.path.join(os.path.dirname(self.weights_to_load), '..', '{}.pdf'.format(filename)), dpi=200)
            fig.savefig(os.path.join(os.path.dirname(self.weights_to_load), '..', '{}.png'.format(filename)), dpi=200)

        return ax


class VizualizeGraphRepresentationWithImages:
    def __init__(self, debug=False,
                 dim=2,
                 loss_fn='oe',
                 weights_to_load='/home/ankit/learning_embeddings/exp/ethec_debug/oelwi_debug/ec_load_emb_lr_0.01/weights/best_model_model.pth',
                 img_weights_to_load='/home/ankit/learning_embeddings/exp/ethec_debug/oelwi_debug/ec_load_emb_lr_0.01/weights/best_model_img_feat_net.pth'):
        torch.manual_seed(0)
        self.load_split = 'train'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        labelmap = ETHECLabelMapMerged()
        if debug:
            labelmap = ETHECLabelMapMergedSmall()

        combined_graphs = load_combined_graphs(debug)

        if debug:
            image_fc7 = np.load('../database/ETHEC/ETHECSmall_embeddings/train.npy')[()]
            image_fc7.update(np.load('../database/ETHEC/ETHECSmall_embeddings/val.npy')[()])
            image_fc7.update(np.load('../database/ETHEC/ETHECSmall_embeddings/test.npy')[()])
        else:
            image_fc7 = np.load('../database/ETHEC/ETHEC_embeddings/train.npy')[()]
            image_fc7.update(np.load('../database/ETHEC/ETHEC_embeddings/val.npy')[()])
            image_fc7.update(np.load('../database/ETHEC/ETHEC_embeddings/test.npy')[()])

        self.G = combined_graphs['graph']
        self.G_tc = nx.transitive_closure(self.G)
        self.labelmap = labelmap

        if loss_fn == 'ec':
            self.model = Embedder(embedding_dim=dim, labelmap=labelmap, normalize=False, K=3.0)
            self.img_feat_net = FeatNet(output_dim=2, normalize=None,
                                        K=3).to(self.device)
            self.criterion = EuclideanConesWithImagesHypernymLoss(labelmap=labelmap,
                                                                 neg_to_pos_ratio=5,
                                                                 feature_dict=image_fc7,
                                                                 alpha=0.01,
                                                                 pick_per_level=True,
                                                                 use_CNN=False)

        else:
            self.model = Embedder(embedding_dim=dim, labelmap=labelmap, normalize=False)
            self.img_feat_net = FeatNet(output_dim=2, normalize=None,
                                        K=None).to(self.device)
            self.criterion = OrderEmbeddingWithImagesHypernymLoss(labelmap=labelmap,
                                                                 neg_to_pos_ratio=5,
                                                                 feature_dict=image_fc7,
                                                                 alpha=1.0,
                                                                 pick_per_level=True,
                                                                 use_CNN=False)

        self.model = nn.DataParallel(self.model)
        self.img_feat_net = nn.DataParallel(self.img_feat_net)
        self.graph_dict = combined_graphs

        self.load_model(weights_to_load, img_weights_to_load)

        # prepare model
        self.testloader = self.create_loader()
        self.img_to_emb = dict()
        self.pass_samples()

        # run vizualize
        self.vizualize()

    def load_model(self, weights_to_load, img_weights_to_load):
        checkpoint = torch.load(weights_to_load, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.epoch = checkpoint['epoch']
        self.optimal_threshold = checkpoint['optimal_threshold']
        checkpoint = torch.load(img_weights_to_load, map_location=self.device)
        self.img_feat_net.load_state_dict(checkpoint['model_state_dict'])
        print('Using optimal threshold = {}'.format(self.optimal_threshold))
        print('Successfully loaded model and img_feat_net epoch {} from {}'.format(self.epoch, weights_to_load))

    def create_loader(self):
        val_test_data_transforms = transforms.Compose([transforms.ToPILImage(),
                                                       transforms.Resize((224, 224)),
                                                       transforms.ToTensor(),
                                                       ])
        test_set = ETHECHierarchyWithImages(self.graph_dict['G_{}'.format(self.load_split)],
                                            imageless_dataloaders=None,
                                            transform=val_test_data_transforms)

        testloader = torch.utils.data.DataLoader(test_set, collate_fn=my_collate,
                                                 batch_size=10,
                                                 num_workers=4,
                                                 shuffle=False)
        return testloader

    def pass_samples(self):
        self.img_feat_net = self.img_feat_net.to(self.device)
        for index, data_item in enumerate(tqdm(self.testloader)):
            inputs_from, inputs_to, status = data_item['from'], data_item['to'], data_item['status']

            self.get_from_and_to_emb(inputs_to)

    def get_from_and_to_emb(self, to_elem):
        # get embeddings for concepts and images
        # do the same for to embeddings
        image_elem = [elem for ix, elem in enumerate(to_elem) if type(elem) == str]
        image_ix = [ix for ix, elem in enumerate(to_elem) if type(elem) == str]

        if len(image_elem) != 0:
            image_emb = self.img_feat_net(self.criterion.get_img_features(image_elem).to(self.device))
            to_embeddings = torch.zeros((len(to_elem), image_emb.shape[-1])).to(self.device)

        if len(image_elem) != 0:
            to_embeddings[image_ix, :] = image_emb

        to_embeddings = to_embeddings.cpu().detach().numpy()
        for i in range(len(to_elem)):
            self.img_to_emb[to_elem[i]] = to_embeddings[i, :]


    def vizualize(self):
        phase = 'test'
        self.model.eval()

        colors = ['c', 'm', 'y', 'k']
        embeddings_x, embeddings_y, annotation, color_list = {}, {}, {}, {}

        connected_to = {}

        fig, ax = plt.subplots()

        for level_id in range(4): #len(self.labelmap.levels)):
            level_start, level_stop = self.labelmap.level_start[level_id], self.labelmap.level_stop[level_id]
            level_color = colors[level_id]
            for label_ix in range(self.labelmap.levels[level_id]):
                emb_id = label_ix + level_start
                emb = self.model(torch.tensor([emb_id]).to(self.device)).cpu().detach()
                emb = emb[0].numpy()

                embeddings_x[emb_id] = emb[0]
                embeddings_y[emb_id] = emb[1]
                annotation[emb_id] = '{}'.format(getattr(self.labelmap,
                                                      '{}_ix_to_str'.format(self.labelmap.level_names[level_id]))[label_ix]
                                                 )
                color_list[emb_id] = level_color

                connected_to[emb_id] = [v for u, v in list(self.G.edges(emb_id))]

                ax.scatter(emb[0], emb[1], c=level_color, alpha=1)

        for from_node in connected_to:
            for to_node in connected_to[from_node]:
                if to_node in embeddings_x:
                    plt.plot([embeddings_x[from_node], embeddings_x[to_node]], [embeddings_y[from_node], embeddings_y[to_node]],
                             'b-', alpha=0.2)

        for key in self.img_to_emb:
            emb = self.img_to_emb[key]
            ax.scatter(emb[0], emb[1], c=level_color, alpha=0.3)

            from_ix = max([u for u, v in list(self.graph_dict['G_{}'.format(self.load_split)].in_edges(key))])

            if from_ix in embeddings_x:
                plt.plot([embeddings_x[from_ix], emb[0]],
                         [embeddings_y[from_ix], emb[1]],
                         'b-', alpha=0.2)


        plt.show()


def create_images():
    path_to_weights = '/home/ankit/Desktop/d2_bs10/oe_n_10_a_0.1_lr_0.1/weights'
    loss_fn = 'oe'
    files = os.listdir(path_to_weights)
    files.sort()
    for filename in files:
        if 'best_model' in filename:
            continue
        viz = VizualizeGraphRepresentation(debug=False, dim=2, loss_fn=loss_fn, title_text='',
                                           weights_to_load=os.path.join(path_to_weights, filename))
        viz.vizualize(save_to_disk=True, filename='{0:04d}'.format(int(filename[:-4])))
        plt.close('all')


if __name__ == '__main__':
    # obj = VizualizeGraphRepresentation(debug=False)
    # obj = VizualizeGraphRepresentationWithImages(debug=True)
    create_images()
