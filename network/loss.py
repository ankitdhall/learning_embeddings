import torch
from torch import nn
from data.db import ETHECLabelMap, ETHECLabelMapMergedSmall

class MultiLevelCELoss(torch.nn.Module):
    def __init__(self, labelmap, level_weights=None, weight=None):
        torch.nn.Module.__init__(self)
        self.labelmap = labelmap
        self.level_weights = [1.0] * len(self.labelmap.levels) if level_weights is None else level_weights
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = []
        if weight is None:
            for level_len in self.labelmap.levels:
                self.criterion.append(nn.CrossEntropyLoss(weight=None, reduction='none'))
        else:
            level_stop, level_start = [], []
            for level_id, level_len in enumerate(self.labelmap.levels):
                if level_id == 0:
                    level_start.append(0)
                    level_stop.append(level_len)
                else:
                    level_start.append(level_stop[level_id - 1])
                    level_stop.append(level_stop[level_id - 1] + level_len)
                self.criterion.append(nn.CrossEntropyLoss(weight=weight[level_start[level_id]:level_stop[level_id]].to(self.device),
                                                          reduction='none'))

        print('==Using the following weights config for multi level cross entropy loss: {}'.format(self.level_weights))

    def forward(self, outputs, labels, level_labels):
        loss = 0.0
        for level_id, level in enumerate(self.labelmap.levels):
            if level_id == 0:
                loss += self.level_weights[level_id] * self.criterion[level_id](outputs[:, 0:level], level_labels[:, level_id])
            else:
                start = sum([self.labelmap.levels[l_id] for l_id in range(level_id)])
                loss += self.level_weights[level_id] * self.criterion[level_id](outputs[:, start:start + level],
                                                                          level_labels[:, level_id])
        return torch.mean(loss)


class LastLevelCELoss(torch.nn.Module):
    def __init__(self, labelmap, level_weights=None, weight=None):
        torch.nn.Module.__init__(self)
        self.labelmap = labelmap
        self.level_weights = [1.0] * len(self.labelmap.levels) if level_weights is None else level_weights
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = []
        self.softmax = torch.nn.Softmax(dim=1)

        self.level_stop, self.level_start = [], []
        for level_id, level_len in enumerate(self.labelmap.levels):
            if level_id == 0:
                self.level_start.append(0)
                self.level_stop.append(level_len)
            else:
                self.level_start.append(self.level_stop[level_id - 1])
                self.level_stop.append(self.level_stop[level_id - 1] + level_len)

        if weight is None:
            for level_len in self.labelmap.levels:
                self.criterion.append(nn.NLLLoss(weight=None, reduction='none'))
        else:
            self.criterion.append(nn.NLLLoss(weight=weight[self.level_start[level_id]:self.level_stop[level_id]].to(self.device),
                                                      reduction='none'))

        print('==Using the following weights config for last level cross entropy loss: {}'.format(self.level_weights))

    def forward(self, outputs, labels, level_labels):
        # print(outputs)
        # print(level_labels)
        outputs_new = torch.zeros((outputs.shape[0], self.labelmap.n_classes)).to(self.device)
        # print("outputs_new", outputs_new)
        outputs_new[:, self.level_start[-1]:self.level_stop[-1]] = self.softmax(outputs[:, :])
        # print("outputs_new", outputs_new)
        for level_index in range(len(self.labelmap.levels)-2, -1, -1):
            # print("--"*30)
            # print("level_index: {}, level len: {}".format(level_index, self.labelmap.levels[level_index]))
            # print("getting child of: {}".format(self.labelmap.level_names[level_index]))
            child_of = getattr(self.labelmap, "child_of_{}_ix".format(self.labelmap.level_names[level_index]))
            for parent_ix in child_of:
                # print("==== parent_ix: {} children: {}".format(parent_ix, child_of[parent_ix]))
                # print("will sum these: {}".format(outputs_new[:, self.level_start[level_index+1]+torch.tensor(child_of[parent_ix])]))
                # print("sum: {}".format(torch.sum(outputs_new[:, self.level_start[level_index+1]+torch.tensor(child_of[parent_ix])], dim=1)))
                outputs_new[:, self.level_start[level_index]+torch.tensor(parent_ix)] = \
                    torch.sum(outputs_new[:, self.level_start[level_index+1]+torch.tensor(child_of[parent_ix])], dim=1)
                # print("outputs_new", outputs_new)

        loss = 0.0
        for level_id, level in enumerate(self.labelmap.levels):
            if level_id == 0:
                # print("outputs level {}: {}".format(level_id, outputs_new[:, 0:level]))
                loss += self.level_weights[level_id] * self.criterion[level_id](torch.log(outputs_new[:, 0:level]), level_labels[:, level_id])
            else:
                start = sum([self.labelmap.levels[l_id] for l_id in range(level_id)])
                # print("outputs level {}: {}".format(level_id, outputs_new[:, start:start + level]))
                loss += self.level_weights[level_id] * self.criterion[level_id](torch.log(outputs_new[:, start:start + level]),
                                                                          level_labels[:, level_id])
        return outputs_new, torch.mean(loss)


class MaskedCELoss(torch.nn.Module):
    def __init__(self, labelmap, level_weights=None):
        torch.nn.Module.__init__(self)
        self.labelmap = labelmap
        self.level_weights = [1.0] * len(self.labelmap.levels) if level_weights is None else level_weights
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = []

        self.level_stop, self.level_start = [], []
        for level_id, level_len in enumerate(self.labelmap.levels):
            if level_id == 0:
                self.level_start.append(0)
                self.level_stop.append(level_len)
            else:
                self.level_start.append(self.level_stop[level_id - 1])
                self.level_stop.append(self.level_stop[level_id - 1] + level_len)

        for level_len in self.labelmap.levels:
            self.criterion.append(nn.CrossEntropyLoss(weight=None, reduction='none'))

        print('==Using the following weights config for masked cross entropy loss: {}'.format(self.level_weights))

    def forward(self, outputs, labels, level_labels, phase='train'):
        outputs_new = -100000000*torch.ones_like(outputs).to(self.device)
        loss = 0.0
        labels_at_eval = []
        # print('outputs shape in loss.py:', outputs.shape)
        # print('level_labels shape in loss.py:', level_labels.shape)
        for sample_id in range(outputs.shape[0]):
            possible_children_dict_orig, new_level_labels = self.labelmap.decode_children(level_labels[sample_id, :])
            found_incorrect_prediction = False
            # print(possible_children_dict_orig, new_level_labels)
            # print(outputs)
            possible_children_dict = {}
            for level_id, k in enumerate(possible_children_dict_orig):
                possible_children_dict[k] = [ix+self.level_start[level_id] for ix in possible_children_dict_orig[k]]
                # print(outputs[sample_id, possible_children_dict[k]].unsqueeze(0), torch.tensor([new_level_labels[level_id]]))
                # print(outputs[sample_id, possible_children_dict[k]].unsqueeze(0).shape, torch.tensor([new_level_labels[level_id]]).shape)
                if not found_incorrect_prediction:
                    loss += self.level_weights[level_id] * self.criterion[level_id](outputs[sample_id, possible_children_dict[k]].unsqueeze(0), torch.tensor([new_level_labels[level_id]]).to(self.device))
                else:
                    loss += self.level_weights[level_id] * self.criterion[level_id](outputs[sample_id, self.level_start[level_id]:self.level_stop[level_id]].unsqueeze(0), torch.tensor([level_labels[sample_id, level_id]]).to(self.device))
                # if phase == 'train':
                #     outputs_new[sample_id, possible_children_dict[k]] = outputs[sample_id, possible_children_dict[k]]
                # else:
                if level_id == 0:
                    predicted_class = torch.argmax(outputs[sample_id, possible_children_dict[k]])
                    outputs_new[sample_id, possible_children_dict[k]] = outputs[sample_id, possible_children_dict[k]]
                    predicted_class_absolute = predicted_class.item()
                else:
                    children_of_prev_level_pred_relative = self.labelmap.get_children_of(predicted_class_absolute-self.level_start[level_id-1], level_id)
                    children_of_prev_level_pred_absolute = [ix+self.level_start[level_id] for ix in children_of_prev_level_pred_relative]
                    predicted_class = torch.argmax(outputs[sample_id, children_of_prev_level_pred_absolute])
                    predicted_class_absolute = children_of_prev_level_pred_absolute[predicted_class]
                    outputs_new[sample_id, children_of_prev_level_pred_absolute] = outputs[sample_id, children_of_prev_level_pred_absolute]
                    # labels_at_eval.append(predicted_class)
                    # outputs_new[sample_id, possible_children_dict[k]] = outputs[sample_id, possible_children_dict[k]]

                if predicted_class != new_level_labels[level_id]:
                    found_incorrect_prediction = True



                # if torch.argmax(outputs[sample_id, possible_children_dict[k]]) != new_level_labels[level_id]:
                #     break
        return outputs_new, torch.mean(loss)


class MultiLabelSMLoss(torch.nn.MultiLabelSoftMarginLoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        print(weight)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if weight is not None:
            weight = weight.to(self.device)
        torch.nn.MultiLabelSoftMarginLoss.__init__(self, weight, size_average, reduce, reduction)

    def forward(self, outputs, labels, level_labels):
        return super().forward(outputs, labels)


class HierarchicalSoftmax(torch.nn.Module):
    def __init__(self, labelmap, input_size, level_weights=None):
        torch.nn.Module.__init__(self)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labelmap = labelmap

        self.level_stop, self.level_start = [], []
        for level_id, level_len in enumerate(self.labelmap.levels):
            if level_id == 0:
                self.level_start.append(0)
                self.level_stop.append(level_len)
            else:
                self.level_start.append(self.level_stop[level_id - 1])
                self.level_stop.append(self.level_stop[level_id - 1] + level_len)

        self.module_dict = {}
        for level_id, level_name in enumerate(self.labelmap.level_names):
            if level_id == 0:
                self.module_dict[level_name] = nn.Linear(input_size, self.labelmap.levels[0])

            # setup linear layer for current nodes which are children of level_id-1
            else:
                child_of_l_1 = getattr(self.labelmap, 'child_of_{}_ix'.format(self.labelmap.level_names[level_id-1]))
                for parent_id in child_of_l_1:
                    self.module_dict['{}_{}'.format(level_name, parent_id)] = nn.Linear(input_size, len(child_of_l_1[parent_id]))

        self.module_dict = nn.ModuleDict(self.module_dict)
        print(self.module_dict)

    def forward(self, x):
        """
        Takes input from the penultimate layer of the model and uses the HierarchicalSoftmax layer in the end to compute
        the logits.
        :param x: <torch.tensor> output of the penultimate layer
        :return: all_log_probs <torch.tensor>, last level log_probs <torch.tensor>
        """
        all_log_probs = torch.zeros((x.shape[0], self.labelmap.n_classes)).to(self.device)

        for level_id, level_name in enumerate(self.labelmap.level_names):
            # print(all_log_probs)
            if level_id == 0:
                # print(level_name)
                # print("saving log probs for: {}:{}".format(self.level_start[0], self.level_stop[0]))
                all_log_probs[:, self.level_start[0]:self.level_stop[0]] = torch.nn.functional.log_softmax(self.module_dict[level_name](x), dim=1)

            # setup linear layer for current nodes which are children of level_id-1
            else:
                child_of_l_1 = getattr(self.labelmap, 'child_of_{}_ix'.format(self.labelmap.level_names[level_id-1]))
                # print(child_of_l_1)
                for parent_id in child_of_l_1:
                    # print('child_of_{}_ix'.format(self.labelmap.level_names[level_id - 1]),
                    #       '{}_{}'.format(level_name, parent_id))
                    # print("saving log probs for: {1} -> {0}".format(self.level_start[level_id] + torch.tensor(child_of_l_1[parent_id]), torch.tensor(child_of_l_1[parent_id])))
                    log_probs = torch.nn.functional.log_softmax(self.module_dict['{}_{}'.format(level_name, parent_id)](x), dim=1)
                    # print("{0} + {1} = {2}".format(log_probs, all_log_probs[:, self.level_start[level_id-1] + parent_id].unsqueeze(1), log_probs + all_log_probs[:, self.level_start[level_id-1] + parent_id].unsqueeze(1)))
                    all_log_probs[:, self.level_start[level_id] + torch.tensor(child_of_l_1[parent_id]).to(self.device)] = log_probs + all_log_probs[:, self.level_start[level_id-1] + parent_id].unsqueeze(1)

        # return only leaf probs
        # print(all_log_probs)
        return all_log_probs, all_log_probs[:, self.level_start[-1]:self.level_stop[-1]]


class HierarchicalSoftmaxLoss(torch.nn.Module):
    def __init__(self, labelmap, level_weights=None):
        torch.nn.Module.__init__(self)
        self.labelmap = labelmap
        self.criterion = torch.nn.NLLLoss()

    def forward(self, outputs, labels, level_labels):
        return self.criterion(outputs, level_labels[:, -1])


if __name__ == '__main__':
    lmap = ETHECLabelMap()
    criterion = MultiLevelCELoss(labelmap=lmap, level_weights=[1, 1, 1, 1])
    output, level_labels = torch.zeros((1, lmap.n_classes)), torch.tensor([[0,
                                                                            7-lmap.levels[0],
                                                                            90-(lmap.levels[0]+lmap.levels[1]),
                                                                            400-(lmap.levels[0]+lmap.levels[1]+lmap.levels[2])]])
    labels = torch.zeros((1, lmap.n_classes))
    labels[0, torch.tensor([0, 7, 90, 400])] = 1
    output[:, 0] = 100
    output[:, 7] = 100
    output[:, 90] = 10000
    output[:, 400] = 10000
    print(output)
    print(labels)
    print(level_labels)
    print('MLCELoss: {}'.format(criterion(output, labels, level_labels)))

    criterion_multi_label = torch.nn.MultiLabelSoftMarginLoss()
    custom_criterion_multi_label = MultiLabelSMLoss()
    print('MLSMLoss: {}'.format(criterion_multi_label(output, labels)))
    print('MLSMLoss: {}'.format(custom_criterion_multi_label(output, labels, level_labels)))

    # print("="*30)
    # torch.manual_seed(0)
    #
    # lmap = ETHECLabelMapMergedSmall()
    # criterion = LastLevelCELoss(labelmap=lmap)
    # print("Labelmap levels: {}".format(lmap.levels))
    #
    # outputs, level_labels = torch.rand((2, lmap.levels[-1])), torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]])
    #
    # print(criterion(outputs, None, level_labels))

    # print("=" * 30, "Masked CE", "=" * 30)
    # torch.manual_seed(0)
    #
    # lmap = ETHECLabelMapMergedSmall()
    # criterion = LastLevelCELoss(labelmap=lmap)
    # print("Labelmap levels: {}".format(lmap.levels))
    #
    # outputs, level_labels = torch.rand((2, lmap.n_classes)), torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]])
    # outputs[0, 0] = 1.0
    #
    # criterion = MaskedCELoss(labelmap=lmap)
    #
    # print(criterion(outputs, None, level_labels))

    print("=" * 30, "Hierarchical softmax", "=" * 30)
    torch.manual_seed(0)

    lmap = ETHECLabelMapMergedSmall()
    print("Labelmap levels: {}".format(lmap.levels))

    hsoftmax = HierarchicalSoftmax(labelmap=lmap, input_size=4, level_weights=None)
    penult_layer = torch.tensor([[1, 2, 1, 2.0], [1, 10, -7, 10], [1, 9, 1, -2]])
    print(hsoftmax(penult_layer))


